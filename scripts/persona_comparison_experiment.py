#!/usr/bin/env python
"""Cold-item baselines and validation-only statistical/LLM hybrids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from responsevec.baselines.individual_choice import SklearnChoiceBaseline
from responsevec.encode import load_option_table
from responsevec.persona_comparison import (build_choice_arrays, fit_lambda,
    fit_position_prior, fit_temperature, inductive_pca_option_vectors,
    log_opinion_pool, predict_position_prior, respondent_split, strict_align,
    temperature_scale)
from responsevec.persona_evaluation import metrics, respondent_bootstrap_nll_gap
from responsevec.persona_head import fit_persona_head, persona_conditionals
from responsevec.persona_router import (fit_demographic_router, fold_roles,
    router_prior, stable_history, update_posterior)


def load_predictions(path, method, k, rows):
    frame = pd.read_parquet(path)
    frame = frame[(frame.method == method) & (frame.k.astype(int) == int(k))].copy()
    frame["probability"] = frame.probabilities_json.map(lambda x: np.asarray(json.loads(x), float))
    aligned = strict_align(rows, frame)
    columns = list(rows.columns)
    output = aligned[[f"{c}_left" if f"{c}_left" in aligned else c for c in columns]].copy()
    output.columns = columns
    output["probability"] = aligned.probability
    output["method"] = method
    output["k"] = int(k)
    return output


def records(rows, probabilities, method, k):
    output = rows.copy().reset_index(drop=True)
    output["probability"] = [np.asarray(x, float)[:int(n)]
                             for x, n in zip(probabilities, output.n_options)]
    output["method"] = method; output["k"] = int(k)
    return output


def persona_stat_inductive(train, validation, test, responses, vectors, banks, roles, args):
    """Fit each train-only head once and predict both held-out respondent splits."""
    outputs = {"validation": [], "test": []}
    early_stop_panels, _ = respondent_split(validation.panel_id.astype(str), args.seed)
    calibration = {domain: data["calibration"] for domain, data in banks["domains"].items()}
    for domain, bank in banks["domains"].items():
        domain_train = train[(train.domain == domain) &
                             train.question_key.isin(roles["calibration"] + roles["train"])]
        domain_heldout = pd.concat([validation[validation.domain == domain],
                                    test[test.domain == domain]], ignore_index=True)
        if domain_heldout.empty:
            continue
        labels = {str(k): int(v) for k, v in bank["panel_cluster"].items()}
        train_panels = set(train.loc[train.domain == domain, "panel_id"].astype(str))
        heldout_panels = set(domain_heldout.panel_id.astype(str))
        if not set(labels) <= train_panels or set(labels) & heldout_panels:
            raise ValueError(f"{domain}: persona bank is not train-respondent-only")
        if set(bank["calibration"]) != set(calibration[domain]):
            raise ValueError(f"{domain}: persona bank calibration does not match reused split")
        router = fit_demographic_router(train[train.domain == domain], labels)
        priors = router_prior(router, domain_heldout, int(banks["m"]))
        posterior_by_split = {}
        for split_name, target in (("validation", validation), ("test", test)):
            domain_target = target[target.domain == domain]
            split_responses = responses[responses.split.eq(split_name) & responses.domain.eq(domain)]
            answer_lookup = {(str(r.panel_id), str(r.question_key)): int(r.answer_index)
                             for r in split_responses.itertuples(index=False)
                             if str(r.question_key) in set(calibration[domain])}
            posterior_by_split[split_name] = {}
            for row in domain_target.itertuples(index=False):
                history = []
                for question in stable_history(str(row.panel_id), calibration[domain], args.k, args.seed):
                    answer = answer_lookup.get((str(row.panel_id), question))
                    if answer is not None:
                        history.append((question, answer))
                posterior = update_posterior(priors[str(row.panel_id)], history, bank["response_prob"])
                posterior_by_split[split_name][str(row.panel_id)] = posterior
        domain_early_stop = validation[(validation.domain == domain) &
                                       validation.panel_id.astype(str).isin(early_stop_panels)]
        head = fit_persona_head(
            domain_train, vectors, labels, int(banks["m"]), args.epochs, args.lr,
            args.weight_decay, args.device, domain_early_stop,
            posterior_by_split["validation"], args.patience)
        for split_name, target in (("validation", validation), ("test", test)):
            domain_target = target[target.domain == domain]
            for row in domain_target.itertuples(index=False):
                posterior = posterior_by_split[split_name][str(row.panel_id)]
                probability = posterior @ persona_conditionals(head, vectors[str(row.question_key)])
                outputs[split_name].append((str(row.row_id), probability))
    frames = []
    for split_name, target in (("validation", validation), ("test", test)):
        probability = dict(outputs[split_name])
        if set(target.row_id.astype(str)) != set(probability):
            raise ValueError(f"persona statistical {split_name} predictions have incomplete coverage")
        frames.append(records(target, [probability[str(x)] for x in target.row_id],
                              "stat_history_inductive", args.k))
    return tuple(frames)


def calibrate_hybrids(val_stat, val_llm, test_stat, test_llm, args):
    joined = strict_align(val_stat, val_llm)
    panels = joined.panel_id_left.astype(str)
    group_a, group_b = respondent_split(panels, args.seed)
    if not group_a or not group_b:
        raise ValueError("validation respondent A/B split has an empty side")
    stat = list(joined.probability_left)
    llm_raw = list(joined.probability_right)
    labels = joined.answer_index_left.to_numpy(int)
    a = panels.isin(group_a).to_numpy(); b = panels.isin(group_b).to_numpy()
    temperature = fit_temperature([p for p, keep in zip(llm_raw, a) if keep], labels[a])
    llm_scaled = temperature_scale(llm_raw, temperature)
    lam = fit_lambda([p for p, keep in zip(stat, b) if keep],
                     [p for p, keep in zip(llm_scaled, b) if keep], labels[b])
    val = [records(val_llm, llm_scaled, "llm_persona_history_temperature", args.k),
           records(val_stat, log_opinion_pool(stat, llm_scaled, lam), "persona_llm_hybrid", args.k)]
    test_joined = strict_align(test_stat, test_llm)
    test_stat_p = list(test_joined.probability_left)
    test_llm_p = temperature_scale(list(test_joined.probability_right), temperature)
    test = [records(test_llm, test_llm_p, "llm_persona_history_temperature", args.k),
            records(test_stat, log_opinion_pool(test_stat_p, test_llm_p, lam), "persona_llm_hybrid", args.k)]
    config = {"temperature": temperature, "lambda": lam, "temperature_panels": sorted(group_a),
              "lambda_panels": sorted(group_b), "temperature_source": "validation_A_only",
              "lambda_source": "validation_B_only",
              "components": ["stat_history_inductive", "llm_persona_history"]}
    return val, test, config, (llm_scaled, test_llm_p)


def save_predictions(frame, path):
    output = frame.copy()
    output["probabilities_json"] = output.probability.map(lambda p: json.dumps(np.asarray(p).tolist()))
    output.drop(columns="probability").to_parquet(path, index=False)


def run(args):
    processed, final, output = Path(args.processed), Path(args.persona_results), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    responses = pd.read_parquet(processed / "responses.parquet")
    with open(final / "split.json", encoding="utf-8") as handle:
        split = json.load(handle)
    roles = fold_roles(split, args.fold)
    table = load_option_table(args.option_table)
    vectors, _, pca_keys = inductive_pca_option_vectors(table, roles, args.r, args.seed)
    train_people = responses[responses.split.eq("train")]
    validation = responses[responses.split.eq("validation") & responses.question_key.isin(roles["validation"])].copy()
    test = responses[responses.split.eq("test") & responses.question_key.isin(roles["test"])].copy()
    train = train_people[train_people.question_key.isin(roles["calibration"] + roles["train"])].copy()
    calibration = {domain: data["calibration"] for domain, data in split["domains"].items()}
    max_options = max(len(value) for value in vectors.values())
    train_arrays = build_choice_arrays(
        train, train_people, vectors, calibration, args.k, args.seed, max_options)
    val_arrays = build_choice_arrays(validation, responses[responses.split.eq("validation")], vectors,
                                     calibration, args.k, args.seed, max_options)
    test_arrays = build_choice_arrays(test, responses[responses.split.eq("test")], vectors,
                                      calibration, args.k, args.seed, max_options)
    model = SklearnChoiceBaseline("hist_gbdt", seed=args.seed).fit(train_arrays)
    val_frames = [records(validation, model.predict_proba(val_arrays), "hist_gbdt", args.k)]
    test_frames = [records(test, model.predict_proba(test_arrays), "hist_gbdt", args.k)]
    for demographic, name in ((False, "cross_item_position"), (True, "demographic_position")):
        prior = fit_position_prior(train, demographic=demographic)
        val_frames.append(records(validation, predict_position_prior(prior, validation), name, args.k))
        test_frames.append(records(test, predict_position_prior(prior, test), name, args.k))
    val_frames.append(records(validation, [np.ones(int(n)) / int(n) for n in validation.n_options], "uniform", args.k))
    test_frames.append(records(test, [np.ones(int(n)) / int(n) for n in test.n_options], "uniform", args.k))
    with open(final / "persona_bank.json", encoding="utf-8") as handle:
        banks = json.load(handle)
    val_stat, test_stat = persona_stat_inductive(
        train_people, validation, test, responses, vectors, banks, roles, args)
    val_llm = load_predictions(final / "llm_validation_predictions.parquet", "llm_persona_history", args.k, validation)
    test_llm = load_predictions(final / "predictions.parquet", "llm_persona_history_uncalibrated", args.k, test)
    val_frames += [val_stat, val_llm]; test_frames += [test_stat, test_llm]
    hybrid_val, hybrid_test, config, scaled = calibrate_hybrids(val_stat, val_llm, test_stat, test_llm, args)
    val_frames += hybrid_val; test_frames += hybrid_test
    val_hist = val_frames[0]
    val_join = strict_align(val_hist, hybrid_val[0])
    b_panels = set(config["lambda_panels"]); use = val_join.panel_id_left.astype(str).isin(b_panels)
    hist_lambda = fit_lambda(list(val_join.loc[use, "probability_left"]),
                             list(val_join.loc[use, "probability_right"]),
                             val_join.loc[use, "answer_index_left"].to_numpy(int))
    test_hist = list(test_frames[0].probability)
    test_frames.append(records(test, log_opinion_pool(test_hist, scaled[1], hist_lambda),
                               "hist_gbdt_llm_hybrid", args.k))
    config.update({"hist_gbdt_secondary_lambda": hist_lambda, "pca_fit_keys": pca_keys})
    validation_predictions = pd.concat(val_frames, ignore_index=True)
    predictions = pd.concat(test_frames, ignore_index=True)
    save_predictions(validation_predictions, output / "validation_predictions.parquet")
    save_predictions(predictions, output / "predictions.parquet")
    pd.DataFrame([{"method": method, "k": args.k, **metrics(frame)}
                  for method, frame in predictions.groupby("method")]).to_csv(output / "results.csv", index=False)
    comparisons = {
        "hybrid_vs_stat_history_inductive": ("stat_history_inductive", "persona_llm_hybrid"),
        "hist_gbdt_vs_stat_history_inductive": ("stat_history_inductive", "hist_gbdt"),
        "hist_gbdt_hybrid_vs_hist_gbdt": ("hist_gbdt", "hist_gbdt_llm_hybrid"),
        "hist_gbdt_hybrid_vs_llm": ("llm_persona_history_temperature", "hist_gbdt_llm_hybrid"),
        "hist_gbdt_hybrid_vs_persona_hybrid": ("persona_llm_hybrid", "hist_gbdt_llm_hybrid"),
    }
    gaps = {}
    for name, (left, right) in comparisons.items():
        gap, low, high = respondent_bootstrap_nll_gap(predictions, left, right, args.seed)
        gaps[name] = {"left": left, "right": right, "nll_difference_right_minus_left": gap,
                      "bootstrap_95_ci": [low, high], "right_beats_left": bool(high < 0)}
    gates = {"hybrid_vs_stat_history_inductive": gaps["hybrid_vs_stat_history_inductive"],
              "hist_gbdt_vs_stat_history_inductive": gaps["hist_gbdt_vs_stat_history_inductive"],
              "hist_gbdt_hybrid_vs_hist_gbdt": gaps["hist_gbdt_hybrid_vs_hist_gbdt"],
              "hist_gbdt_hybrid_vs_llm": gaps["hist_gbdt_hybrid_vs_llm"],
              "hist_gbdt_hybrid_vs_persona_hybrid": gaps["hist_gbdt_hybrid_vs_persona_hybrid"],
              "llm_component_selected": bool(config["lambda"] > 0), "bootstrap_seed": args.seed}
    (output / "hybrid_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True); parser.add_argument("--option-table", required=True)
    parser.add_argument("--persona-results", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0); parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--k", type=int, default=5); parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
