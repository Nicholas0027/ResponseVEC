#!/usr/bin/env python3
"""Direction A: how many pilot respondents does a new survey item need?

Pre-registered in ``01_narrative/CROSS_SURVEY_BLUEPRINT.md`` section 2b before any
few-shot number was computed.

Three attempts to predict item loadings from text have failed (option embeddings,
BGE, structured LLM schema), while the respondent side is established.  This script
concedes the fully-cold premise and measures the *pilot cost* instead: for each
held-out target family and each pilot size m, fit that item's option coefficients
on m TRAIN respondents only, then score the untouched test respondents.

Pilot respondents come from the train split and are never scored, so no test
respondent is ever observed during fitting.  m=0 reduces to the cold prior and
m=all reduces to the warm oracle, which together give the honesty control G-A3.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("cold", HERE / "run_cold_item_loading.py")
cold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cold)

spec2 = importlib.util.spec_from_file_location("kceiling", HERE / "run_k_ceiling.py")
kceiling = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(kceiling)

PILOT_SIZES = [0, 5, 10, 20, 50, 100, 200, 500]


def pooled_prior(coefficient_frame: pd.DataFrame, exclude_items: set[str]) -> np.ndarray:
    """Mean coefficient vector over training-family option rows, by option position.

    Used as the shrinkage target.  Indexed by (n_options, option_position) so that
    a 5-point scale is shrunk toward other 5-point scales rather than toward the
    global mean of every scale type.
    """
    usable = coefficient_frame[~coefficient_frame.item.isin(exclude_items)]
    table = {}
    for (n_opt, pos), group in usable.groupby(["n_options", "option_position"]):
        table[(int(n_opt), int(pos))] = group[
            ["coef_intercept", "coef_z1", "coef_z2"]
        ].mean().to_numpy()
    global_mean = usable[["coef_intercept", "coef_z1", "coef_z2"]].mean().to_numpy()
    return table, global_mean


def prior_for_item(option_frame: pd.DataFrame, table: dict,
                   global_mean: np.ndarray) -> np.ndarray:
    rows = []
    for row in option_frame.itertuples(index=False):
        key = (int(row.n_options), int(row.option_position))
        rows.append(table.get(key, global_mean))
    return np.asarray(rows, float)


def fit_pilot_coefficients(z_pilot: np.ndarray, answers: np.ndarray,
                           codes: list[int], prior: np.ndarray,
                           kappa: float, shrink: bool) -> np.ndarray:
    """Per-option logistic fit on the pilot rows, shrunk toward the pooled prior."""
    m = len(answers)
    if m == 0:
        return prior.copy()
    from sklearn.linear_model import LogisticRegression

    raw = []
    for code in codes:
        binary = (answers == code).astype(int)
        if binary.min() == binary.max() or m < 3:
            prevalence = float(np.clip(binary.mean(), 1e-3, 1 - 1e-3))
            raw.append([np.log(prevalence / (1 - prevalence)), 0.0, 0.0])
            continue
        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        model.fit(z_pilot, binary)
        raw.append([float(model.intercept_[0]),
                    float(model.coef_[0, 0]), float(model.coef_[0, 1])])
    raw = np.asarray(raw, float)
    if not shrink:
        return raw
    weight = m / (m + kappa)
    return weight * raw + (1.0 - weight) * prior


def select_kappa(coefficient_frame: pd.DataFrame, options: pd.DataFrame,
                 train: pd.DataFrame, z_train: np.ndarray,
                 train_families: list[str],
                 candidates=(0.5, 1.0, 2.0, 5.0, 20.0, 50.0, 200.0),
                 probe_m: int = 20, seed: int = 1701) -> tuple[float, dict]:
    """Leave-one-training-family-out selection of the shrinkage constant.

    Selected once at a single probe pilot size and then frozen across all m, so
    that kappa cannot be tuned per operating point.
    """
    rng = np.random.default_rng(seed)
    scores = {}
    for kappa in candidates:
        losses = []
        for held in train_families:
            held_items = coefficient_frame[
                coefficient_frame.family.eq(held)
            ].item.unique().tolist()
            if not held_items:
                continue
            table, global_mean = pooled_prior(coefficient_frame, set(held_items))
            for item in held_items:
                option_frame = options[options.item.eq(item)].sort_values("option_position")
                codes = option_frame.option_code.astype(int).tolist()
                prior = prior_for_item(option_frame, table, global_mean)
                observed = train[item].notna().to_numpy()
                idx = np.flatnonzero(observed)
                if len(idx) < probe_m + 30:
                    continue
                rng.shuffle(idx)
                pilot_idx, eval_idx = idx[:probe_m], idx[probe_m:probe_m + 400]
                coefficients = fit_pilot_coefficients(
                    z_train[pilot_idx],
                    train[item].to_numpy()[pilot_idx].astype(int),
                    codes, prior, kappa, shrink=True)
                probabilities, out_codes = cold.predictions_from_coefficients(
                    z_train[eval_idx], option_frame, coefficients)
                lookup = {c: i for i, c in enumerate(out_codes)}
                truth = train[item].to_numpy()[eval_idx].astype(int)
                for answer, probability in zip(truth, probabilities):
                    if int(answer) in lookup:
                        losses.append(
                            -np.log(max(probability[lookup[int(answer)]], cold.EPS)))
        scores[str(kappa)] = float(np.mean(losses)) if losses else float("inf")
    best = min(candidates, key=lambda k: scores[str(k)])
    return float(best), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv")
    parser.add_argument("--source-catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--source-manifest", default="cross_survey/metadata/ces2020_cross_construct_manifest.json")
    parser.add_argument("--options", default="cross_survey/metadata/ces2020_all_item_options.csv")
    parser.add_argument("--folds", default="cross_survey/metadata/ces2020_cold_item_folds.json")
    parser.add_argument("--output", default="cross_survey/results/phase1/fewshot_item_calibration.json")
    parser.add_argument("--pilot-sizes", type=int, nargs="*", default=PILOT_SIZES)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    source_manifest = json.loads(Path(args.source_manifest).read_text())
    fold_manifest = json.loads(Path(args.folds).read_text())
    source_catalog = pd.read_csv(args.source_catalog)
    sources = source_catalog[
        source_catalog.wave.eq("source_pre") & source_catalog.found
        & source_catalog.item_specific_text.fillna(False)
        & source_catalog.family.isin(source_manifest["source_allowed_families"])
    ].item.tolist()
    options = pd.read_csv(args.options)
    items = fold_manifest["retained_items"]
    loading_source_items = options[
        options.wave.eq("source_pre") & ~options.item.isin(sources)
    ].item.drop_duplicates().tolist()
    option_bank_items = loading_source_items + items
    options = options[options.item.isin(option_bank_items)].copy()
    columns = ["caseid", "starttime_post"] + sorted(
        set(sources + loading_source_items + items))
    frame = pd.read_csv(args.input, usecols=columns, low_memory=False)
    frame = frame[frame.starttime_post.notna()].reset_index(drop=True)
    buckets = frame.caseid.map(lambda x: kceiling.bucket(x, args.seed))
    train = frame[buckets <= 5].reset_index(drop=True)
    test = frame[buckets >= 8].reset_index(drop=True)

    encoder = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), sources)])
    x_train = encoder.fit_transform(train[sources])
    x_test = encoder.transform(test[sources])
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=args.seed)
    z_train = svd.fit_transform(x_train)
    z_test = svd.transform(x_test)
    mean, std = z_train.mean(axis=0), z_train.std(axis=0)
    z_train = (z_train - mean) / np.maximum(std, 1e-8)
    z_test = (z_test - mean) / np.maximum(std, 1e-8)

    coefficient_rows = []
    warm_coefficients = {}
    for item in option_bank_items:
        option_frame = options[options.item.eq(item)].sort_values("option_position")
        codes = option_frame.option_code.astype(int).tolist()
        coefficients = cold.fit_option_coefficients(z_train, train[item], codes)
        warm_coefficients[item] = coefficients
        for row, coefficient in zip(option_frame.itertuples(index=False), coefficients):
            coefficient_rows.append({
                "item": item, "family": row.family,
                "option_code": int(row.option_code),
                "option_position": int(row.option_position),
                "n_options": int(row.n_options),
                "coef_intercept": coefficient[0],
                "coef_z1": coefficient[1], "coef_z2": coefficient[2],
            })
    coefficient_frame = pd.DataFrame(coefficient_rows)

    target_families = sorted({
        family for family, _ in fold_manifest["family_to_fold"].items()})
    train_families = sorted(
        set(coefficient_frame.family.unique()) - set(target_families))
    kappa, kappa_scores = select_kappa(
        coefficient_frame, options, train, z_train, train_families, seed=args.seed)
    print(f"selected kappa={kappa} from {kappa_scores}")

    score_records = []
    coverage = []
    rng = np.random.default_rng(args.seed)
    for item in items:
        option_frame = options[options.item.eq(item)].sort_values("option_position")
        family = str(option_frame.family.iloc[0])
        codes = option_frame.option_code.astype(int).tolist()
        # Shrinkage prior excludes every target item, so no held-out family
        # information enters the prior.
        table, global_mean = pooled_prior(coefficient_frame, set(items))
        prior = prior_for_item(option_frame, table, global_mean)

        test_rows = test[item].notna().to_numpy()
        respondents = test.loc[test_rows, "caseid"].to_numpy()
        answers = test.loc[test_rows, item].astype(int).to_numpy()
        z_eval = z_test[test_rows]

        pilot_pool = np.flatnonzero(train[item].notna().to_numpy())
        train_answers = train[item].to_numpy()
        coverage.append({"item": item, "family": family,
                         "pilot_pool": int(len(pilot_pool)),
                         "test_rows": int(test_rows.sum())})

        uniform = np.full((len(z_eval), len(option_frame)), 1 / len(option_frame))
        cold.append_scores(score_records, "uniform", item, family, respondents,
                           answers, uniform, np.asarray(codes))
        p, out_codes = cold.predictions_from_coefficients(
            z_eval, option_frame, warm_coefficients[item])
        cold.append_scores(score_records, "warm_oracle", item, family, respondents,
                           answers, p, out_codes)

        for m in args.pilot_sizes:
            if m > len(pilot_pool):
                continue
            n_draws = 1 if m == 0 else args.draws
            for draw in range(n_draws):
                pick = rng.permutation(pilot_pool)[:m] if m else np.empty(0, int)
                for shrink, label in ((True, "fewshot"), (False, "fewshot_raw")):
                    coefficients = fit_pilot_coefficients(
                        z_train[pick], train_answers[pick].astype(int) if m else
                        np.empty(0, int), codes, prior, kappa, shrink)
                    p, out_codes = cold.predictions_from_coefficients(
                        z_eval, option_frame, coefficients)
                    cold.append_scores(score_records, f"{label}_m{m}_d{draw}", item,
                                       family, respondents, answers, p, out_codes)

    scores = pd.DataFrame(score_records)
    # Average the repeated draws into one row set per (method_family, m).
    scores["arm"] = scores.method.str.replace(r"_d\d+$", "", regex=True)
    pooled = scores.groupby(["arm", "item", "family", "panel_id"], as_index=False).agg(
        nll=("nll", "mean"), correct=("correct", "mean"))
    pooled = pooled.rename(columns={"arm": "method"})

    summary = pooled.groupby("method").agg(
        nll=("nll", "mean"), accuracy=("correct", "mean"),
        rows=("nll", "size"), respondents=("panel_id", "nunique"),
        items=("item", "nunique"), families=("family", "nunique"),
    ).reset_index()

    uniform_nll = float(summary.loc[summary.method.eq("uniform"), "nll"].iloc[0])
    warm_nll = float(summary.loc[summary.method.eq("warm_oracle"), "nll"].iloc[0])
    warm_acc = float(summary.loc[summary.method.eq("warm_oracle"), "accuracy"].iloc[0])
    uniform_acc = float(summary.loc[summary.method.eq("uniform"), "accuracy"].iloc[0])
    headroom_nll = uniform_nll - warm_nll
    headroom_acc = warm_acc - uniform_acc
    summary["nll_headroom_recovered_pct"] = (
        (uniform_nll - summary.nll) / headroom_nll * 100.0)
    summary["accuracy_headroom_recovered_pct"] = (
        (summary.accuracy - uniform_acc) / headroom_acc * 100.0)

    comparisons = {}
    for m in args.pilot_sizes:
        arm = f"fewshot_m{m}"
        if arm not in pooled.method.values:
            continue
        comparisons[f"{arm}_vs_uniform"] = {
            "respondent": cold.clustered_comparison(pooled, arm, "uniform",
                                                    "respondent", args.seed),
            "family": cold.clustered_comparison(pooled, arm, "uniform",
                                                "item", args.seed),
        }
        comparisons[f"warm_oracle_vs_{arm}"] = {
            "respondent": cold.clustered_comparison(pooled, "warm_oracle", arm,
                                                    "respondent", args.seed),
            "family": cold.clustered_comparison(pooled, "warm_oracle", arm,
                                                "item", args.seed),
        }

    payload = {
        "dataset": "ces2020", "phase": "fewshot_item_calibration",
        "post_freeze": True, "exploratory": False,
        "preregistered": "01_narrative/CROSS_SURVEY_BLUEPRINT.md section 2b",
        "concession": "This is NOT cold-item transfer; it measures pilot cost for a "
                      "new item and concedes the fully-cold premise.",
        "respondent_dimensions": 2,
        "shrinkage_kappa": kappa, "kappa_loo_scores": kappa_scores,
        "pilot_sizes": args.pilot_sizes, "draws_per_size": args.draws,
        "tfidf_reference_nll": 1.2905,
        "uniform_nll": uniform_nll, "warm_oracle_nll": warm_nll,
        "nll_headroom": headroom_nll, "accuracy_headroom": headroom_acc,
        "split_counts": {"train": len(train), "test": len(test)},
        "item_coverage": coverage,
        "summary": summary.to_dict(orient="records"),
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    show = summary[summary.method.str.startswith(("fewshot_m", "uniform", "warm"))]
    print(show.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
