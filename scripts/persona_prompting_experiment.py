#!/usr/bin/env python
"""Persona Effect Prompting (Hu & Collier, ACL 2024) ADAPTED baseline [P1].

Runs the five prompt-only persona-effect variants (no-persona, demographic,
demographic+behaviour, history, shuffled-behaviour control) through a
persona-conditioned reader on the frozen doubly-cold protocol (Table A).
Temperature is fit on VALIDATION respondents only and applied to TEST once.

The reader is injectable. With ``--run-llm`` the frozen Qwen3 CausalExtractor
(thinking disabled) supplies option-token probabilities; otherwise a
deterministic dummy reader lets the whole pipeline run and be verified on CPU.
This is an ADAPTED (not EXACT) reproduction of Hu & Collier: personas come from
the frozen per-domain persona bank, the reader is Qwen3 option-token
probabilities, and the setting is cold-item / cold-respondent. Structure,
dummy_reader, records(), save_predictions(), and the --llm-* CLI flags mirror
scripts/personadb_experiment.py for reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from responsevec.encode import load_option_table
from responsevec.persona_comparison import (fit_temperature,
    inductive_pca_option_vectors, temperature_scale)
from responsevec.persona_evaluation import metrics
from responsevec.persona_prompting import (VARIANTS, build_behavior_persona_index,
    build_shuffled_history, score_variant)
from responsevec.persona_router import fold_roles


def records(rows, probabilities, method):
    output = rows.copy().reset_index(drop=True)
    if "survey_weight" in output:
        output["survey_weight"] = np.asarray(output.survey_weight, dtype=np.float32)
    output["probability"] = [np.asarray(x, float)[: int(n)]
                             for x, n in zip(probabilities, output.n_options)]
    output["method"] = method
    return output


def dummy_reader(seed):
    """Deterministic, label-free option probabilities for CPU smoke testing.

    Depends only on the prompt text (which never contains the target label), so
    it exercises alignment and leakage guards without a GPU.
    """
    def option_prob_fn(prompts, n_options, label_to_semantic):
        out = np.zeros((len(prompts), max(int(n) for n in n_options)), np.float32)
        for i, (prompt, n) in enumerate(zip(prompts, n_options)):
            digest = hashlib.sha256(f"{seed}|{prompt}".encode()).digest()
            logits = np.frombuffer(digest, np.uint8)[: int(n)].astype(np.float32) / 64.0
            probs = np.exp(logits - logits.max())
            out[i, : int(n)] = probs / probs.sum()
        return out
    return option_prob_fn


def _answer_lookup(responses, calibration_keys):
    """(panel_id, question_key) -> answer_index for calibration answers only."""
    subset = responses[responses.question_key.isin(calibration_keys)]
    return {(str(r.panel_id), str(r.question_key)): int(r.answer_index)
            for r in subset.itertuples(index=False)}


def _text_maps(responses):
    catalogue = responses.drop_duplicates("question_key")
    question_text = {str(r.question_key): str(r.question)
                     for r in catalogue.itertuples(index=False)}
    options_by_question = {}
    for r in catalogue.itertuples(index=False):
        value = r.options_json
        options_by_question[str(r.question_key)] = (
            json.loads(value) if isinstance(value, str) else list(value))
    return question_text, options_by_question


def run(args):
    processed = Path(args.processed)
    results = Path(args.persona_results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    responses = pd.read_parquet(processed / "responses.parquet")
    with open(results / "split.json", encoding="utf-8") as handle:
        split = json.load(handle)
    with open(results / "persona_bank.json", encoding="utf-8") as handle:
        persona_bank = json.load(handle)
    roles = fold_roles(split, args.fold)
    table = load_option_table(args.option_table)
    # PCA option vectors: only used to route respondents to behaviour personas via
    # calibration profiles; fit on calibration/train items only (no test labels).
    vectors, _, _ = inductive_pca_option_vectors(table, roles, args.r, args.seed)
    calibration_by_domain = {domain: data["calibration"]
                             for domain, data in split["domains"].items()}
    calibration_keys = set(roles["calibration"])

    validation = responses[responses.split.eq("validation")
                           & responses.question_key.isin(roles["validation"])].copy()
    test = responses[responses.split.eq("test")
                     & responses.question_key.isin(roles["test"])].copy()

    # History / behaviour persona use calibration answers ONLY, within each split.
    train_cal = responses[responses.split.eq("train")
                          & responses.question_key.isin(calibration_keys)]
    val_cal = responses[responses.split.eq("validation")
                        & responses.question_key.isin(calibration_keys)]
    test_cal = responses[responses.split.eq("test")
                         & responses.question_key.isin(calibration_keys)]
    val_lookup = _answer_lookup(val_cal, calibration_keys)
    test_lookup = _answer_lookup(test_cal, calibration_keys)

    # Behaviour-persona text: assign each held-out respondent to a frozen persona
    # (train-fit KMeans) using their own calibration answers, then read the frozen
    # per-domain persona_text. Assignment uses the persona bank profile space.
    val_behavior = build_behavior_persona_index(val_cal, persona_bank)
    test_behavior = build_behavior_persona_index(test_cal, persona_bank)

    question_text, options_by_question = _text_maps(responses)

    # Frozen question-aligned shuffle for the falsification control, per split.
    # Donor answers/demographics come from each split's calibration responses.
    val_shuffle = build_shuffled_history(val_cal, args.k, args.seed, calibration_by_domain)
    test_shuffle = build_shuffled_history(test_cal, args.k, args.seed, calibration_by_domain)

    reader = make_reader(args)

    val_raw, test_raw = {}, {}
    for variant in VARIANTS:
        val_raw[variant] = score_variant(
            validation, variant, reader, k=args.k, seed=args.seed,
            calibration_by_domain=calibration_by_domain, answer_lookup=val_lookup,
            question_text=question_text, options_by_question=options_by_question,
            behavior_index=val_behavior, shuffled_history=val_shuffle)
        test_raw[variant] = score_variant(
            test, variant, reader, k=args.k, seed=args.seed,
            calibration_by_domain=calibration_by_domain, answer_lookup=test_lookup,
            question_text=question_text, options_by_question=options_by_question,
            behavior_index=test_behavior, shuffled_history=test_shuffle)

    val_frames, test_frames, temperatures = [], [], {}
    for variant in VARIANTS:
        val_frames.append(records(validation, val_raw[variant], variant + "_uncalibrated"))
        test_frames.append(records(test, test_raw[variant], variant + "_uncalibrated"))
        # Temperature fit on VALIDATION labels only; applied to TEST once.
        temperature = fit_temperature(val_raw[variant],
                                      validation.answer_index.to_numpy(int))
        temperatures[variant] = temperature
        val_frames.append(records(validation,
                                  temperature_scale(val_raw[variant], temperature), variant))
        test_frames.append(records(test,
                                   temperature_scale(test_raw[variant], temperature), variant))

    validation_predictions = pd.concat(val_frames, ignore_index=True)
    predictions = pd.concat(test_frames, ignore_index=True)
    save_predictions(validation_predictions, output / "validation_predictions.parquet")
    save_predictions(predictions, output / "predictions.parquet")
    pd.DataFrame([{"method": method, "fold": args.fold, **metrics(frame)}
                  for method, frame in predictions.groupby("method")]).to_csv(
        output / "results.csv", index=False)
    config = {"fold": args.fold, "seed": args.seed, "k": args.k, "r": args.r,
              "variants": list(VARIANTS), "temperatures": temperatures,
              "temperature_source": "validation_only",
              "reader": "qwen_causal" if args.run_llm else "dummy",
              "mode": "ADAPTED", "method_family": "persona_effect_prompting",
              "reference": "Hu & Collier, ACL 2024"}
    (output / "persona_prompting_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")


def make_reader(args):
    if args.run_llm:
        from responsevec.persona_prompting import make_causal_reader
        return make_causal_reader(args.llm_model, args.seed, args.llm_max_length,
                                  args.llm_batch_size, args.llm_quantization)
    return dummy_reader(args.seed)


def save_predictions(frame, path):
    output = frame.copy()
    output["probabilities_json"] = output.probability.map(
        lambda p: json.dumps(np.asarray(p).tolist()))
    output.drop(columns="probability").to_parquet(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--option-table", required=True)
    parser.add_argument("--persona-results", required=True,
                        help="directory holding the frozen split.json + persona_bank.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--llm-max-length", type=int, default=512)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--llm-quantization", default=None,
                        help="e.g. nf4 for large backbones on a single card")
    run(parser.parse_args())
