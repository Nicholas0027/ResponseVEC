#!/usr/bin/env python
"""Persona-DB (Sun et al., COLING 2025) ADAPTED baseline on the frozen protocol.

Runs the five Persona-DB variants (History-Full, History-Retrieval, IntSum,
w/o JOIN, full) through a persona-conditioned reader. Temperature is fit on
VALIDATION respondents only and applied to TEST once.

The reader is injectable. With ``--run-llm`` the frozen Qwen3-8B CausalExtractor
(thinking disabled) supplies option-token probabilities; otherwise a
deterministic dummy reader lets the whole pipeline run and be verified on CPU.
This is an ADAPTED (not EXACT) reproduction: persona keys are rule-based, the
reader is Qwen3-8B, and the setting is cold-item / cold-respondent.
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
    inductive_pca_option_vectors, strict_align, temperature_scale)
from responsevec.persona_evaluation import metrics
from responsevec.persona_router import fold_roles
from responsevec.personadb import (VARIANTS, assemble_evidence,
    build_collaborative_index, build_prompt, build_self_databases,
    make_causal_reader)
from responsevec.prompting_rv import OPTION_LABELS


def records(rows, probabilities, method):
    output = rows.copy().reset_index(drop=True)
    # Re-materialize numeric columns as fresh (writable) arrays: columns read from
    # a parquet-backed frame can be read-only, which trips the reused metrics().
    if "survey_weight" in output:
        # Store as float32 so metrics' to_numpy(float) (float64) forces a fresh,
        # writable copy rather than a read-only view of the parquet buffer.
        output["survey_weight"] = np.asarray(output.survey_weight, dtype=np.float32)
    output["probability"] = [np.asarray(x, float)[: int(n)]
                             for x, n in zip(probabilities, output.n_options)]
    output["method"] = method
    return output


def dummy_reader(seed):
    """Deterministic, label-free option probabilities for CPU smoke testing.

    Depends only on the prompt text (which never contains the target label),
    so it exercises alignment and leakage guards without a GPU.
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


def _label_to_semantic(n):
    return list(range(int(n)))


def score_variant(target, self_dbs, index, option_vectors, variant, reader, args):
    """Build prompts for one variant and return semantic-order probabilities."""
    import json as _json
    prompts, n_options, permutations = [], [], []
    join_count = 0
    for row in target.itertuples(index=False):
        options = _json.loads(row.options_json) if isinstance(row.options_json, str) \
            else list(row.options_json)
        target_vectors = np.asarray(option_vectors[str(row.question_key)], np.float32)
        self_db = self_dbs.get(str(row.panel_id))
        evidence, keys, join_fired = assemble_evidence(
            self_db, index, target_vectors, variant, args.top_neighbors,
            args.top_evidence, args.join_threshold)
        join_count += int(join_fired)
        permutation = _label_to_semantic(row.n_options)
        prompts.append(build_prompt(row, options, evidence, keys, permutation))
        n_options.append(int(row.n_options))
        permutations.append(permutation)
    probabilities = reader(prompts, n_options, permutations)
    return [probabilities[i] for i in range(len(target))], join_count


def run(args):
    processed = Path(args.processed)
    results = Path(args.persona_results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    responses = pd.read_parquet(processed / "responses.parquet")
    with open(results / "split.json", encoding="utf-8") as handle:
        split = json.load(handle)
    roles = fold_roles(split, args.fold)
    table = load_option_table(args.option_table)
    vectors, _, _ = inductive_pca_option_vectors(table, roles, args.r, args.seed)
    calibration = {domain: data["calibration"] for domain, data in split["domains"].items()}

    train_people = responses[responses.split.eq("train")]
    validation = responses[responses.split.eq("validation")
                           & responses.question_key.isin(roles["validation"])].copy()
    test = responses[responses.split.eq("test")
                     & responses.question_key.isin(roles["test"])].copy()

    # Self databases: calibration answers only, within each respondent split.
    calibration_keys = set(roles["calibration"])
    train_cal = train_people[train_people.question_key.isin(calibration_keys)]
    val_cal = responses[responses.split.eq("validation")
                        & responses.question_key.isin(calibration_keys)]
    test_cal = responses[responses.split.eq("test")
                         & responses.question_key.isin(calibration_keys)]
    train_self = build_self_databases(train_cal, vectors, calibration)
    val_self = build_self_databases(val_cal, vectors, calibration)
    test_self = build_self_databases(test_cal, vectors, calibration)

    # Collaborative index: TRAIN respondents only.
    index = build_collaborative_index(train_self)

    reader = make_causal_reader(args.llm_model, args.seed, args.llm_max_length,
                                args.llm_batch_size) if args.run_llm \
        else dummy_reader(args.seed)

    val_raw, test_raw, join_report = {}, {}, {}
    for variant in VARIANTS:
        val_probs, val_join = score_variant(validation, val_self, index, vectors,
                                             variant, reader, args)
        test_probs, test_join = score_variant(test, test_self, index, vectors,
                                               variant, reader, args)
        val_raw[variant] = val_probs
        test_raw[variant] = test_probs
        join_report[variant] = {"validation_join_rows": val_join,
                                "test_join_rows": test_join}

    val_frames, test_frames, temperatures = [], [], {}
    for variant in VARIANTS:
        val_frames.append(records(validation, val_raw[variant], variant + "_uncalibrated"))
        test_frames.append(records(test, test_raw[variant], variant + "_uncalibrated"))
        # Temperature fit on VALIDATION only; applied to TEST once.
        temperature = fit_temperature(val_raw[variant], validation.answer_index.to_numpy(int))
        temperatures[variant] = temperature
        val_frames.append(records(validation, temperature_scale(val_raw[variant], temperature),
                                  variant))
        test_frames.append(records(test, temperature_scale(test_raw[variant], temperature),
                                   variant))

    validation_predictions = pd.concat(val_frames, ignore_index=True)
    predictions = pd.concat(test_frames, ignore_index=True)
    save_predictions(validation_predictions, output / "validation_predictions.parquet")
    save_predictions(predictions, output / "predictions.parquet")
    pd.DataFrame([{"method": method, "fold": args.fold, **metrics(frame)}
                  for method, frame in predictions.groupby("method")]).to_csv(
        output / "results.csv", index=False)
    config = {"fold": args.fold, "seed": args.seed, "k": args.k,
              "top_neighbors": args.top_neighbors, "top_evidence": args.top_evidence,
              "join_threshold": args.join_threshold, "temperatures": temperatures,
              "temperature_source": "validation_only", "join_report": join_report,
              "reader": "qwen_causal" if args.run_llm else "dummy",
              "mode": "ADAPTED", "n_train_neighbors_available": len(index["panels"])}
    (output / "personadb_config.json").write_text(json.dumps(config, indent=2),
                                                  encoding="utf-8")


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
                        help="directory holding the frozen split.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--top-neighbors", type=int, default=8)
    parser.add_argument("--top-evidence", type=int, default=5)
    parser.add_argument("--join-threshold", type=int, default=3)
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--llm-max-length", type=int, default=512)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    run(parser.parse_args())
