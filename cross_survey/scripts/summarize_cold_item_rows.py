#!/usr/bin/env python3
"""Recompute cold-item method gaps with respondent and FAMILY bootstraps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(frame, left, right, unit, seed=1701, draws=5000):
    keys = ["panel_id", "item", "family"]
    a = frame[frame.method.eq(left)][keys + ["nll", "correct"]]
    b = frame[frame.method.eq(right)][keys + ["nll", "correct"]]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    merged["nll_gap"] = merged.nll_a - merged.nll_b
    merged["acc_gap"] = merged.correct_a - merged.correct_b
    column = "panel_id" if unit == "respondent" else "family"
    grouped = list(merged.groupby(column))
    clusters = np.asarray([
        group[["nll_gap", "acc_gap"]].mean().to_numpy()
        for _, group in grouped
    ])
    sizes = np.asarray([len(group) for _, group in grouped], float)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, 2))
    for draw in range(draws):
        pick = rng.integers(0, len(clusters), len(clusters))
        boot[draw] = np.average(clusters[pick], axis=0, weights=sizes[pick])
    return {
        "nll": {"difference": float(merged.nll_gap.mean()),
                "ci": np.quantile(boot[:, 0], [0.025, 0.975]).tolist()},
        "accuracy": {"difference": float(merged.acc_gap.mean()),
                     "ci": np.quantile(boot[:, 1], [0.025, 0.975]).tolist()},
        "clusters": len(clusters), "unit": unit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/results/phase1/cold_item_loading.rows.parquet")
    parser.add_argument("--output", default="cross_survey/results/phase1/cold_item_loading_inference.json")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    pairs = [
        ("tfidf", "uniform"), ("bge", "uniform"), ("tfidf", "bge"),
        ("tfidf_oracle_intercept", "population_oracle"),
        ("bge_oracle_intercept", "population_oracle"),
        ("warm_oracle", "population_oracle"),
    ]
    result = {"source": args.input, "comparisons": {}}
    for left, right in pairs:
        result["comparisons"][f"{left}_vs_{right}"] = {
            unit: compare(frame, left, right, unit)
            for unit in ("respondent", "family")
        }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
