#!/usr/bin/env python3
"""Respondent-bootstrap selector comparisons from a coverage run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_difference(a, b, seed=1701, draws=5000):
    """Mean(a-b); negative NLL favors a, positive accuracy favors a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired respondent arrays differ in length")
    difference = a - b
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for draw in range(draws):
        pick = rng.integers(0, len(difference), len(difference))
        boot[draw] = difference[pick].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"difference": float(difference.mean()),
            "ci": [float(lo), float(hi)], "draws": draws}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    records = [r for r in payload["results"] if int(r["k"]) == args.k]
    random_records = [r for r in records if r["selector"] == "random"]
    if not random_records:
        raise RuntimeError("no random records")
    random_nll = np.mean([r["respondent_nll"] for r in random_records], axis=0)
    random_accuracy = np.mean(
        [r["respondent_accuracy"] for r in random_records], axis=0
    )
    comparisons = {}
    for record in records:
        if record["selector"] in ("random", "family_balanced"):
            continue
        name = record["selector"]
        comparisons[f"{name}_vs_random_mean"] = {
            "nll": bootstrap_difference(record["respondent_nll"], random_nll,
                                         args.seed),
            "accuracy": bootstrap_difference(record["respondent_accuracy"],
                                              random_accuracy, args.seed),
        }
    # Direct deployable semantic comparison: negative NLL / positive accuracy
    # favors TF-IDF over BGE.
    by_name = {r["selector"]: r for r in records}
    if "tfidf_target_aware" in by_name and "bge_target_aware" in by_name:
        tfidf, bge = by_name["tfidf_target_aware"], by_name["bge_target_aware"]
        comparisons["tfidf_vs_bge"] = {
            "nll": bootstrap_difference(tfidf["respondent_nll"],
                                         bge["respondent_nll"], args.seed),
            "accuracy": bootstrap_difference(tfidf["respondent_accuracy"],
                                              bge["respondent_accuracy"], args.seed),
        }
    if "supervised_mi_oracle" in by_name and "tfidf_target_aware" in by_name:
        oracle = by_name["supervised_mi_oracle"]
        tfidf = by_name["tfidf_target_aware"]
        comparisons["mi_oracle_vs_tfidf"] = {
            "nll": bootstrap_difference(oracle["respondent_nll"],
                                         tfidf["respondent_nll"], args.seed),
            "accuracy": bootstrap_difference(oracle["respondent_accuracy"],
                                              tfidf["respondent_accuracy"], args.seed),
        }
    result = {"source": args.input, "k": args.k,
              "bootstrap_unit": "respondent", "comparisons": comparisons}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
