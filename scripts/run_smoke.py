#!/usr/bin/env python
"""Execute the real cache/head/metric chain with deterministic fake encoders.

No number from this script is scientific. Its purpose is to prove that a fresh
CPU environment can traverse the same file contracts the A100 run will use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from responsevec.eval.eval_rv import primary_family
from responsevec.eval.metrics import add_probability_scores, compute_metric_tables
from responsevec.utils import write_json


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--workdir", default="artifacts/smoke")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project / args.config).read_text())
    workdir = (project / args.workdir).resolve()
    for key in ("processed", "cache", "heads", "predictions", "metrics", "figures"):
        base["paths"][key] = str(workdir / key)
    base["paths"]["sociobench_repo"] = str(workdir / "unused_sociobench")
    base["decoder"]["projection_dim"] = 24
    base["decoder"]["batch_size"] = 128
    base["decoder"]["early_stopping_patience"] = 3
    smoke_config = workdir / "smoke_config.yaml"
    workdir.mkdir(parents=True, exist_ok=True)
    smoke_config.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    py = sys.executable

    run([py, "scripts/prepare_data.py", "--config", str(smoke_config), "--synthetic"], project)
    run([py, "scripts/encode_options.py", "--config", str(smoke_config), "--synthetic-encoder", "--overwrite"], project)
    for role, split in (("train", "train"), ("validation", "validation"), ("test", "test")):
        for family in ("causal", "input_centric", "response_centric", "sentence"):
            run([
                py, "scripts/extract_representations.py", "--config", str(smoke_config),
                "--family", family, "--fold", "0", "--role", role, "--split", split,
                "--k", "5", "--option-seed", "0", "--selection", "semantic",
                "--synthetic-encoder", "--overwrite", "--shard-size", "256",
            ], project)
        run([
            py, "scripts/extract_respondentvec.py", "--config", str(smoke_config),
            "--split", split, "--k", "5", "--synthetic-encoder", "--overwrite", "--shard-size", "256",
        ], project)
    run([
        py, "scripts/train_primary.py", "--config", str(smoke_config),
        "--fold", "0", "--k", "5", "--option-seeds", "0",
        "--epochs", "5", "--seeds", "1701", "--device", "cpu",
        "--include-respondent-vec",
    ], project)

    prediction_path = workdir / "predictions" / "fold_00" / "k_5" / "predictions_seed_averaged.parquet"
    predictions = add_probability_scores(pd.read_parquet(prediction_path))
    required = {"response_centric", "direct_selected", "raw_selected", "input_centric", "sentence", "respondent_vec"}
    missing = required - set(predictions["method"])
    if missing:
        raise AssertionError(f"smoke predictions missing methods: {missing}")
    metric_dir = workdir / "metrics" / "fold_00"
    compute_metric_tables(predictions, metric_dir)
    # Primary method is response_centric (LLM2Vec-Gen): the three preregistered
    # H1/H2/H3 comparisons (vs direct generation, vs input-centric LLM2Vec, vs
    # the free raw hidden-state control). A secondary, non-gating check
    # (raw vs a generic off-the-shelf sentence embedder) mirrors LLMGeovec's
    # "not just any embedder" control but does not gate the primary claim.
    primary = primary_family(
        predictions, "response_centric",
        ["direct_selected", "input_centric", "raw_selected"],
        metric="nll", k=5, item_pool="unseen", replicates=200,
    )
    primary.to_csv(metric_dir / "primary_smoke.csv", index=False)
    secondary = primary_family(
        predictions, "raw_selected", ["sentence"],
        metric="nll", k=5, item_pool="unseen", replicates=200,
    )
    secondary.to_csv(metric_dir / "secondary_smoke.csv", index=False)
    summary = {
        "status": "PASS", "scientific_results": False,
        "prediction_rows": len(predictions),
        "methods": sorted(set(predictions["method"])),
        "primary_code_path_exercised": len(primary) == 3,
        "config": str(smoke_config),
    }
    write_json(workdir / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
