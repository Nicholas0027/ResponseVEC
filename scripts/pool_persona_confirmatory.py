#!/usr/bin/env python
"""Pool frozen persona confirmatory folds without further model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from responsevec.persona_evaluation import metrics, respondent_bootstrap_nll_gap


def load_predictions(path: Path, fold: int) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["probability"] = frame.probabilities_json.map(
        lambda value: np.asarray(json.loads(value), float))
    frame["outer_fold"] = int(fold)
    return frame


def comparison_record(frame, left, right, seed):
    gap, low, high = respondent_bootstrap_nll_gap(frame, left, right, seed, 5000)
    return {"left": left, "right": right, "nll_difference_right_minus_left": gap,
            "bootstrap_95_ci": [low, high], "right_beats_left": bool(high < 0.0),
            "left_beats_right": bool(low > 0.0)}


def item_bootstrap_record(frame, left, right, seed, draws=10000):
    subset = frame[frame.method.isin([left, right])]
    wide = subset.pivot(index=["domain", "question_key", "row_id", "answer_index"],
                        columns="method", values="probability").reset_index()
    wide["gap"] = [-np.log(max(b[int(y)], 1e-12)) + np.log(max(a[int(y)], 1e-12))
                   for y, a, b in zip(wide.answer_index, wide[left], wide[right])]
    items = wide.groupby(["domain", "question_key"]).gap.mean().reset_index()
    domains = [group.gap.to_numpy() for _, group in items.groupby("domain")]
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.mean(np.concatenate([
        values[rng.integers(0, len(values), len(values))] for values in domains
    ])) for _ in range(int(draws))])
    return {"unit": "question_key stratified by domain", "items": len(items),
            "nll_difference_right_minus_left": float(items.gap.mean()),
            "bootstrap_95_ci": list(map(float, np.quantile(boot, [0.025, 0.975]))),
            "items_right_better": int((items.gap < 0).sum()), "draws": int(draws)}


def run(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    comparison_frames, router_frames = [], []
    for fold in args.folds:
        comparison = Path(args.comparison_root) / args.comparison_pattern.format(fold=fold)
        router = Path(args.router_root) / args.router_pattern.format(fold=fold)
        comparison_frames.append(load_predictions(comparison / "predictions.parquet", fold))
        router_frames.append(load_predictions(router / "predictions.parquet", fold))
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    routers = pd.concat(router_frames, ignore_index=True)
    per_fold = []
    for (fold, method), frame in comparisons.groupby(["outer_fold", "method"]):
        per_fold.append({"outer_fold": fold, "method": method, **metrics(frame)})
    pooled = [{"method": method, **metrics(frame)} for method, frame in comparisons.groupby("method")]
    pd.DataFrame(per_fold).to_csv(output / "per_fold_results.csv", index=False)
    pd.DataFrame(pooled).to_csv(output / "pooled_results.csv", index=False)
    history = routers[routers.k.astype(int).eq(int(args.history_k))]
    gates = {
        "folds": list(map(int, args.folds)),
        "primary_hybrid_gate": comparison_record(
            comparisons, "hist_gbdt", "hist_gbdt_llm_hybrid", args.seed),
        "router_validity_gate": comparison_record(
            history, "stat_history", "stat_shuffled", args.seed),
        "llm_router_gate": comparison_record(
            routers[routers.k.astype(int).eq(int(args.k))],
            "llm_persona_prior", "llm_persona_history", args.seed),
        "development_fold_excluded": 0 not in set(args.folds),
        "bootstrap_unit": "panel_id",
        "bootstrap_draws": 5000,
        "bootstrap_seed": int(args.seed),
        "post_freeze_item_robustness": item_bootstrap_record(
            comparisons, "hist_gbdt", "hist_gbdt_llm_hybrid", args.seed),
    }
    gates["confirmatory_success"] = bool(
        gates["primary_hybrid_gate"]["right_beats_left"] and
        gates["router_validity_gate"]["left_beats_right"])
    (output / "confirmatory_gates.json").write_text(
        json.dumps(gates, indent=2), encoding="utf-8")
    print(pd.DataFrame(pooled).sort_values("nll").to_string(index=False))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--router-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--comparison-pattern", default="confirmatory_fold_0{fold}")
    parser.add_argument("--router-pattern", default="confirmatory_fold_0{fold}")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--history-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    run(parser.parse_args())
