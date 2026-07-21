#!/usr/bin/env python
"""E0/G0: quantify usable history signal and unseen-item headroom before GPU work."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from responsevec.baselines.item_conditional import ItemConditionalModel
from responsevec.config import load_and_prepare
from responsevec.data import PanelStore
from responsevec.item_graph import ItemGraph, compute_item_graph
from responsevec.pipeline import fit_fold_prior, protocol_b_item_keys
from responsevec.prior import PopulationPrior
from responsevec.protocols import ItemFolds, build_protocol_a, build_protocol_b
from responsevec.utils import write_json


def macro_nll(frame: pd.DataFrame) -> float:
    panel = frame.groupby(["domain", "panel_id"], as_index=False)["nll"].mean()
    return float(panel.groupby("domain")["nll"].mean().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    store = PanelStore.from_dir(config["paths"]["processed"])
    folds = ItemFolds.load(Path(config["paths"]["processed"]) / "item_folds.json")
    train_rows = store.responses[store.responses["split"].eq("train")]

    # Seen-item history budget: fixed target units across K.
    prior_seen = PopulationPrior(**{
        "country_shrinkage": config["prior"]["country_shrinkage"],
        "laplace": config["prior"]["laplace"],
    }).fit(train_rows)
    graph_frame = train_rows[~train_rows["is_unseen_item"]]
    graph = ItemGraph(compute_item_graph(graph_frame, shrinkage_lambda=20.0, min_n_jk=10))
    conditional = ItemConditionalModel(prior_seen, graph, min_pair_count=10).fit(train_rows)
    units = build_protocol_a(store, "test")
    rows = []
    target_lookup = store.responses.set_index(["panel_id", "question_key"])
    for k in (0, 5):
        for unit in units:
            target = target_lookup.loc[(unit.panel_id, unit.question_key)]
            ordered_ids = store.order_lookup[(unit.panel_id, int(config["data"]["calibration_seed"]))]
            panel = store.by_panel[unit.panel_id].set_index("question_id")
            history = []
            for question_id in ordered_ids:
                if question_id not in panel.index:
                    continue
                source = panel.loc[question_id]
                if str(source.question_key) in unit.eligible_source_keys:
                    history.append((str(source.question_key), int(source.answer_index)))
                if len(history) >= k:
                    break
            probability = conditional.predict_one(
                unit.question_key, unit.domain, str(target.country), int(target.n_options), history
            )
            rows.append({
                "k": k, "domain": unit.domain, "panel_id": unit.panel_id,
                "nll": float(-np.log(max(probability[int(target.answer_index)], 1e-12))),
            })
    seen = pd.DataFrame(rows)
    seen_k0, seen_k5 = macro_nll(seen[seen["k"].eq(0)]), macro_nll(seen[seen["k"].eq(5)])
    seen_budget = seen_k0 - seen_k5

    # Unseen-item headroom: scale-position fallback vs an explicitly forbidden
    # train-answer oracle. The oracle is used only to decide whether the axis
    # contains measurable signal and is never a model baseline.
    prior_fold, train_keys = fit_fold_prior(
        store, folds, args.fold,
        PopulationPrior(config["prior"]["country_shrinkage"], config["prior"]["laplace"]),
    )
    test_units = build_protocol_b(store, folds, "test", outer_fold=args.fold, target_role="test")
    oracle_tables = {}
    for key, group in train_rows[train_rows["question_key"].isin(protocol_b_item_keys(folds, args.fold)["test"])].groupby("question_key"):
        n_options = int(group["n_options"].max())
        counts = np.bincount(group["answer_index"].astype(int), minlength=n_options) + 0.5
        oracle_tables[str(key)] = counts / counts.sum()
    scale_losses, oracle_losses = [], []
    for unit in test_units:
        target = target_lookup.loc[(unit.panel_id, unit.question_key)]
        scale = prior_fold.predict(unit.question_key, str(target.country), int(target.n_options), item_is_seen=False)
        oracle = oracle_tables.get(unit.question_key, np.full(int(target.n_options), 1 / int(target.n_options)))
        answer = int(target.answer_index)
        scale_losses.append(-np.log(max(scale[answer], 1e-12)))
        oracle_losses.append(-np.log(max(oracle[answer], 1e-12)))
    unseen_headroom = float(np.mean(scale_losses) - np.mean(oracle_losses))

    thresholds = config["evaluation"]["signal_audit"]
    result = {
        "seen_k0_nll": seen_k0, "seen_k5_nll": seen_k5,
        "seen_history_budget_nats": seen_budget,
        "unseen_oracle_headroom_nats": unseen_headroom,
        "seen_gate_pass": bool(seen_budget >= float(thresholds["seen_budget_min_nats"])),
        "unseen_gate_pass": bool(unseen_headroom >= float(thresholds["unseen_open_field_min_nats"])),
        "oracle_is_non_deployable_diagnostic_only": True,
    }
    output = Path(config["paths"]["metrics"]) / "signal_audit.json"
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    main()
