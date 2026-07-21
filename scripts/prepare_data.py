#!/usr/bin/env python
"""Prepare SocioBench (or deterministic synthetic smoke data) and item folds."""

from __future__ import annotations

import argparse
from pathlib import Path

from responsevec.config import load_and_prepare
from responsevec.data import PanelStore, assign_splits, make_synthetic_panels, prepare_sociobench
from responsevec.protocols import build_item_folds
from responsevec.utils import write_json


def save_synthetic(processed: Path, seed: int, config) -> None:
    responses, orders = make_synthetic_panels(
        n_panels=48, n_items_per_domain=18, n_unseen_items=0,
        domains=("Environment", "Role of Government", "Social Inequality", "Work Orientations"),
        seed=seed,
    )
    # Run the SAME split assignment the real path uses, so the synthetic data
    # has an ood_intersection split and Protocol D (R3) can be smoke-tested.
    data_cfg = config["data"]
    responses, _ = assign_splits(
        responses, seed, tuple(data_cfg["respondent_split"]),
        list(data_cfg["intersection_attributes"]),
        float(data_cfg["intersection_holdout_fraction"]),
        int(data_cfg["min_cell_size"]),
    )
    processed.mkdir(parents=True, exist_ok=True)
    responses.to_parquet(processed / "responses.parquet", index=False)
    orders.to_parquet(processed / "calibration_orders.parquet", index=False)
    responses.drop_duplicates("panel_id").to_parquet(processed / "panels.parquet", index=False)
    responses.drop_duplicates("question_key").to_parquet(processed / "items.parquet", index=False)
    write_json(processed / "audit.json", {
        "synthetic_smoke_only": True,
        "rows": len(responses), "panels": responses["panel_id"].nunique(),
        "items": responses["question_key"].nunique(),
        "splits": responses["split"].value_counts().to_dict(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--synthetic", action="store_true", help="write deterministic non-scientific smoke data")
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    processed = Path(config["paths"]["processed"])
    if args.synthetic:
        save_synthetic(processed, int(config["seed"]), config)
    else:
        prepare_sociobench(config)
    store = PanelStore.from_dir(processed)
    fold_config = config["item_folds"]
    folds = build_item_folds(
        store,
        float(fold_config["calibration_pool_fraction"]),
        int(fold_config["calibration_pool_cap"]),
        int(fold_config["n_folds"]),
        int(config["seed"]),
    )
    folds.save(processed / "item_folds.json")
    print({
        "processed": str(processed), "rows": len(store.responses),
        "panels": store.responses["panel_id"].nunique(),
        "items": store.responses["question_key"].nunique(), "folds": folds.n_folds,
    })


if __name__ == "__main__":
    main()
