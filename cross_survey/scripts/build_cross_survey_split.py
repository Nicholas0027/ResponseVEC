#!/usr/bin/env python3
"""Lock the preliminary CES source/target and respondent split manifest.

Selection uses coverage, marginal entropy, and item-family diversity only. No
predictive model is run before this manifest is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def family(item: str) -> str:
    # CES grids append letters and/or numeric punches (e.g. 442a--442e,
    # 430a_1--430a_8). All belong to one questionnaire block and must not
    # occupy multiple target slots in this preliminary family-balanced split.
    match = re.match(r"^(CC20_\d+)", item, re.IGNORECASE)
    return match.group(1) if match else item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", default="cross_survey/metadata/ces2020_item_catalog.csv"
    )
    parser.add_argument(
        "--inventory", default="cross_survey/metadata/dataset_inventory.json"
    )
    parser.add_argument(
        "--output", default="cross_survey/metadata/ces2020_split_manifest.json"
    )
    parser.add_argument("--targets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    inventory = json.loads(Path(args.inventory).read_text())
    eligible = catalog[catalog.eligible_closed_choice].copy()
    eligible["family"] = eligible.item.map(family)

    source = sorted(eligible[eligible.wave.eq("source_pre")].item.tolist())
    targets = eligible[
        eligible.wave.eq("target_post") & (eligible.paired_coverage >= 0.95)
    ].copy()
    # One representative per question family prevents one multi-select grid
    # from occupying the whole target set. Within family choose coverage first,
    # then entropy: this is outcome distribution information, not model skill.
    targets = targets.sort_values(
        ["family", "paired_coverage", "entropy_nats", "item"],
        ascending=[True, False, False, True],
    ).drop_duplicates("family")
    targets = targets.sort_values(
        ["paired_coverage", "entropy_nats", "family"],
        ascending=[False, False, True],
    ).head(args.targets)
    target_items = targets.item.tolist()

    if len(source) < 40 or len(target_items) < 10:
        raise RuntimeError("C0 failed after family-balanced target selection")
    payload = {
        "dataset": "ces2020",
        "phase": "preliminary_cpu_ceiling",
        "post_freeze": True,
        "seed": args.seed,
        "input_sha256": inventory["input_sha256"],
        "source_items": source,
        "source_item_count": len(source),
        "target_items": target_items,
        "target_item_count": len(target_items),
        "target_families": targets.family.tolist(),
        "target_selection": (
            "eligible coverage>=0.95; one item per variable family; rank by "
            "coverage then marginal entropy; fixed before model fitting"
        ),
        "respondent_split": {
            "algorithm": "sha256(f'{seed}|{caseid}') first uint64 modulo 10",
            "train_buckets": [0, 1, 2, 3, 4, 5],
            "validation_buckets": [6, 7],
            "test_buckets": [8, 9],
        },
        "k_grid": [0, 1, 3, 5, 10, 20, 40, "full"],
        "initial_k_grid": [0, 5, 20, "full"],
        "initial_history_seeds": [1701],
        "primary_metrics": ["accuracy", "nll"],
        "warning": (
            "Question wording, ordinal status, repeated concepts, and branch "
            "logic are not yet adjudicated. This is a preliminary ceiling, "
            "not the final benchmark split."
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_items": len(source),
        "target_items": target_items,
        "target_families": payload["target_families"],
        "manifest_sha256": payload["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
