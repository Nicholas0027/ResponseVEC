#!/usr/bin/env python3
"""Audit CES 2020 for same-person pre/post closed-choice transfer.

This stage is outcome-agnostic: it inventories coverage and item shapes but does
not fit any predictive model or choose targets by performance. Candidate items
are declared from variable names and response support alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_RE = re.compile(r"^CC20_3")
TARGET_RE = re.compile(r"^CC20_4")
EXCLUDE_RE = re.compile(
    r"(_t$|_timing$|_other$|_dk_flag$|_nv$|_GA\d+$|^page_|text|name|party)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(float)
    return float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())


def item_record(name: str, values: pd.Series, n_paired: int, wave: str) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    unique = sorted(float(v) for v in observed.unique())
    return {
        "item": name,
        "wave": wave,
        "paired_nonmissing": int(len(observed)),
        "paired_coverage": float(len(observed) / max(n_paired, 1)),
        "n_unique": len(unique),
        "values": unique[:50],
        "min": float(observed.min()) if len(observed) else None,
        "max": float(observed.max()) if len(observed) else None,
        "entropy_nats": entropy(observed) if len(observed) else None,
        "eligible_closed_choice": bool(
            len(observed) >= 2000
            and 2 <= len(unique) <= 11
            and len(observed) / max(n_paired, 1) >= 0.50
        ),
    }


def describe_counts(values: pd.Series) -> dict:
    quantiles = values.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {str(index): float(value) for index, value in quantiles.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv",
    )
    parser.add_argument(
        "--catalog", default="cross_survey/metadata/ces2020_item_catalog.csv"
    )
    parser.add_argument(
        "--inventory", default="cross_survey/metadata/dataset_inventory.json"
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)
    # The release is 61k x 717 and contains hundreds of large text/timing
    # columns. Loading all of them exceeds the editor sandbox memory budget and
    # is unnecessary for a schema audit. Read the header first, then only IDs,
    # post-completion marker, and CC20_3xx/4xx candidates.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    selected = ["caseid", "starttime_post"] + [
        column for column in header
        if (SOURCE_RE.match(column) or TARGET_RE.match(column))
        and not EXCLUDE_RE.search(column)
    ]
    frame = pd.read_csv(path, usecols=selected, low_memory=False)
    if frame.caseid.nunique() != len(frame):
        raise ValueError("caseid is not unique")
    paired = frame[frame.starttime_post.notna()].copy()
    n_paired = len(paired)

    candidates: list[tuple[str, str]] = []
    exclusions = {"pattern": [], "nonnumeric": []}
    for column in frame.columns:
        wave = "source_pre" if SOURCE_RE.match(column) else (
            "target_post" if TARGET_RE.match(column) else None
        )
        if wave is None:
            continue
        if EXCLUDE_RE.search(column):
            exclusions["pattern"].append(column)
            continue
        numeric = pd.to_numeric(paired[column], errors="coerce")
        if numeric.notna().sum() == 0:
            exclusions["nonnumeric"].append(column)
            continue
        candidates.append((column, wave))

    records = [item_record(name, paired[name], n_paired, wave)
               for name, wave in candidates]
    catalog = pd.DataFrame(records).sort_values(["wave", "item"])
    catalog_path = Path(args.catalog)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)

    eligible_source = catalog[
        catalog.wave.eq("source_pre") & catalog.eligible_closed_choice
    ].item.tolist()
    eligible_target = catalog[
        catalog.wave.eq("target_post") & catalog.eligible_closed_choice
    ].item.tolist()
    source_counts = paired[eligible_source].notna().sum(axis=1)
    target_counts = paired[eligible_target].notna().sum(axis=1)
    joint = (source_counts >= 20) & (target_counts >= 10)

    inventory = {
        "dataset": "ces2020",
        "input": str(path),
        "input_sha256": sha256(path),
        "rows": int(len(frame)),
        "unique_respondents": int(frame.caseid.nunique()),
        "paired_pre_post_respondents": n_paired,
        "source_candidates": int((catalog.wave == "source_pre").sum()),
        "target_candidates": int((catalog.wave == "target_post").sum()),
        "eligible_source_items": len(eligible_source),
        "eligible_target_items": len(eligible_target),
        "respondents_with_20_source_10_target": int(joint.sum()),
        "source_nonmissing_quantiles": describe_counts(source_counts),
        "target_nonmissing_quantiles": describe_counts(target_counts),
        "eligibility_rule": {
            "minimum_nonmissing": 2000,
            "minimum_paired_coverage": 0.50,
            "minimum_categories": 2,
            "maximum_categories": 11,
        },
        "C0_data_viability": bool(
            joint.sum() >= 2000
            and len(eligible_source) >= 20
            and len(eligible_target) >= 10
        ),
        "excluded_columns": exclusions,
        "notes": [
            "Candidate declaration uses names and coverage only; no target labels or model results.",
            "Question wording and response labels must be joined from the questionnaires before modelling.",
            "Repeated pre/post items and branching/randomization are not yet removed.",
        ],
    }
    inventory_path = Path(args.inventory)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    print(json.dumps({
        key: inventory[key] for key in (
            "rows", "paired_pre_post_respondents", "eligible_source_items",
            "eligible_target_items", "respondents_with_20_source_10_target",
            "C0_data_viability",
        )
    }, indent=2))
    print(f"wrote {catalog_path}")
    print(f"wrote {inventory_path}")


if __name__ == "__main__":
    main()
