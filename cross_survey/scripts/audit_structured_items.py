#!/usr/bin/env python3
"""Audit structured-item validity and v1/v2 stability before calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/results/structured_items_qwen32b.jsonl")
    parser.add_argument("--output", default="cross_survey/results/structured_items_audit.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines()
               if line.strip()]
    valid = [record for record in records if record.get("valid")]
    table = {(record["item"], record["template"]): record for record in valid}
    item_types, scale_types, stance_a, stance_b, intensity_a, intensity_b = [], [], [], [], [], []
    paired_items = []
    for item in sorted({record["item"] for record in valid}):
        a, b = table.get((item, "v1")), table.get((item, "v2"))
        if not a or not b:
            continue
        paired_items.append(item)
        item_types.append(a["data"]["item_type"] == b["data"]["item_type"])
        scale_types.append(a["data"]["scale_type"] == b["data"]["scale_type"])
        options_a = {int(x["code"]): x for x in a["data"]["options"]}
        options_b = {int(x["code"]): x for x in b["data"]["options"]}
        for code in sorted(set(options_a) & set(options_b)):
            stance_a.append(float(options_a[code]["stance_direction"]))
            stance_b.append(float(options_b[code]["stance_direction"]))
            intensity_a.append(float(options_a[code]["behavior_intensity"]))
            intensity_b.append(float(options_b[code]["behavior_intensity"]))
    stance_rho = float(spearmanr(stance_a, stance_b).statistic) if len(stance_a) >= 3 else 0.0
    intensity_rho = float(spearmanr(intensity_a, intensity_b).statistic) if len(intensity_a) >= 3 else 0.0
    payload = {
        "records": len(records), "valid_records": len(valid),
        "valid_rate": len(valid) / max(len(records), 1),
        "paired_items": len(paired_items),
        "item_type_agreement": float(np.mean(item_types)) if item_types else 0.0,
        "scale_type_agreement": float(np.mean(scale_types)) if scale_types else 0.0,
        "option_stance_spearman": stance_rho,
        "option_intensity_spearman": intensity_rho,
        "gate_thresholds": {"valid_rate": 0.90, "stance_spearman": 0.50},
    }
    payload["stable_enough_to_calibrate"] = bool(
        payload["valid_rate"] >= 0.90 and stance_rho >= 0.50
    )
    errors = pd.Series([record.get("error") for record in records if not record.get("valid")])
    payload["error_counts"] = errors.value_counts().head(20).to_dict()
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
