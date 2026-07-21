from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from responsevec.eval.metrics import add_probability_scores


def _row(probability, answer):
    return {"probabilities_json": json.dumps(list(probability)), "answer_index": answer}


def test_rps_perfect_prediction_is_zero():
    frame = pd.DataFrame([_row([0, 0, 1, 0, 0], 2)])
    assert add_probability_scores(frame)["rps"].iloc[0] == pytest.approx(0.0)


def test_rps_hand_computed():
    # p = (0.5, 0.5, 0), truth = category 2 (last).
    # CDF_pred = (0.5, 1.0), CDF_true = (0, 0) at thresholds 0,1.
    # RPS = (0.25 + 1.0) / (3-1) = 0.625
    frame = pd.DataFrame([_row([0.5, 0.5, 0.0], 2)])
    assert add_probability_scores(frame)["rps"].iloc[0] == pytest.approx(0.625)


def test_rps_penalizes_distance_ordinally():
    # Mass adjacent to the truth must score better than mass far away —
    # the property accuracy and plain NLL both lack.
    near = pd.DataFrame([_row([0.0, 1.0, 0.0, 0.0, 0.0], 0)])
    far = pd.DataFrame([_row([0.0, 0.0, 0.0, 0.0, 1.0], 0)])
    assert add_probability_scores(near)["rps"].iloc[0] < add_probability_scores(far)["rps"].iloc[0]


def test_rps_flows_into_macro_metrics():
    from responsevec.eval.metrics import respondent_macro_metrics

    rows = []
    for i in range(4):
        rows.append(
            {
                **_row([0.7, 0.2, 0.1], 0), "method": "m", "k": 0, "split": "test", "item_pool": "seen",
                "domain": "d", "panel_id": f"p{i}", "correct": 1, "normalized_ordinal_error": 0.0,
                "nll": 0.36, "brier": 0.14,
            }
        )
    macro = respondent_macro_metrics(pd.DataFrame(rows))
    assert "rps" in macro.columns
    assert macro["rps"].notna().all()
