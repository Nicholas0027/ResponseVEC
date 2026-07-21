"""Tests for Protocol D (R3 OOD-demographic-intersection)."""

from __future__ import annotations

import pandas as pd
import pytest

from responsevec.protocols import (
    EvalUnit,
    assert_no_leakage,
    build_protocol_a,
    build_protocol_d,
)


def _store_with_ood():
    """Minimal PanelStore-like object with an ood_intersection split."""
    rows = []
    for pid in ("p1", "p2"):
        for qid in ("q1", "q2", "q3"):
            rows.append({
                "row_id": f"{pid}::{qid}", "panel_id": pid, "domain": "Environment",
                "question_id": qid, "question_key": f"Environment::{qid}",
                "question": f"Q {qid}?", "options_json": '["a","b"]', "n_options": 2,
                "answer_index": 0, "answer_text": "a", "is_unseen_item": False,
                "split": "ood_intersection", "survey_weight": 1.0,
                "demographic_text": "demo", "country": "X", "sex": "F",
                "age_bin": "30-44", "education": "Tertiary",
            })
    return type("S", (), {"responses": pd.DataFrame(rows)})()


def test_protocol_d_uses_ood_split():
    store = _store_with_ood()
    units = build_protocol_d(store, split="ood_intersection")
    assert len(units) == 6
    assert all(u.protocol == "A" for u in units)  # reuses Protocol A rule
    assert all(u.item_pool == "seen" for u in units)


def test_protocol_d_rejects_non_ood_split():
    store = _store_with_ood()
    with pytest.raises(ValueError, match="ood_intersection"):
        build_protocol_d(store, split="train")


def test_protocol_d_no_self_leakage():
    store = _store_with_ood()
    units = build_protocol_d(store, split="ood_intersection")
    summary = assert_no_leakage(units)
    assert summary["units_checked"] == 6
