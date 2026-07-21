from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from responsevec.data import PanelStore, make_synthetic_panels
from responsevec.pipeline import materialize_respondent_prompts, respondentvec_cache_directory
from responsevec.prompting_rv import build_respondent_prompt
from responsevec.protocols import (
    RespondentUnit,
    build_item_folds,
    build_protocol_b_candidates,
    build_respondentvec_units,
)
from responsevec.training import HeadArrays, arrays_from_respondent_cache


@pytest.fixture(scope="module")
def store():
    frame, orders = make_synthetic_panels(
        n_panels=40, n_items_per_domain=14, n_unseen_items=2,
        domains=("Environment", "Role of Government"), seed=3,
    )
    return PanelStore(frame, orders)


@pytest.fixture(scope="module")
def folds(store):
    return build_item_folds(store, calibration_pool_fraction=0.4, calibration_pool_cap=8, n_folds=6, seed=1)


# --- build_respondent_prompt: target-free by construction ------------------


def test_respondent_prompt_never_mentions_a_target_question():
    history = [{"question": "climate change concern level", "answer_text": "Very concerned"}]
    prompt = build_respondent_prompt("Country: A; Age group: 30-44", history)
    assert "Target question" not in prompt
    assert "Options:" not in prompt
    assert "Respondent background:" in prompt
    assert "climate change concern level" in prompt


def test_respondent_prompt_empty_history_renders_placeholder():
    prompt = build_respondent_prompt("Country: A", [])
    assert "No relevant previous responses are available." in prompt


# --- build_respondentvec_units: dedup per (panel_id, domain) ---------------


def test_respondentvec_units_one_per_panel_domain(store, folds):
    units = build_respondentvec_units(store, folds, "test")
    candidates = build_protocol_b_candidates(store, folds, "test")
    expected_keys = {(u.panel_id, u.domain) for u in candidates}
    assert {(u.panel_id, u.domain) for u in units} == expected_keys
    # No duplicates: exactly one RespondentUnit per (panel_id, domain).
    seen = [(u.panel_id, u.domain) for u in units]
    assert len(seen) == len(set(seen))


def test_respondentvec_units_share_eligible_sources_with_candidates(store, folds):
    units = {(u.panel_id, u.domain): u for u in build_respondentvec_units(store, folds, "test")}
    for candidate in build_protocol_b_candidates(store, folds, "test"):
        key = (candidate.panel_id, candidate.domain)
        # Every candidate target for this respondent must see the IDENTICAL
        # eligible history the RespondentVec unit was built from — this is the
        # leakage-safety argument RespondentUnit's docstring relies on.
        assert units[key].eligible_source_keys == candidate.eligible_source_keys


def test_respondentvec_units_eligible_sources_within_calibration_pool(store, folds):
    for unit in build_respondentvec_units(store, folds, "test"):
        pool = folds.calibration_pool[unit.domain]
        assert unit.eligible_source_keys <= pool


# --- materialize_respondent_prompts -----------------------------------------


def test_materialize_respondent_prompts_uses_history_selection_all(store, folds):
    units = build_respondentvec_units(store, folds, "test")
    rows, prompts = materialize_respondent_prompts(store, units, k=3)
    assert len(rows) == len(units) == len(prompts)
    assert set(rows.columns) >= {"row_id", "panel_id", "domain", "k", "prompt_hash"}
    assert (rows["k"] == 3).all()
    # row_id must be unique per (panel_id, domain, k) -- no target axis at all.
    assert rows["row_id"].is_unique


def test_materialize_respondent_prompts_k0_has_no_history(store, folds):
    units = build_respondentvec_units(store, folds, "test")
    _, prompts = materialize_respondent_prompts(store, units, k=0)
    assert all("No relevant previous responses are available." in p for p in prompts)


def test_materialize_respondent_prompts_deterministic(store, folds):
    units = build_respondentvec_units(store, folds, "test")
    _, prompts_a = materialize_respondent_prompts(store, units, k=3, history_seed=7)
    _, prompts_b = materialize_respondent_prompts(store, units, k=3, history_seed=7)
    assert prompts_a == prompts_b


# --- respondentvec_cache_directory: fold-independent, no option_seed axis --


def test_cache_directory_has_no_fold_or_option_seed_axis():
    a = respondentvec_cache_directory("/root", respondent_split="test", k=5)
    b = respondentvec_cache_directory("/root", respondent_split="test", k=5)
    assert a == b
    assert "option_" not in str(a)
    assert "fold_" not in str(a)
    different_k = respondentvec_cache_directory("/root", respondent_split="test", k=8)
    assert different_k != a


# --- arrays_from_respondent_cache: join-adapter -----------------------------


class _FakeCache:
    def __init__(self, rows: pd.DataFrame, vectors: np.ndarray):
        self._rows = rows
        self._vectors = vectors

    def read_rows(self):
        return self._rows

    def read_vectors(self):
        return self._vectors


def _target_arrays(panel_ids, domains):
    n = len(panel_ids)
    rows = pd.DataFrame({"panel_id": panel_ids, "domain": domains, "question_key": [f"q{i}" for i in range(n)]})
    return HeadArrays(
        rows=rows,
        z=np.zeros((n, 4), dtype=np.float32),  # placeholder z, to be REPLACED by the join
        option_matrix=np.ones((n, 3, 2), dtype=np.float32),
        option_mask=np.ones((n, 3), dtype=np.float32),
        log_prior=np.zeros((n, 3), dtype=np.float32),
        targets=np.zeros(n, dtype=np.int64),
        ordinal_mask=np.ones(n, dtype=bool),
        direct_probabilities=None,
    )


def test_join_replaces_z_and_preserves_target_specific_arrays():
    target = _target_arrays(["p1", "p2"], ["D", "D"])
    target.option_matrix[0] = 11.0  # mark row 0 so we can verify it survives the join untouched
    respondent_rows = pd.DataFrame({"panel_id": ["p1", "p2"], "domain": ["D", "D"]})
    respondent_vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cache = _FakeCache(respondent_rows, respondent_vectors)

    joined = arrays_from_respondent_cache(target, cache)
    assert joined.z.shape == (2, 2)
    np.testing.assert_array_equal(joined.z, respondent_vectors)
    # option_matrix/log_prior/targets are untouched target-specific arrays.
    np.testing.assert_array_equal(joined.option_matrix, target.option_matrix)
    np.testing.assert_array_equal(joined.targets, target.targets)


def test_join_drops_rows_with_no_cached_respondent_vector():
    target = _target_arrays(["p1", "p2", "p3"], ["D", "D", "D"])
    respondent_rows = pd.DataFrame({"panel_id": ["p1", "p3"], "domain": ["D", "D"]})  # p2 missing
    respondent_vectors = np.array([[1.0], [2.0]], dtype=np.float32)
    cache = _FakeCache(respondent_rows, respondent_vectors)

    joined = arrays_from_respondent_cache(target, cache)
    assert len(joined) == 2
    assert set(joined.rows["panel_id"]) == {"p1", "p3"}


def test_join_raises_when_nothing_matches():
    target = _target_arrays(["p1"], ["D"])
    respondent_rows = pd.DataFrame({"panel_id": ["other"], "domain": ["D"]})
    cache = _FakeCache(respondent_rows, np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="no target rows matched"):
        arrays_from_respondent_cache(target, cache)


def test_join_distinguishes_by_domain_not_panel_id_alone():
    """The same panel_id in two different domains must get different vectors —
    the join key is (panel_id, domain), never panel_id alone."""
    target = _target_arrays(["p1", "p1"], ["D1", "D2"])
    respondent_rows = pd.DataFrame({"panel_id": ["p1", "p1"], "domain": ["D1", "D2"]})
    respondent_vectors = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
    cache = _FakeCache(respondent_rows, respondent_vectors)

    joined = arrays_from_respondent_cache(target, cache)
    d1_row = joined.rows.reset_index(drop=True)
    d1_vector = joined.z[d1_row["domain"] == "D1"][0]
    d2_vector = joined.z[d1_row["domain"] == "D2"][0]
    np.testing.assert_array_equal(d1_vector, [1.0, 1.0])
    np.testing.assert_array_equal(d2_vector, [9.0, 9.0])
