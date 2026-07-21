from __future__ import annotations

import pytest

from responsevec.data import PanelStore, make_synthetic_panels
from responsevec.protocols import (
    assert_fixed_target_set,
    assert_no_leakage,
    build_item_folds,
    build_protocol_a,
    build_protocol_b,
    build_protocol_b_candidates,
    build_protocol_c,
)


@pytest.fixture(scope="module")
def store():
    frame, orders = make_synthetic_panels(
        n_panels=40, n_items_per_domain=14, n_unseen_items=2,
        domains=("Environment", "Role of Government"), seed=3,
    )
    return PanelStore(frame, orders)


def test_protocol_a_target_never_in_own_history(store):
    units = build_protocol_a(store, "test")
    assert units
    summary = assert_no_leakage(units)
    assert summary["units_checked"] == len(units)
    assert all(u.item_pool == "seen" for u in units)


def test_item_folds_partition_is_disjoint_and_covers(store):
    folds = build_item_folds(store, calibration_pool_fraction=0.4, calibration_pool_cap=8, n_folds=6, seed=1)
    for domain, target_folds in folds.target_folds.items():
        pool = folds.calibration_pool[domain]
        seen_items = set(
            store.responses.loc[store.responses["domain"].eq(domain), "question_key"].unique()
        )
        union = set(pool)
        for i, fold_a in enumerate(target_folds):
            assert pool.isdisjoint(fold_a)                    # pool never a target
            for j, fold_b in enumerate(target_folds):
                if i != j:
                    assert fold_a.isdisjoint(fold_b)          # folds mutually disjoint
            union |= set(fold_a)
        assert union == seen_items                            # pool + folds cover all seen items


def test_protocol_b_history_only_from_calibration_pool(store):
    folds = build_item_folds(store, 0.4, 8, 6, seed=1)
    units = build_protocol_b(store, folds, "test", eval_fold=0)
    assert units
    assert all(u.item_pool == "unseen" and u.fold == 0 for u in units)
    # every eligible history key must live in the calibration pool
    for u in units:
        assert u.eligible_source_keys <= folds.calibration_pool[u.domain]
    assert_no_leakage(units, folds=folds)                     # raises if any fold item leaks


def test_protocol_b_target_fold_items_never_leak_across_folds(store):
    folds = build_item_folds(store, 0.4, 8, 6, seed=1)
    all_units = []
    for fold in range(6):
        all_units.extend(build_protocol_b(store, folds, "test", eval_fold=fold))
    # union of all target question_keys must be disjoint from every eligible history set
    target_keys = {u.question_key for u in all_units}
    for u in all_units:
        assert u.eligible_source_keys.isdisjoint(target_keys)


def test_protocol_b_role_partition_is_two_thirds_one_sixth_one_sixth(store):
    folds = build_item_folds(store, 0.4, 8, 6, seed=1)
    for domain in folds.target_folds:
        train = folds.role_items(domain, 0, "train")
        validation = folds.role_items(domain, 0, "validation")
        test = folds.role_items(domain, 0, "test")
        assert train.isdisjoint(validation | test)
        assert validation.isdisjoint(test)
        candidate = frozenset().union(*folds.target_folds[domain])
        assert train | validation | test == candidate
        # Outer fold 0 uses folds 2..5 for training: an exact 4:1:1 allocation
        # in fold units even when integer item counts differ by one.
        assert train == frozenset().union(*folds.target_folds[domain][2:])


def test_protocol_b_train_and_validation_roles_use_declared_items(store):
    folds = build_item_folds(store, 0.4, 8, 6, seed=1)
    for role, split in (("train", "train"), ("validation", "validation"), ("test", "test")):
        units = build_protocol_b(store, folds, split, outer_fold=0, target_role=role)
        assert units
        assert all(u.question_key in folds.role_items(u.domain, 0, role) for u in units)
        assert_no_leakage(units, folds)


def test_protocol_b_candidate_cache_is_fold_independent(store):
    folds = build_item_folds(store, 0.4, 8, 6, seed=1)
    units = build_protocol_b_candidates(store, folds, "test")
    assert units
    assert all(unit.target_role == "candidate" for unit in units)
    candidate_items = {
        key for domain_folds in folds.target_folds.values()
        for item_fold in domain_folds for key in item_fold
    }
    assert {unit.question_key for unit in units} == candidate_items
    assert_no_leakage(units, folds)


def test_protocol_c_targets_only_held_out_domain(store):
    units = build_protocol_c(store, held_out_domain="Environment", split="test")
    assert units
    assert all(u.domain == "Environment" for u in units)
    assert_no_leakage(units)


def test_fixed_target_set_invariant(store):
    units = build_protocol_a(store, "test")
    summary = assert_fixed_target_set(units, k_values=[0, 1, 3, 5, 8])
    assert summary["k_invariant"] is True
    assert summary["n_targets"] == len({(u.panel_id, u.question_key) for u in units})
