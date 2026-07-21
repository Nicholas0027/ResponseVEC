from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from responsevec.baselines.item_conditional import ItemConditionalModel
from responsevec.item_graph import ItemGraph, compute_item_graph
from responsevec.prior import PopulationPrior


def _world(n_panels: int = 120, seed: int = 0):
    """Train world where q_a and q_b are near-perfectly coupled (same latent),
    q_c is independent noise. All 3-option items, one domain, one country."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_panels):
        latent = rng.integers(0, 3)
        a = latent
        b = latent if rng.random() < 0.9 else int(rng.integers(0, 3))
        c = int(rng.integers(0, 3))
        for q, y in [("d::qa", a), ("d::qb", b), ("d::qc", c)]:
            rows.append(
                {
                    "panel_id": f"p{i}", "domain": "d", "country": "X", "question_key": q,
                    "answer_index": int(y), "n_options": 3, "is_unseen_item": False, "split": "train",
                    "normalized_answer": y / 2.0,
                }
            )
    frame = pd.DataFrame(rows)
    prior = PopulationPrior().fit(frame)
    graph = ItemGraph(compute_item_graph(frame, shrinkage_lambda=5, min_n_jk=10))
    model = ItemConditionalModel(prior, graph, min_pair_count=20).fit(frame)
    return model, prior


def test_k0_reduces_exactly_to_prior():
    model, prior = _world()
    p = model.predict_one("d::qa", "d", "X", 3, history=[])
    np.testing.assert_allclose(p, prior.predict("d::qa", "X", 3, True), rtol=1e-9)


def test_correlated_history_shifts_prediction_toward_conditional():
    model, prior = _world()
    p_prior = prior.predict("d::qb", "X", 3, True)
    p_given = model.predict_one("d::qb", "d", "X", 3, history=[("d::qa", 2)])
    # Knowing qa=2 must raise qb=2 well above its prior probability.
    assert p_given[2] > p_prior[2] + 0.15
    assert p_given.sum() == pytest.approx(1.0)


def test_uncorrelated_history_barely_moves_prediction():
    model, prior = _world()
    p_prior = prior.predict("d::qb", "X", 3, True)
    p_given = model.predict_one("d::qb", "d", "X", 3, history=[("d::qc", 2)])
    # qc is independent -> |C| ~ 0 -> weight ~ 0 -> near-prior prediction.
    assert abs(p_given[2] - p_prior[2]) < 0.1


def test_unknown_history_item_is_skipped_not_error():
    model, _ = _world()
    p = model.predict_one("d::qb", "d", "X", 3, history=[("d::missing", 1)])
    assert p.sum() == pytest.approx(1.0)


def test_rejects_non_train_fit_and_unseen_prediction():
    model, prior = _world()
    frame = pd.DataFrame(
        [{"panel_id": "p", "domain": "d", "country": "X", "question_key": "q", "answer_index": 0,
          "n_options": 3, "is_unseen_item": False, "split": "test", "normalized_answer": 0.0}]
    )
    with pytest.raises(ValueError, match="train"):
        ItemConditionalModel(prior, ItemGraph({}), min_pair_count=1).fit(frame)

    from responsevec.baselines.item_conditional import predict_item_conditional

    with pytest.raises(ValueError, match="unseen"):
        predict_item_conditional(None, model, [0], 0, "/tmp/x.parquet", item_pool="unseen")


def test_conditional_tables_exclude_unseen_items():
    rows = []
    rng = np.random.default_rng(1)
    for i in range(60):
        latent = int(rng.integers(0, 3))
        rows.append({"panel_id": f"p{i}", "domain": "d", "country": "X", "question_key": "d::seen",
                     "answer_index": latent, "n_options": 3, "is_unseen_item": False, "split": "train",
                     "normalized_answer": latent / 2.0})
        rows.append({"panel_id": f"p{i}", "domain": "d", "country": "X", "question_key": "d::unseen",
                     "answer_index": latent, "n_options": 3, "is_unseen_item": True, "split": "train",
                     "normalized_answer": latent / 2.0})
    frame = pd.DataFrame(rows)
    prior = PopulationPrior().fit(frame)
    seen_only = frame[~frame["is_unseen_item"]]
    graph = ItemGraph(compute_item_graph(seen_only, shrinkage_lambda=5, min_n_jk=10))
    model = ItemConditionalModel(prior, graph, min_pair_count=10).fit(frame)
    assert not any("d::unseen" in key for pair in model.conditionals for key in pair)
