from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from responsevec.prior import PopulationPrior


def _rows(question, country, answers, n_options=5, unseen=False, split="train"):
    return [
        {"question_key": question, "country": country, "answer_index": a, "n_options": n_options,
         "is_unseen_item": unseen, "split": split}
        for a in answers
    ]


def _frame(rows):
    return pd.DataFrame(rows)


def test_rejects_non_train_rows():
    frame = _frame(_rows("q1", "A", [0, 1], split="test"))
    with pytest.raises(ValueError, match="train"):
        PopulationPrior().fit(frame)


def test_question_marginal_hand_computed():
    # 8 answers on q1: [0]*4 + [1]*3 + [2]*1 -> laplace 0.5 -> (4.5,3.5,1.5,.5,.5)/10.5
    frame = _frame(_rows("q1", "A", [0] * 4 + [1] * 3 + [2]))
    prior = PopulationPrior(laplace=0.5).fit(frame)
    # Unknown country falls back to the question marginal.
    p = prior.predict("q1", "NEVER", 5, item_is_seen=True)
    expected = np.array([4.5, 3.5, 1.5, 0.5, 0.5]) / 10.5
    np.testing.assert_allclose(p, expected, rtol=1e-9)


def test_country_shrinkage_blends_toward_question_marginal():
    # Country B has only 2 answers, both option 4; question marginal is flat-ish.
    rows = _rows("q1", "A", [0, 1, 2, 3, 4] * 8) + _rows("q1", "B", [4, 4])
    prior = PopulationPrior(country_shrinkage=20.0).fit(_frame(rows))
    p_b = prior.predict("q1", "B", 5, item_is_seen=True)
    p_q = prior.predict("q1", "NEVER", 5, item_is_seen=True)
    # Small country cell -> heavily shrunk toward question marginal, but still
    # tilted toward option 4 relative to it.
    assert p_b[4] > p_q[4]
    assert p_b[4] < 0.9  # not the raw country MLE (which would be ~1.0)
    assert p_b.sum() == pytest.approx(1.0)


def test_unseen_items_use_position_marginal_never_item_stats():
    rows = _rows("q_seen", "A", [0] * 10) + _rows("q_unseen", "A", [4] * 10, unseen=True)
    prior = PopulationPrior().fit(_frame(rows))
    # No per-item table may exist for the unseen item.
    assert "q_unseen" not in prior.question_tables
    p = prior.predict("q_unseen", "A", 5, item_is_seen=False)
    # Position marginal comes from SEEN items only (all option 0 here), so the
    # unseen item's true answers (all option 4) must NOT leak into it.
    assert p[0] > p[4]


def test_unknown_scale_length_falls_back_to_uniform():
    prior = PopulationPrior().fit(_frame(_rows("q1", "A", [0, 1], n_options=5)))
    p = prior.predict("q_new", "A", 7, item_is_seen=False)
    np.testing.assert_allclose(p, np.full(7, 1 / 7))


def test_save_load_round_trip(tmp_path):
    rows = _rows("q1", "A", [0, 1, 2]) + _rows("q1", "B", [3, 3])
    prior = PopulationPrior().fit(_frame(rows))
    prior.save(tmp_path)
    loaded = PopulationPrior.load(tmp_path)
    for args in [("q1", "A", 5, True), ("q1", "B", 5, True), ("qx", "A", 5, False)]:
        np.testing.assert_allclose(loaded.predict(*args), prior.predict(*args))


def test_protocol_b_allowed_keys_prevent_item_prior_leakage():
    frame = _frame(_rows("q_train", "A", [0, 1, 1]) + _rows("q_test", "A", [4, 4, 4]))
    prior = PopulationPrior().fit(frame, allowed_question_keys={"q_train"})
    assert "q_train" in prior.question_tables
    assert "q_test" not in prior.question_tables
    prediction = prior.predict("q_test", "A", 5, item_is_seen=False)
    assert np.isclose(prediction.sum(), 1.0)
    assert prediction[4] < 0.5  # held-out q_test labels did not dominate the fallback
