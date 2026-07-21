from __future__ import annotations

import numpy as np

from responsevec.baselines.representation import (
    build_pplug_matrix,
    pplug_history_z,
    sentence_encoder_z,
)


def test_sentence_encoder_z_is_passthrough_float32():
    emb = np.random.default_rng(0).normal(size=(5, 8))
    z = sentence_encoder_z(emb)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z, emb.astype(np.float32))


def test_pplug_empty_history_is_zero_vector():
    z = pplug_history_z(np.ones(4), [])
    assert z.shape == (4,)
    assert np.all(z == 0.0)


def test_pplug_weights_toward_similar_answer():
    # target aligned with the first answer embedding; pplug z should lean toward it.
    target = np.array([1.0, 0.0])
    answers = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    z = pplug_history_z(target, answers, temperature=0.2)
    # z is a convex combo; closer to answers[0] than answers[1]
    assert np.linalg.norm(z - answers[0]) < np.linalg.norm(z - answers[1])


def test_pplug_uniform_when_equal_similarity():
    target = np.array([1.0, 1.0])
    answers = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]  # symmetric wrt target
    z = pplug_history_z(target, answers, temperature=1.0)
    np.testing.assert_allclose(z, [0.5, 0.5], atol=1e-6)


def test_build_pplug_matrix_shapes():
    targets = np.random.default_rng(1).normal(size=(3, 6))
    histories = [
        [np.ones(6), np.zeros(6)],
        [],
        [np.ones(6)],
    ]
    matrix = build_pplug_matrix(targets, histories)
    assert matrix.shape == (3, 6)
    assert np.all(matrix[1] == 0.0)  # empty history -> zero row
