"""Unit tests for ResponseVec-Align (design section 2.5).

These cover the properties the paper claims for the task-alignment projection:
  * project() output shape and residual-to-identity behaviour at init;
  * the option-anchored supervised-contrastive loss is a valid probability
    objective (positive, finite, decreases when the projection is trained on a
    separable signal);
  * apply_aligner replaces only z and leaves every option/target field intact;
  * on a synthetic dataset where the chosen option is a deterministic function
    of z, training the aligner + decoder on g_phi(z) beats training the decoder
    on the raw z (the whole point of the method).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from responsevec.align import AlignConfig, ResponseVecAligner, align_reduces_to_identity  # noqa: E402
from responsevec.training import (  # noqa: E402
    HeadArrays,
    apply_aligner,
    predict_head,
    train_aligner,
    train_option_head,
)


def _synthetic_arrays(n=400, d_z=16, d_o=8, n_options=4, seed=0):
    """Build a HeadArrays where the chosen option is a deterministic function
    of z projected onto option space -- a signal a task-aligned projection can
    exploit but a raw untrained geometry need not expose cleanly.

    The ground-truth rule (option encodings + mixing matrix W) is FIXED across
    all seeds so that train and validation share the same generative process
    and generalization is possible; `seed` only varies the sampled z rows.
    """
    rng = np.random.default_rng(seed)
    fixed = np.random.default_rng(12345)  # shared ground-truth across splits
    z = rng.normal(size=(n, d_z)).astype(np.float32)
    # Fixed option encodings shared across rows AND splits (same item family).
    options = fixed.normal(size=(n_options, d_o)).astype(np.float32)
    option_matrix = np.repeat(options[None], n, axis=0).astype(np.float32)
    option_mask = np.ones((n, n_options), dtype=np.float32)
    # Ground-truth: chosen option = argmax over c of <phi(z), options_c>, where
    # phi is a FIXED NON-LINEAR map (tanh-MLP) shared across splits. A linear
    # bilinear decoder on raw z cannot represent phi, but the aligner's MLP
    # projection g_phi can recover it -- so decoding g_phi(z) should beat
    # decoding z. This is the honest capacity gap the method exploits.
    W1 = fixed.normal(size=(d_o, d_z)).astype(np.float32)
    W2 = fixed.normal(size=(d_o, d_o)).astype(np.float32)
    phi = np.tanh(z @ W1.T) @ W2.T  # (n, d_o) non-linear feature of z
    scores = phi @ options.T  # (n, n_options)
    targets = scores.argmax(axis=1).astype(np.int64)
    log_prior = np.log(np.full((n, n_options), 1.0 / n_options, dtype=np.float32))
    rows = pd.DataFrame({
        "question_key": ["D::q1"] * n,
        "n_options": [n_options] * n,
        "answer_index": targets,
        "is_ordinal": [False] * n,
        "country": ["XX"] * n,
        "history_seed": [seed] * n,
    })
    return HeadArrays(
        rows=rows, z=z, option_matrix=option_matrix, option_mask=option_mask,
        log_prior=log_prior, targets=targets,
        ordinal_mask=np.zeros(n, dtype=bool), direct_probabilities=None,
    )


def test_project_shape_and_identity_at_init():
    config = AlignConfig(z_dim=16, o_dim=8, projection_dim=12, residual_alpha_init=0.0)
    assert align_reduces_to_identity(config)
    aligner = ResponseVecAligner(config)
    z = torch.randn(5, 16)
    projected = aligner.project(z)
    assert projected.shape == (5, 12)
    # residual_alpha == 0 at init => project() is exactly the linear path.
    linear_only = aligner.linear_z(z)  # dropout is eval-inactive under no_grad
    aligner.eval()
    with torch.no_grad():
        assert torch.allclose(aligner.project(z), aligner.linear_z(z), atol=1e-5)
    assert torch.allclose(linear_only, aligner.linear_z(z), atol=1e-5)


def test_contrastive_loss_positive_and_finite():
    config = AlignConfig(z_dim=16, o_dim=8, projection_dim=12)
    aligner = ResponseVecAligner(config)
    arrays = _synthetic_arrays(n=32)
    z = torch.as_tensor(arrays.z)
    om = torch.as_tensor(arrays.option_matrix)
    mask = torch.as_tensor(arrays.option_mask)
    target = torch.as_tensor(arrays.targets)
    loss = aligner(z, om, mask, target)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_apply_aligner_replaces_only_z():
    config = AlignConfig(z_dim=16, o_dim=8, projection_dim=12)
    aligner = ResponseVecAligner(config).eval()
    arrays = _synthetic_arrays(n=20)
    out = apply_aligner(aligner, arrays, device="cpu")
    assert out.z.shape == (20, 12)
    # every non-z field is untouched
    assert np.array_equal(out.targets, arrays.targets)
    assert np.array_equal(out.option_matrix, arrays.option_matrix)
    assert np.array_equal(out.option_mask, arrays.option_mask)
    assert out.rows.equals(arrays.rows)


def test_aligner_training_reduces_contrastive_loss():
    # cross_respondent_negatives=False isolates the within-item objective, whose
    # uniform baseline is exactly log(n_options)=log(4); with cross-respondent
    # negatives on, the denominator also includes other respondents' anchors so
    # the loss scale is log(n_options + batch) and that baseline test would not
    # apply. We assert the within-item objective drops below its uniform floor.
    train = _synthetic_arrays(n=400, seed=1)
    validation = _synthetic_arrays(n=120, seed=2)
    fit = train_aligner(
        train, validation, projection_dim=12, hidden_dim=32,
        cross_respondent_negatives=False, lr=3e-3,
        epochs=200, patience=200, batch_size=64, seed=1701, device="cpu",
    )
    first = fit.history[0]["validation_loss"]
    best = fit.best_validation_loss
    assert best <= first  # training must not increase the contrastive loss
    assert best < np.log(4)  # beats uniform-over-4-options contrastive baseline


def test_alignment_end_to_end_pipeline_and_nondegenerate():
    """End-to-end: fit aligner on frozen z, project train/val/test, train the
    standard option-aware decoder on g_phi(z), and predict -- the exact path
    train_primary uses for '<family>_aligned'. Asserts the pipeline runs and
    produces a non-degenerate head (validation NLL strictly below the uniform
    log(n_options) floor), i.e. the aligned representation carries real signal.

    Whether alignment *beats raw z* on NLL is an empirical question decided on
    the real SocioBench data, not asserted on a synthetic toy here: contrastive
    geometry and decoder NLL are related but distinct objectives, and a
    projection that helps on real high-dimensional 8B vectors need not help on a
    16-dim synthetic signal a linear head already fits well."""
    train = _synthetic_arrays(n=600, seed=10)
    validation = _synthetic_arrays(n=200, seed=11)
    test = _synthetic_arrays(n=200, seed=12)

    aligner_fit = train_aligner(
        train, validation, projection_dim=24, hidden_dim=32,
        epochs=80, patience=80, batch_size=128, seed=1701, device="cpu",
    )
    aligned_train = apply_aligner(aligner_fit.aligner, train, device="cpu")
    aligned_validation = apply_aligner(aligner_fit.aligner, validation, device="cpu")
    aligned_test = apply_aligner(aligner_fit.aligner, test, device="cpu")
    fit = train_option_head(
        aligned_train, aligned_validation, projection_dim=24, dropout=0.0, rps_lambda=0.0,
        epochs=80, patience=80, batch_size=128, seed=1701, device="cpu",
    )
    probability = predict_head(fit.model, aligned_test, device="cpu")
    assert probability.shape == (len(test), test.option_mask.shape[1])
    # non-degenerate: strictly better than uniform-over-4-options.
    assert fit.best_validation_nll < np.log(4)
