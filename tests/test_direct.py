from __future__ import annotations

import numpy as np
import pytest

from responsevec.direct import (
    DirectCalibrator,
    fit_prior_alpha,
    fit_temperature,
    masked_renormalize,
    nll,
    prior_blend,
    temperature_scale,
)


def _safe_log(p):
    return np.log(np.clip(p, 1e-12, None))


def test_masked_renormalize_sums_to_one_over_valid():
    p = np.array([[0.5, 0.3, 0.2]])
    mask = np.array([[1.0, 1.0, 0.0]])
    out = masked_renormalize(p, mask)
    assert out[0, 2] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_temperature_high_flattens_low_sharpens():
    p = np.array([[0.7, 0.2, 0.1]])
    mask = np.ones((1, 3))
    log_p = _safe_log(p)
    flat = temperature_scale(log_p, mask, temperature=5.0)
    sharp = temperature_scale(log_p, mask, temperature=0.2)
    # flatter -> closer to uniform; sharper -> more peaked on the max
    assert flat[0].max() < p[0].max()
    assert sharp[0].max() > p[0].max()


def test_prior_blend_endpoints():
    log_p_llm = _safe_log(np.array([[0.8, 0.1, 0.1]]))
    log_prior = _safe_log(np.array([[0.2, 0.3, 0.5]]))
    mask = np.ones((1, 3))
    at_zero = prior_blend(log_p_llm, log_prior, 0.0, mask)
    at_one = prior_blend(log_p_llm, log_prior, 1.0, mask)
    np.testing.assert_allclose(at_zero[0], [0.2, 0.3, 0.5], atol=1e-6)   # pure prior
    np.testing.assert_allclose(at_one[0], [0.8, 0.1, 0.1], atol=1e-6)    # pure LLM


def test_fit_temperature_recovers_flattening_for_overconfident():
    """Overconfident-but-correct-order predictions -> best temperature > 1."""
    rng = np.random.default_rng(0)
    n = 400
    targets = rng.integers(0, 3, n)
    probs = np.full((n, 3), 0.02)
    probs[np.arange(n), targets] = 0.96                 # extremely confident & correct
    # inject 25% wrong-but-confident to make raw NLL bad -> flattening helps
    flip = rng.random(n) < 0.25
    wrong = (targets + 1) % 3
    probs[flip] = 0.02
    probs[np.arange(n)[flip], wrong[flip]] = 0.96
    probs = probs / probs.sum(axis=1, keepdims=True)
    mask = np.ones((n, 3))
    best_t, best_nll = fit_temperature(probs, mask, targets)
    assert best_t > 1.0
    assert best_nll < nll(probs, targets)               # calibration improved NLL


def test_fit_prior_alpha_prefers_prior_when_llm_uninformative():
    rng = np.random.default_rng(1)
    n = 300
    prior_vec = np.array([0.6, 0.3, 0.1])
    targets = rng.choice(3, size=n, p=prior_vec)
    probs = np.full((n, 3), 1.0 / 3)                    # uninformative LLM
    log_prior = np.log(np.tile(prior_vec, (n, 1)))
    mask = np.ones((n, 3))
    best_alpha, _ = fit_prior_alpha(probs, log_prior, mask, targets)
    assert best_alpha < 0.5                              # leans on the prior


def test_direct_calibrator_beats_raw_on_miscalibrated():
    rng = np.random.default_rng(2)
    n = 400
    prior_vec = np.array([0.5, 0.3, 0.2])
    targets = rng.choice(3, size=n, p=prior_vec)
    # overconfident LLM that is only sometimes right
    probs = np.full((n, 3), 0.02)
    guess = rng.choice(3, size=n, p=prior_vec)
    probs[np.arange(n), guess] = 0.96
    probs = probs / probs.sum(axis=1, keepdims=True)
    log_prior = np.log(np.tile(prior_vec, (n, 1)))
    mask = np.ones((n, 3))

    cal = DirectCalibrator().fit(probs, log_prior, mask, targets, max_options=3)
    calibrated = cal.predict(probs, log_prior, mask)
    assert nll(calibrated, targets) < nll(probs, targets)


def test_neural_direct_calibrator_uses_option_aware_contract():
    torch = pytest.importorskip("torch")
    from responsevec.direct import DirectLogitCalibrator

    model = DirectLogitCalibrator(o_dim=7, projection_dim=5, max_options=4, prior_eta=0.5)
    probabilities = torch.tensor([[0.6, 0.3, 0.1, 0.0], [0.4, 0.6, 0.0, 0.0]])
    option_matrix = torch.randn(2, 4, 7)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    log_prior = torch.log(torch.tensor([[0.4, 0.4, 0.2, 1.0], [0.5, 0.5, 1.0, 1.0]]))
    output = model(probabilities, option_matrix, mask, log_prior).exp()
    np.testing.assert_allclose(output.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)
    assert output[0, 3].item() == pytest.approx(0.0)
