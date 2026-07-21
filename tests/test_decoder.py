from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from responsevec.encode import build_option_table, load_option_table, save_option_table, stack_option_matrix

torch = pytest.importorskip("torch")

from responsevec.decoder import (  # noqa: E402
    DecoderConfig,
    ObjectiveWeights,
    OptionAwareDecoder,
    masked_log_softmax,
    rps_loss,
    total_loss,
)


class _FakeEncoder:
    """Deterministic encoder: hash text -> fixed 6-dim vector."""

    def encode(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(rng.normal(size=6))
        return np.asarray(out)


# --- option encoder ---------------------------------------------------------


def test_build_and_stack_option_table(tmp_path):
    catalogue = pd.DataFrame({
        "question_key": ["D::q1", "D::q2"],
        "question": ["Concern?", "Agree?"],
        "options_json": [json.dumps(["Low", "Med", "High"]), json.dumps(["No", "Yes"])],
    })
    table = build_option_table(catalogue, _FakeEncoder())
    assert table["D::q1"].shape == (3, 6)
    assert table["D::q2"].shape == (2, 6)

    save_option_table(table, tmp_path / "opts.npz")
    reloaded = load_option_table(tmp_path / "opts.npz")
    np.testing.assert_allclose(reloaded["D::q1"], table["D::q1"])

    matrix, mask = stack_option_matrix(table, ["D::q1", "D::q2"], max_options=3)
    assert matrix.shape == (2, 3, 6)
    assert mask[0].tolist() == [1, 1, 1]
    assert mask[1].tolist() == [1, 1, 0]        # q2 has 2 options, third slot invalid
    np.testing.assert_allclose(matrix[1, 2], 0.0)


# --- decoder mechanics ------------------------------------------------------


def test_masked_log_softmax_zeros_invalid_slots():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    probs = masked_log_softmax(scores, mask).exp()
    assert probs[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_decoder_forward_shapes_and_normalization():
    cfg = DecoderConfig(z_dim=8, o_dim=6, projection_dim=5, max_options=4)
    dec = OptionAwareDecoder(cfg)
    b, C = 3, 4
    z = torch.randn(b, 8)
    opts = torch.randn(b, C, 6)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    log_prior = torch.log(torch.tensor([[.25, .25, .25, .25]] * b))
    log_probs = dec(z, opts, mask, log_prior)
    probs = log_probs.exp()
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)
    assert probs[1, 2].item() == pytest.approx(0.0, abs=1e-6)  # masked


def test_rps_loss_zero_for_perfect_confident_prediction():
    # perfectly confident on the true (ordinal) class -> RPS 0
    log_probs = torch.log(torch.tensor([[1e-9, 1e-9, 1.0]]))
    log_probs = log_probs - torch.logsumexp(log_probs, -1, keepdim=True)
    mask = torch.ones(1, 3)
    target = torch.tensor([2])
    assert rps_loss(log_probs, target, mask).item() == pytest.approx(0.0, abs=1e-4)


def test_rps_penalizes_far_errors_more_than_near():
    mask = torch.ones(1, 3)
    target = torch.tensor([0])
    near = torch.log(torch.tensor([[1e-6, 1.0, 1e-6]]))    # predicts class 1 (adjacent)
    far = torch.log(torch.tensor([[1e-6, 1e-6, 1.0]]))     # predicts class 2 (distant)
    near = near - torch.logsumexp(near, -1, keepdim=True)
    far = far - torch.logsumexp(far, -1, keepdim=True)
    assert rps_loss(far, target, mask).item() > rps_loss(near, target, mask).item()


# --- learning behaviour -----------------------------------------------------


def _train(decoder, batch, weights, steps=300, lr=1e-2):
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss, _, _ = total_loss(decoder, batch, weights)
        loss.backward()
        opt.step()
    with torch.no_grad():
        lp, _, _ = total_loss(decoder, batch, weights)
    return total_loss(decoder, batch, weights)[1].item()


def test_signal_recovery_beats_noise():
    """A z that encodes the answer should train to far lower NLL than pure-noise
    z on the same items — the decoder can exploit real representation signal."""
    torch.manual_seed(0)
    n, C, d_o = 128, 3, 6
    targets = torch.randint(0, C, (n,))
    option_matrix = torch.randn(n, C, d_o)
    mask = torch.ones(n, C)
    log_prior = torch.log(torch.full((n, C), 1.0 / C))

    # informative z: the option vector of the true answer plus small noise
    z_signal = torch.stack([option_matrix[i, targets[i]] for i in range(n)]) + 0.1 * torch.randn(n, d_o)
    z_noise = torch.randn(n, d_o)

    def make_batch(z):
        return {"z": z, "option_matrix": option_matrix, "option_mask": mask,
                "log_prior": log_prior, "target": targets}

    nll_signal = _train(OptionAwareDecoder(DecoderConfig(z_dim=d_o, o_dim=d_o, projection_dim=6, dropout=0.0, max_options=C)),
                        make_batch(z_signal), ObjectiveWeights())
    nll_noise = _train(OptionAwareDecoder(DecoderConfig(z_dim=d_o, o_dim=d_o, projection_dim=6, dropout=0.0, max_options=C)),
                       make_batch(z_noise), ObjectiveWeights())
    assert nll_signal < nll_noise - 0.2
    assert nll_signal < float(np.log(C))          # beats uniform


def test_reduces_to_prior_when_z_uninformative():
    """With z=0 (no information) and a skewed prior, the eta*log-prior term
    should let the head recover the prior far better than uniform."""
    torch.manual_seed(0)
    n, C, d_o = 96, 3, 6
    prior = torch.tensor([0.6, 0.3, 0.1])
    targets = torch.multinomial(prior, n, replacement=True)
    batch = {
        "z": torch.zeros(n, d_o),
        "option_matrix": torch.randn(n, C, d_o),
        "option_mask": torch.ones(n, C),
        "log_prior": torch.log(prior).unsqueeze(0).repeat(n, 1),
        "target": targets,
    }
    dec = OptionAwareDecoder(DecoderConfig(
        z_dim=d_o, o_dim=d_o, projection_dim=6, dropout=0.0,
        max_options=C, prior_eta_init=1.0, learnable_eta=False,
    ))
    nll = _train(dec, batch, ObjectiveWeights())
    assert nll < float(np.log(C))                  # beats uniform by leaning on the prior
    assert dec.eta.item() == pytest.approx(1.0, rel=1e-3)  # same fixed prior weight used by every family
