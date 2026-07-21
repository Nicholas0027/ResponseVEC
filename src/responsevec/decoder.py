"""Option-aware bilinear decoder and its training objectives (design §2.4, §2.6).

Given a frozen respondent representation z (dim d_z, from any family) and the
frozen option vectors o_c (dim d_o, from the shared encoder E_o), the decoder
scores option c as

    s_c = (P_z z)^T (P_o o_c) / tau + b_pos[c] + eta * log prior_c

and applies a masked softmax over the options present for that item. Only
P_z, P_o, tau, b_pos, and eta are trainable — a few hundred-K parameters,
identical in count across every representation family, so a win reflects
representation quality rather than head capacity (the parameter-matched
argument, design §2.2/§6.4 baseline 6).

Design properties enforced/tested here:
  * variable options per item via the validity mask;
  * K=0 / uninformative z -> the eta*log-prior term dominates, so the head
    reduces to the population prior (test_reduces_to_prior);
  * eta is train-only-prior-blend (shared fairly across methods, design §2.6);
  * objectives: cross-entropy NLL (primary) + ordinal RPS auxiliary
    (rps_lambda) + optional group-JS calibration (group_lambda, secondary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:  # torch is an optional (llm extra) dependency; the module imports on CPU dev.
    import torch
    import torch.nn as nn

    _TORCH = True
except Exception:  # pragma: no cover - exercised only where torch is absent
    _TORCH = False
    nn = object  # type: ignore

NEG_INF = -1e9


@dataclass
class DecoderConfig:
    z_dim: int
    o_dim: int
    projection_dim: int = 256
    dropout: float = 0.10
    temperature_init: float = 0.1
    prior_eta_init: float = 0.5
    # Primary comparisons fix the validation-selected eta identically across
    # representation families. Learning a separate eta per family would
    # confound representation quality with unequal access to the prior.
    learnable_eta: bool = False
    max_options: int = 11


def masked_log_softmax(scores, mask):
    """Log-softmax over valid options only (mask 1=valid). Invalid slots get
    -inf before the softmax and 0 probability after."""
    if scores.shape != mask.shape:
        raise ValueError(f"scores/mask shape mismatch: {scores.shape} vs {mask.shape}")
    if torch.any(mask.sum(dim=-1) <= 0):
        raise ValueError("every decoder row must contain at least one valid option")
    scores = scores.masked_fill(mask <= 0, float("-inf"))
    return torch.log_softmax(scores, dim=-1)


if _TORCH:

    class OptionAwareDecoder(nn.Module):
        def __init__(self, config: DecoderConfig):
            super().__init__()
            self.config = config
            self.proj_z = nn.Linear(config.z_dim, config.projection_dim, bias=False)
            self.proj_o = nn.Linear(config.o_dim, config.projection_dim, bias=False)
            self.dropout = nn.Dropout(config.dropout)
            # Scale-length-aware positional bias b_pos(c, C_q), not one global
            # vector shared across two-, five-, and eleven-option questions.
            self.position_bias = nn.Parameter(
                torch.zeros(config.max_options + 1, config.max_options)
            )
            # tau and eta kept positive via softplus of a raw parameter.
            self.raw_tau = nn.Parameter(torch.tensor(float(np.log(np.expm1(config.temperature_init)))))
            eta0 = torch.tensor(float(np.log(np.expm1(max(config.prior_eta_init, 1e-3)))))
            self.raw_eta = nn.Parameter(eta0, requires_grad=config.learnable_eta)

        @property
        def temperature(self):
            return torch.nn.functional.softplus(self.raw_tau) + 1e-4

        @property
        def eta(self):
            return torch.nn.functional.softplus(self.raw_eta)

        def forward(self, z, option_matrix, option_mask, log_prior):
            """z: (b, d_z); option_matrix: (b, C, d_o); option_mask: (b, C);
            log_prior: (b, C). Returns log-probabilities (b, C)."""
            zc = self.dropout(self.proj_z(z))                    # (b, p)
            oc = self.proj_o(option_matrix)                      # (b, C, p)
            bilinear = torch.einsum("bp,bcp->bc", zc, oc)        # (b, C)
            scores = bilinear / self.temperature
            counts = option_mask.sum(dim=-1).long()
            if torch.any(counts > self.config.max_options):
                raise ValueError("option count exceeds DecoderConfig.max_options")
            row_bias = self.position_bias[counts, : scores.shape[1]]
            scores = scores + row_bias
            scores = scores + self.eta * log_prior
            return masked_log_softmax(scores, option_mask)

    def rps_loss(log_probs, targets, mask, ordinal_mask=None):
        """Ranked probability score on the ORDINAL option axis (design §2.6):
        mean over items of sum_c (CDF_pred_c - CDF_true_c)^2, normalized by
        (n_options - 1). Options are already in semantic (ordinal) order."""
        probs = log_probs.exp() * mask
        cdf_pred = torch.cumsum(probs, dim=-1)
        true_onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        cdf_true = torch.cumsum(true_onehot, dim=-1)
        n_options = mask.sum(dim=-1).clamp(min=2.0)
        # The final CDF is always one and is not part of the RPS definition.
        cutoff_mask = mask.clone()
        last_valid = (n_options.long() - 1).unsqueeze(1)
        cutoff_mask.scatter_(1, last_valid, 0.0)
        per_item = ((cdf_pred - cdf_true) ** 2 * cutoff_mask).sum(dim=-1) / (n_options - 1.0)
        if ordinal_mask is not None:
            selected = per_item[ordinal_mask.bool()]
            return selected.mean() if len(selected) else per_item.new_zeros(())
        return per_item.mean()

    def group_js_loss(log_probs, mask, group_ids, target_dist):
        """Optional calibration term (design §2.6, secondary): Jensen-Shannon
        divergence between each group's mean predicted distribution and the
        group's empirical target distribution. Enabled only AFTER the
        representation-only comparison so it can't confound the main claim."""
        probs = log_probs.exp() * mask
        total = torch.zeros((), dtype=probs.dtype, device=probs.device)
        unique = torch.unique(group_ids)
        for g in unique:
            sel = group_ids == g
            group_masks = mask[sel]
            if not torch.all(group_masks == group_masks[0]):
                raise ValueError(
                    "group_js_loss groups must include question identity so every row in a group shares option semantics"
                )
            mean_pred = probs[sel].mean(dim=0)
            empirical = (target_dist[sel] * group_masks).mean(dim=0)
            empirical = empirical / empirical.sum().clamp_min(1e-12)
            m = 0.5 * (mean_pred + empirical) + 1e-12
            js = 0.5 * (mean_pred * (mean_pred.clamp_min(1e-12).log() - m.log())).sum() + \
                 0.5 * (empirical * (empirical.clamp_min(1e-12).log() - m.log())).sum()
            total = total + js
        return total / max(len(unique), 1)


@dataclass
class ObjectiveWeights:
    rps_lambda: float = 0.0
    group_lambda: float = 0.0


def total_loss(decoder, batch, weights: "ObjectiveWeights"):
    """Assemble NLL + rps_lambda*RPS + group_lambda*JS for one batch dict with
    keys z, option_matrix, option_mask, log_prior, target (and optionally
    group_ids, target_dist)."""
    log_probs = decoder(batch["z"], batch["option_matrix"], batch["option_mask"], batch["log_prior"])
    nll = torch.nn.functional.nll_loss(log_probs, batch["target"])
    loss = nll
    if weights.rps_lambda > 0:
        loss = loss + weights.rps_lambda * rps_loss(
            log_probs, batch["target"], batch["option_mask"], batch.get("ordinal_mask")
        )
    if weights.group_lambda > 0 and "group_ids" in batch:
        loss = loss + weights.group_lambda * group_js_loss(
            log_probs, batch["option_mask"], batch["group_ids"], batch["target_dist"]
        )
    return loss, nll.detach(), log_probs.detach()
