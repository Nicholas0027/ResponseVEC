"""Task-aligned projection of a frozen ResponseVec (design §2.5, "ResponseVec-Align").

Motivation (from the G1 audit).  The zero-shot response-centric ResponseVec
beats direct generation and a discriminative encoder ablation on held-out-item
NLL, but does *not* significantly beat the raw hidden-state mean and is edged
out by a general-purpose sentence embedder.  The diagnosis is not the encoder
choice but the *absence of task alignment*: the generative embedding preserves
output semantics in general, yet nothing has shaped its geometry toward the
actual objective — predicting which option *this* respondent selects.  This is
exactly the gap that supervised-contrastive / reconstruction fine-tuning closes
for LLM embedders in the wider literature; we close it here with a lightweight
projection while keeping the 8B encoder frozen (so the expensive extraction
cache is reused verbatim and the compute-frontier story is unchanged).

Method.  Let ``z`` be the frozen d_z-dimensional ResponseVec of a respondent on
a target item, and let ``o_c`` be the frozen option vectors from the shared
option encoder E_o.  We learn a projection ``g_phi : R^{d_z} -> R^{m}`` (a small
MLP, residual to a linear identity so it can *reduce to the raw vector* when
alignment adds nothing — the parameter-matched honesty constraint).  The
projection is trained with an **option-anchored supervised contrastive loss**:
for a batch, the aligned respondent embedding ``g_phi(z_i)`` is pulled toward the
projected encoding of the option the respondent actually chose and pushed away
from the other valid options of the same item and from the chosen-option
anchors of other respondents.  Concretely, with a shared option projection
``h_psi`` (tied to nothing in the decoder — the decoder is trained *after*, on
frozen ``g_phi``), the per-example loss is the InfoNCE

    L_i = - log  exp(<u_i, v_{i,c*}>/t)
                 / sum_{c in valid_i} exp(<u_i, v_{i,c}>/t)

where ``u_i = normalize(g_phi(z_i))``, ``v_{i,c} = normalize(h_psi(o_{i,c}))``,
``c*`` is the chosen option, and ``t`` is a learned temperature.  This is a
*within-item* contrastive objective over the true option set, so it directly
optimizes the same option-discrimination geometry the downstream option-aware
decoder relies on, without ever touching the encoder.  An optional
cross-respondent term adds other respondents' chosen-option anchors as extra
negatives (``cross_respondent_negatives``), sharpening person-level
discriminability.

Fairness / honesty properties (mirrored from decoder.py):
  * ``g_phi`` is applied identically to *any* family's z, so "ResponseVec-Align"
    can be run on raw_mean / input_centric / sentence too, and a win must come
    from the response-centric geometry, not from extra capacity handed only to
    our method.  train_primary exposes it as ``<family>_aligned`` methods.
  * Residual-to-identity init (``g(z) = z_projected + alpha * MLP(z)`` with
    ``alpha`` initialised near 0) means an untrained / useless alignment
    collapses to a plain linear projection, so the ablation "does alignment
    help?" is a clean, capacity-controlled comparison.
  * The encoder is frozen; only ``g_phi`` (and its option projection ``h_psi``)
    receive gradients — a few hundred-K parameters, ~0 extra GPU-hours beyond
    the already-paid extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # torch is an optional (llm extra) dependency; module imports on CPU dev.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except Exception:  # pragma: no cover - exercised only where torch is absent
    _TORCH = False
    nn = object  # type: ignore

NEG_INF = -1e9


@dataclass
class AlignConfig:
    """Configuration for the ResponseVec-Align projection head."""

    z_dim: int
    o_dim: int
    projection_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.10
    temperature_init: float = 0.07
    # Residual gate init: near 0 => g(z) starts as a plain linear projection of
    # z, so an untrained head reduces to a raw-vector baseline (honesty).
    residual_alpha_init: float = 0.0
    cross_respondent_negatives: bool = True
    learnable_temperature: bool = True
    max_options: int = 11


if _TORCH:

    class ResponseVecAligner(nn.Module):
        """Frozen-encoder task-alignment projection g_phi plus option projection h_psi.

        ``project`` returns the aligned respondent embedding g_phi(z) in the raw
        (un-normalized) projection space; the downstream option-aware decoder
        consumes this exactly like any other frozen z.  ``forward`` computes the
        option-anchored supervised-contrastive loss used to *train* g_phi.
        """

        def __init__(self, config: AlignConfig):
            super().__init__()
            self.config = config
            # Linear "identity" path: the projection that survives when the MLP
            # residual is gated off (residual_alpha ~ 0 at init).
            self.linear_z = nn.Linear(config.z_dim, config.projection_dim, bias=False)
            self.mlp_z = nn.Sequential(
                nn.Linear(config.z_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.projection_dim),
            )
            self.residual_alpha = nn.Parameter(torch.tensor(float(config.residual_alpha_init)))
            # Option projection h_psi into the SAME contrastive space.
            self.proj_o = nn.Linear(config.o_dim, config.projection_dim, bias=False)
            self.dropout = nn.Dropout(config.dropout)
            raw_t = float(np.log(np.expm1(max(config.temperature_init, 1e-4))))
            self.raw_temperature = nn.Parameter(
                torch.tensor(raw_t), requires_grad=config.learnable_temperature
            )

        @property
        def temperature(self):
            return F.softplus(self.raw_temperature) + 1e-4

        def project(self, z):
            """Aligned respondent embedding g_phi(z) in projection space (b, m).

            g(z) = W_lin z + alpha * MLP(z).  Not L2-normalized here so the
            downstream bilinear decoder keeps its own temperature scaling; the
            contrastive loss normalizes internally.
            """
            base = self.linear_z(self.dropout(z))
            residual = self.mlp_z(z)
            return base + self.residual_alpha * residual

        def project_options(self, option_matrix):
            """h_psi(o) for every option: (b, C, m)."""
            return self.proj_o(option_matrix)

        def forward(self, z, option_matrix, option_mask, target):
            """Option-anchored supervised-contrastive InfoNCE loss (scalar).

            z: (b, d_z); option_matrix: (b, C, d_o); option_mask: (b, C) with
            1=valid; target: (b,) index of the chosen option in [0, C).
            Positives are the chosen option's projected encoding; negatives are
            the other valid options of the same item, optionally augmented with
            other respondents' chosen-option anchors in the batch.
            """
            if z.shape[0] != option_matrix.shape[0]:
                raise ValueError("z/option batch mismatch")
            if torch.any(option_mask.sum(dim=-1) <= 0):
                raise ValueError("every alignment row must contain at least one valid option")
            u = F.normalize(self.project(z), dim=-1)                 # (b, m)
            v = F.normalize(self.project_options(option_matrix), dim=-1)  # (b, C, m)
            temperature = self.temperature

            # Within-item logits: <u_i, v_{i,c}> over the item's own options.
            within = torch.einsum("bm,bcm->bc", u, v) / temperature   # (b, C)
            within = within.masked_fill(option_mask <= 0, NEG_INF)

            logits = within
            target = target.long()
            if self.config.cross_respondent_negatives and z.shape[0] > 1:
                # Chosen-option anchor of every batch member as extra negatives.
                batch_index = torch.arange(z.shape[0], device=z.device)
                chosen = v[batch_index, target]                        # (b, m)
                cross = torch.einsum("bm,km->bk", u, chosen) / temperature  # (b, b)
                # The diagonal (a respondent's own chosen anchor) is the positive
                # already represented in `within` at column `target`; mask it out
                # of the cross block to avoid double-counting the positive.
                eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
                cross = cross.masked_fill(eye, NEG_INF)
                logits = torch.cat([within, cross], dim=-1)            # (b, C + b)

            log_prob = F.log_softmax(logits, dim=-1)
            positive_log_prob = log_prob[torch.arange(z.shape[0], device=z.device), target]
            return -positive_log_prob.mean()


def align_reduces_to_identity(config: "AlignConfig") -> bool:
    """Documentation helper: with residual_alpha_init==0 an untrained aligner's
    project() is exactly the linear path W_lin z, i.e. a plain learned linear
    projection of the raw vector — the capacity-matched null the ablation
    compares against."""
    return float(config.residual_alpha_init) == 0.0
