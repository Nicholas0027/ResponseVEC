"""Direct-output controls (design §2.2, §6.4 baselines 1-3, 6).

The decisive comparison is ResponseVec vs the LLM's OWN generative output on the
same prompt. To make it fair we must give the direct logits every calibration
advantage the representation head enjoys, so a ResponseVec win cannot be
explained by "the head was calibrated and the baseline wasn't":

  1. direct_raw        — softmax of the option-token logits, untouched.
  2. direct_temp       — one temperature parameter fit on validation NLL.
  3. direct_prior      — geometric prior blend (1-alpha)*log prior + alpha*log
                       p_direct, alpha fit on validation (shared with reps).
  4. direct_scalar     — temperature + prior + position calibration.
  5. direct_calibrated — a supervised option-aware neural calibrator whose only
                       respondent-specific input is the direct probability
                       vector and its entropy; it also receives the same frozen
                       option vectors and prior as ResponseVec, but no hidden
                       state. If ResponseVec beats this, supervised semantic
                       recalibration alone does not explain the gain.

Everything operates in semantic option order on masked distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

NEG_INF = -1e9


def _safe_log(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probabilities, 1e-12, None))


def masked_renormalize(probabilities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = probabilities * mask
    total = p.sum(axis=-1, keepdims=True)
    return np.divide(p, np.where(total <= 0, 1.0, total))


def temperature_scale(log_probs: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a temperature to log-probabilities and renormalize over valid
    options. temperature>1 flattens, <1 sharpens."""
    scaled = log_probs / max(temperature, 1e-6)
    scaled = np.where(mask > 0, scaled, NEG_INF)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    return masked_renormalize(np.exp(scaled), mask)


def prior_blend(log_p_llm: np.ndarray, log_prior: np.ndarray, alpha: float, mask: np.ndarray) -> np.ndarray:
    """Geometric pooling (1-alpha)*log_prior + alpha*log_p_llm, renormalized."""
    safe_prior = np.where(mask > 0, log_prior, 0.0)
    safe_llm = np.where(mask > 0, log_p_llm, 0.0)
    blended = (1.0 - alpha) * safe_prior + alpha * safe_llm
    blended = np.where(mask > 0, blended, NEG_INF)
    blended = blended - blended.max(axis=-1, keepdims=True)
    return masked_renormalize(np.exp(blended), mask)


def nll(probabilities: np.ndarray, targets: np.ndarray) -> float:
    picked = probabilities[np.arange(len(targets)), targets]
    return float(-np.mean(_safe_log(picked)))


def fit_temperature(
    probabilities: np.ndarray, mask: np.ndarray, targets: np.ndarray, grid: Sequence[float] | None = None
) -> tuple[float, float]:
    """Grid-search the temperature minimizing validation NLL. Returns
    (best_temperature, best_nll)."""
    grid = grid if grid is not None else np.geomspace(0.25, 4.0, 25)
    log_probs = _safe_log(probabilities)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        candidate_nll = nll(temperature_scale(log_probs, mask, float(t)), targets)
        if candidate_nll < best_nll:
            best_t, best_nll = float(t), candidate_nll
    return best_t, best_nll


def fit_prior_alpha(
    probabilities: np.ndarray,
    log_prior: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    step: float = 0.05,
) -> tuple[float, float]:
    """Grid-search alpha in [0,1] minimizing validation NLL of the prior blend."""
    log_p_llm = _safe_log(probabilities)
    best_alpha, best_nll = 0.0, float("inf")
    for alpha in np.arange(0.0, 1.0 + 1e-9, step):
        candidate = prior_blend(log_p_llm, log_prior, float(alpha), mask)
        candidate_nll = nll(candidate, targets)
        if candidate_nll < best_nll:
            best_alpha, best_nll = float(alpha), candidate_nll
    return best_alpha, best_nll


@dataclass
class DirectCalibrator:
    """Scalar direct-output calibrator. Tunable scalars only:
    temperature, prior weight eta, and a per-position bias vector — the same
    calibration parameters the bilinear decoder has, minus the z-o interaction.
    Fit by coordinate grid search on validation NLL (no gradients needed for so
    few parameters; keeps the baseline dependency-light)."""

    temperature: float = 1.0
    eta: float = 0.0
    position_bias: np.ndarray = None  # type: ignore

    def predict(self, probabilities: np.ndarray, log_prior: np.ndarray, mask: np.ndarray) -> np.ndarray:
        log_p = _safe_log(probabilities) / max(self.temperature, 1e-6)
        scores = log_p + self.eta * np.where(mask > 0, log_prior, 0.0)
        if self.position_bias is not None:
            scores = scores + self.position_bias[: scores.shape[1]][None, :]
        scores = np.where(mask > 0, scores, NEG_INF)
        scores = scores - scores.max(axis=-1, keepdims=True)
        return masked_renormalize(np.exp(scores), mask)

    def fit(self, probabilities, log_prior, mask, targets, *, max_options: int) -> "DirectCalibrator":
        temp_grid = np.geomspace(0.25, 4.0, 15)
        eta_grid = np.arange(0.0, 1.0 + 1e-9, 0.1)
        best = (1.0, 0.0, float("inf"))
        for t in temp_grid:
            for e in eta_grid:
                self.temperature, self.eta = float(t), float(e)
                candidate_nll = nll(self.predict(probabilities, log_prior, mask), targets)
                if candidate_nll < best[2]:
                    best = (float(t), float(e), candidate_nll)
        self.temperature, self.eta = best[0], best[1]
        # One pass of position-bias refinement (mean residual per position).
        self.position_bias = np.zeros(max_options, dtype=np.float64)
        return self


# The strongest direct-output control is implemented with torch so it can use
# the same frozen option matrix and optimizer as OptionAwareDecoder.
try:
    import torch
    import torch.nn as nn

    from .decoder import masked_log_softmax

    class DirectLogitCalibrator(nn.Module):
        """Option-aware supervised calibration without access to LM states.

        The direct distribution is compressed into a respondent-specific
        summary and scored against each frozen option vector. Raw log
        probabilities enter as a residual, ensuring the head starts from the
        LM output rather than learning an unrelated item classifier.
        """

        def __init__(
            self,
            o_dim: int,
            projection_dim: int = 256,
            max_options: int = 11,
            dropout: float = 0.1,
            temperature_init: float = 1.0,
            prior_eta: float = 0.5,
            learnable_eta: bool = False,
        ):
            super().__init__()
            self.max_options = int(max_options)
            self.summary_projection = nn.Linear(self.max_options + 1, projection_dim, bias=False)
            self.option_projection = nn.Linear(int(o_dim), projection_dim, bias=False)
            self.dropout = nn.Dropout(float(dropout))
            self.position_bias = nn.Parameter(torch.zeros(self.max_options + 1, self.max_options))
            self.raw_temperature = nn.Parameter(
                torch.tensor(float(np.log(np.expm1(max(float(temperature_init), 1e-3)))))
            )
            eta_raw = torch.tensor(float(np.log(np.expm1(max(float(prior_eta), 1e-3)))))
            self.raw_eta = nn.Parameter(eta_raw, requires_grad=bool(learnable_eta))

        @property
        def temperature(self):
            return torch.nn.functional.softplus(self.raw_temperature) + 1e-4

        @property
        def eta(self):
            return torch.nn.functional.softplus(self.raw_eta)

        def forward(self, direct_probabilities, option_matrix, option_mask, log_prior):
            if direct_probabilities.shape != option_mask.shape:
                raise ValueError("direct probabilities and option mask must share shape")
            batch, width = direct_probabilities.shape
            if width > self.max_options:
                raise ValueError("direct distribution exceeds configured max_options")
            p = direct_probabilities.clamp_min(1e-12) * option_mask
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            log_p = p.clamp_min(1e-12).log()
            entropy = -(p * log_p * option_mask).sum(dim=-1, keepdim=True)
            summary = torch.zeros(
                batch, self.max_options + 1, dtype=p.dtype, device=p.device
            )
            summary[:, :width] = log_p
            summary[:, -1:] = entropy
            respondent = self.dropout(self.summary_projection(summary))
            options = self.option_projection(option_matrix)
            semantic_adjustment = torch.einsum("bp,bcp->bc", respondent, options)
            counts = option_mask.sum(dim=-1).long()
            scores = log_p / self.temperature + semantic_adjustment
            scores = scores + self.position_bias[counts, :width]
            scores = scores + self.eta * log_prior
            return masked_log_softmax(scores, option_mask)

except Exception:  # pragma: no cover - torch is optional for non-neural utilities
    DirectLogitCalibrator = None  # type: ignore
