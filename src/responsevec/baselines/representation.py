"""Representation baselines that feed the SAME option-aware decoder as
ResponseVec, differing only in how the respondent vector z is built (design
§6.4 baselines 7-9, 15). Because the decoder and the frozen option encoder are
held fixed, these isolate "does the LLM's internal state beat a cheaper z?".

  * SentenceEncoderZ  — z is a plain text embedding (BGE) of the same canonical
                      prompt. If this matches ResponseVec, no LLM internals are
                      needed. (baseline 7)
  * PPlugHistoryZ     — z is a similarity-weighted average of the sentence
                      embeddings of the respondent's history answers, in the
                      PPlug personalization style. Tests whether a shallow
                      aggregate of past answers already captures the signal.
                      (baseline 8)
  * DemographicOnlyZ  — z is a one-hot demographic vector (no text model at all)
                      projected by the decoder; the floor for "representation"
                      claims. (baseline 9)

None of these read target labels or target-item statistics. History answers
used by PPlugHistoryZ are the leakage-safe R_q output from prompting_rv.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def sentence_encoder_z(prompt_embeddings: np.ndarray) -> np.ndarray:
    """Identity pass-through: z is the (already cached) sentence embedding of the
    canonical prompt. Kept as a named function so the pipeline treats every
    z-builder uniformly."""
    return np.asarray(prompt_embeddings, dtype=np.float32)


def pplug_history_z(
    target_embedding: np.ndarray,
    history_answer_embeddings: Sequence[np.ndarray],
    temperature: float = 0.5,
) -> np.ndarray:
    """PPlug-style z for one respondent-target: similarity-weighted average of
    the respondent's history-answer embeddings, weights = softmax(cos(target,
    answer)/temperature). With no history, returns a zero vector (the decoder's
    prior term then carries the prediction)."""
    if len(history_answer_embeddings) == 0:
        return np.zeros_like(target_embedding, dtype=np.float32)
    answers = np.asarray(history_answer_embeddings, dtype=np.float64)
    target = np.asarray(target_embedding, dtype=np.float64)
    target_norm = target / max(np.linalg.norm(target), 1e-12)
    answer_norms = answers / np.clip(np.linalg.norm(answers, axis=1, keepdims=True), 1e-12, None)
    sims = answer_norms @ target_norm
    weights = np.exp((sims - sims.max()) / max(temperature, 1e-6))
    weights = weights / weights.sum()
    return (weights[:, None] * answers).sum(axis=0).astype(np.float32)


def build_pplug_matrix(
    target_embeddings: np.ndarray,
    history_answer_embeddings: Sequence[Sequence[np.ndarray]],
    temperature: float = 0.5,
) -> np.ndarray:
    """Vectorized pplug_history_z over a batch of rows."""
    return np.asarray(
        [
            pplug_history_z(target_embeddings[i], history_answer_embeddings[i], temperature)
            for i in range(len(target_embeddings))
        ],
        dtype=np.float32,
    )
