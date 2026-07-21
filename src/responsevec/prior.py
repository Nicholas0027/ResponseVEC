"""Population prior: the statistical backbone of the v2 prior x tilt design.

    log p_final = (1-alpha) * log p_prior + alpha * log p_LLM

The prior gives every LLM variant the same "memorize the train marginals"
ability the statistical baselines always had — information parity, so any gap
measures personalization, not access to training statistics.

Backoff ladder (per prediction):
  1. (question_key, country) counts, shrunk toward the question marginal by
     n/(n+lambda) when the country cell is small;
  2. question_key marginal (train respondents, seen items only);
  3. UNSEEN items have no per-item statistics by construction (that is the
     whole point of the OOD-item axis) -> global answer-POSITION marginal per
     scale length (e.g. "5-option items skew toward position 1-2"), which is
     the only statistic that transfers to an item nobody in train answered.

Leakage guards: fit() accepts train-split rows only and never builds per-item
tables for unseen items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .utils import write_json


class PopulationPrior:
    def __init__(self, country_shrinkage: float = 20.0, laplace: float = 0.5):
        self.country_shrinkage = float(country_shrinkage)
        self.laplace = float(laplace)
        self.question_tables: dict[str, np.ndarray] = {}
        self.country_tables: dict[tuple[str, str], np.ndarray] = {}
        self.position_tables: dict[int, np.ndarray] = {}
        self.fitted = False

    def fit(
        self,
        train_responses: pd.DataFrame,
        allowed_question_keys: set[str] | frozenset[str] | None = None,
    ) -> "PopulationPrior":
        """Fit using training respondents and, when supplied, only the item
        keys assigned to the current fold's training role.

        ``allowed_question_keys`` is mandatory for Protocol B callers. Without
        it, item-specific tables for validation/test items would reveal the
        held-out response distribution even though respondent IDs are disjoint.
        The scale-position table is also learned from the allowed training
        items only.
        """
        splits = set(train_responses["split"].unique()) if "split" in train_responses.columns else set()
        if splits - {"train"}:
            raise ValueError(f"PopulationPrior must be fit on split=='train' only, got {splits}")
        seen = train_responses[~train_responses["is_unseen_item"]]
        if allowed_question_keys is not None:
            # Protocol B replaces the legacy is_unseen_item axis entirely.
            seen = train_responses
            allowed = {str(key) for key in allowed_question_keys}
            seen = seen[seen["question_key"].astype(str).isin(allowed)]
        if seen.empty:
            raise ValueError("PopulationPrior.fit received no eligible training rows")

        # Re-fitting the same instance for another fold must not retain tables
        # from the previous fold.
        self.question_tables.clear()
        self.country_tables.clear()
        self.position_tables.clear()

        for question_key, group in seen.groupby("question_key"):
            n_options = int(group["n_options"].max())
            counts = np.bincount(group["answer_index"].astype(int), minlength=n_options).astype(np.float64)
            self.question_tables[question_key] = self._normalize(counts)
            for country, sub in group.groupby("country"):
                sub_counts = np.bincount(sub["answer_index"].astype(int), minlength=n_options).astype(np.float64)
                weight = len(sub) / (len(sub) + self.country_shrinkage)
                blended = weight * self._normalize(sub_counts) + (1 - weight) * self.question_tables[question_key]
                self.country_tables[(question_key, str(country))] = blended / blended.sum()

        # Global position marginal per scale length, from SEEN items only —
        # the only statistic legitimately available for unseen items.
        for n_options, group in seen.groupby("n_options"):
            counts = np.bincount(group["answer_index"].astype(int), minlength=int(n_options)).astype(np.float64)
            self.position_tables[int(n_options)] = self._normalize(counts)
        self.fitted = True
        return self

    def _normalize(self, counts: np.ndarray) -> np.ndarray:
        smoothed = counts + self.laplace
        return smoothed / smoothed.sum()

    def predict(self, question_key: str, country: str, n_options: int, item_is_seen: bool) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PopulationPrior.predict called before fit")
        if item_is_seen:
            table = self.country_tables.get((question_key, str(country)))
            if table is None:
                table = self.question_tables.get(question_key)
            if table is not None:
                return table[:n_options] / table[:n_options].sum()
        # Unseen item (or an item somehow absent from train): position marginal.
        table = self.position_tables.get(int(n_options))
        if table is None:
            return np.full(n_options, 1.0 / n_options)
        return table[:n_options] / table[:n_options].sum()

    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "country_shrinkage": self.country_shrinkage,
            "laplace": self.laplace,
            "question_tables": {k: v.tolist() for k, v in self.question_tables.items()},
            "country_tables": {f"{k[0]}|||{k[1]}": v.tolist() for k, v in self.country_tables.items()},
            "position_tables": {str(k): v.tolist() for k, v in self.position_tables.items()},
        }
        write_json(directory / "population_prior.json", payload)

    @classmethod
    def load(cls, directory: str | Path) -> "PopulationPrior":
        payload = json.loads((Path(directory) / "population_prior.json").read_text())
        prior = cls(payload["country_shrinkage"], payload["laplace"])
        prior.question_tables = {k: np.asarray(v) for k, v in payload["question_tables"].items()}
        prior.country_tables = {
            tuple(k.split("|||", 1)): np.asarray(v) for k, v in payload["country_tables"].items()
        }
        prior.position_tables = {int(k): np.asarray(v) for k, v in payload["position_tables"].items()}
        prior.fitted = True
        return prior
