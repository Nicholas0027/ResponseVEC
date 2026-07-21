"""Item-conditional collaborative-filtering baseline — the honest statistical
frontier for seen items.

For target item j and each known answer (q_k, y_k), the train-split empirical
conditional P(y_j | y_k on q_k) (additively shrunk toward item j's marginal)
provides a likelihood-ratio tilt on the prior, pooled log-linearly with
weights proportional to the shrunk |Spearman| correlation |C_jk|:

    log p(y_j | H) = log prior_j(y_j) + sum_k w_jk * [log cond_jk(y_j|y_k) - log prior_j(y_j)]

K=0 reduces exactly to the prior. Like MIRT, this method has no parameters for
UNSEEN items and refuses to predict them — that structural absence is the
point of the OOD-item axis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data import PanelStore, demographic_record_fields
from ..item_graph import ItemGraph
from ..prior import PopulationPrior


class ItemConditionalModel:
    def __init__(
        self,
        prior: PopulationPrior,
        item_graph: ItemGraph,
        min_pair_count: int = 20,
        conditional_shrinkage: float = 5.0,
        weight_scale: float = 1.0,
    ):
        self.prior = prior
        self.item_graph = item_graph
        self.min_pair_count = int(min_pair_count)
        self.conditional_shrinkage = float(conditional_shrinkage)
        self.weight_scale = float(weight_scale)
        # (target_key, history_key) -> (n_hist_options, n_target_options) row-normalized table
        self.conditionals: dict[tuple[str, str], np.ndarray] = {}
        self.fitted = False

    def fit(
        self,
        train_responses: pd.DataFrame,
        allowed_question_keys: set[str] | frozenset[str] | None = None,
    ) -> "ItemConditionalModel":
        splits = set(train_responses["split"].unique()) if "split" in train_responses.columns else set()
        if splits - {"train"}:
            raise ValueError(f"ItemConditionalModel must be fit on split=='train' only, got {splits}")
        seen = train_responses[~train_responses["is_unseen_item"]]
        if allowed_question_keys is not None:
            seen = train_responses
            seen = seen[seen["question_key"].astype(str).isin({str(k) for k in allowed_question_keys})]
        self.conditionals.clear()

        meta = seen.drop_duplicates("question_key").set_index("question_key")[["n_options", "country"]]
        wide = seen.pivot_table(index="panel_id", columns="question_key", values="answer_index", aggfunc="first")
        items = list(wide.columns)
        for domain, domain_group in seen.groupby("domain"):
            domain_items = [q for q in items if q in set(domain_group["question_key"])]
            for target in domain_items:
                n_target = int(meta.at[target, "n_options"])
                target_col = wide[target]
                for history in domain_items:
                    if history == target:
                        continue
                    pair = pd.DataFrame({"h": wide[history], "t": target_col}).dropna()
                    if len(pair) < self.min_pair_count:
                        continue
                    n_hist = int(meta.at[history, "n_options"])
                    table = np.zeros((n_hist, n_target), dtype=np.float64)
                    for h_val, t_val in zip(pair["h"].astype(int), pair["t"].astype(int)):
                        table[h_val, t_val] += 1.0
                    # Shrink each row toward the target's overall marginal.
                    marginal = self._target_marginal(target_col, n_target)
                    for h_val in range(n_hist):
                        row_count = table[h_val].sum()
                        smoothed = table[h_val] + self.conditional_shrinkage * marginal
                        table[h_val] = smoothed / smoothed.sum() if smoothed.sum() > 0 else marginal
                        del row_count
                    self.conditionals[(target, history)] = table
        self.fitted = True
        return self

    @staticmethod
    def _target_marginal(target_col: pd.Series, n_target: int) -> np.ndarray:
        counts = np.bincount(target_col.dropna().astype(int), minlength=n_target).astype(np.float64) + 0.5
        return counts / counts.sum()

    def predict_one(
        self,
        target_key: str,
        domain: str,
        country: str,
        n_options: int,
        history: Iterable[tuple[str, int]],
    ) -> np.ndarray:
        """history: [(question_key, RAW answer_index), ...]."""
        if not self.fitted:
            raise RuntimeError("predict_one called before fit")
        prior = self.prior.predict(target_key, country, n_options, item_is_seen=True)
        log_p = np.log(np.clip(prior, 1e-12, None))
        for history_key, answer_index in history:
            table = self.conditionals.get((target_key, history_key))
            if table is None or not 0 <= int(answer_index) < table.shape[0]:
                continue
            weight = self.weight_scale * abs(self.item_graph.get(domain, target_key, history_key))
            if weight <= 0:
                continue
            conditional = np.clip(table[int(answer_index), :n_options], 1e-12, None)
            log_p += weight * (np.log(conditional) - np.log(np.clip(prior, 1e-12, None)))
        log_p -= log_p.max()
        p = np.exp(log_p)
        return p / p.sum()


def predict_item_conditional(
    store: PanelStore,
    model: ItemConditionalModel,
    k_values: Iterable[int],
    calibration_seed: int,
    output_path: str | Path,
    split: str = "test",
    item_pool: str = "seen",
) -> pd.DataFrame:
    if item_pool == "unseen":
        raise ValueError(
            "ItemConditionalModel has no conditional tables for unseen items — report N/A on the OOD-item "
            "axis rather than calling predict_item_conditional(item_pool='unseen')."
        )
    records: list[dict[str, Any]] = []
    for k in k_values:
        targets = store.target_rows(split, int(k), calibration_seed, item_pool=item_pool)
        for row in targets.itertuples(index=False):
            history_rows = store.history_rows(row.panel_id, row.question_id, int(k), calibration_seed)
            history = list(zip(history_rows["question_key"].astype(str), history_rows["answer_index"].astype(int)))
            probabilities = model.predict_one(row.question_key, row.domain, row.country, int(row.n_options), history)
            target = int(row.answer_index)
            predicted = int(probabilities.argmax())
            records.append(
                {
                    "method": "item_conditional", "row_id": row.row_id, "panel_id": row.panel_id, "domain": row.domain,
                    "question_id": row.question_id, "question_key": row.question_key, "k": int(k), "split": split,
                    "item_pool": item_pool, "calibration_seed": int(calibration_seed), "option_seed": 0,
                    "answer_index": target, "n_options": int(row.n_options), "survey_weight": float(row.survey_weight),
                    "probabilities_json": json.dumps(probabilities.tolist()), "predicted_index": predicted,
                    "nll": float(-np.log(max(probabilities[target], 1e-12))),
                    "brier": float(np.square(probabilities - np.eye(int(row.n_options))[target]).sum()),
                    "normalized_ordinal_error": float(abs(predicted - target) / max(1, int(row.n_options) - 1)),
                    "correct": int(predicted == target),
                    **demographic_record_fields(row),
                }
            )
    output = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output
