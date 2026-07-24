"""Strong individual-level discriminative baselines.

The sklearn baselines convert each respondent-question row into candidate-option
rows and learn whether each option is the observed choice.  Candidate scores
are normalized within the item, so variable option counts are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


@dataclass
class SklearnChoiceBaseline:
    model_name: str = "random_forest"
    seed: int = 1701
    max_train_rows: int | None = 250_000

    def _demographics(self, arrays, fit=False):
        frame = arrays.rows[list(self.demographic_columns)].fillna("Missing").astype(str)
        if fit:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            return self.encoder.fit_transform(frame)
        return self.encoder.transform(frame).astype(np.float32)

    def _history_mean(self, arrays):
        mask = arrays.history_mask[..., None]
        return (arrays.history * mask).sum(1) / np.clip(mask.sum(1), 1.0, None)

    def _candidate_features(self, arrays, fit=False, include_labels=True):
        demographics = self._demographics(arrays, fit=fit)
        history = self._history_mean(arrays)
        features, labels, layout = [], [], []
        for row in range(len(arrays)):
            n = int(arrays.option_mask[row].sum())
            for option in range(n):
                option_vector = arrays.option_matrix[row, option]
                features.append(np.concatenate([
                    demographics[row], history[row], option_vector,
                    np.asarray([
                        arrays.log_prior[row, option], option / max(n - 1, 1), n / arrays.option_mask.shape[1],
                        float(np.dot(history[row], option_vector)),
                        float(np.linalg.norm(history[row] - option_vector)),
                    ], dtype=np.float32),
                ]))
                if include_labels:
                    labels.append(int(option == arrays.targets[row]))
                layout.append((row, option))
        return np.asarray(features, np.float32), np.asarray(labels, np.int64), layout

    def fit(self, arrays, demographic_columns=("country", "sex", "age_bin", "education", "income_quintile")):
        self.demographic_columns = tuple(
            column for column in demographic_columns if column in arrays.rows.columns
        )
        if self.max_train_rows and sum(arrays.option_mask.sum(axis=1)) > self.max_train_rows:
            rng = np.random.default_rng(self.seed)
            mean_options = max(float(arrays.option_mask.sum(axis=1).mean()), 1.0)
            n_rows = min(len(arrays), max(1, int(self.max_train_rows / mean_options)))
            chosen_rows = np.sort(rng.choice(len(arrays), n_rows, replace=False))
            arrays = replace(
                arrays,
                rows=arrays.rows.iloc[chosen_rows].reset_index(drop=True),
                demographic_codes=arrays.demographic_codes[chosen_rows], history=arrays.history[chosen_rows],
                history_mask=arrays.history_mask[chosen_rows], option_matrix=arrays.option_matrix[chosen_rows],
                option_mask=arrays.option_mask[chosen_rows], log_prior=arrays.log_prior[chosen_rows],
                targets=arrays.targets[chosen_rows], ordinal_mask=arrays.ordinal_mask[chosen_rows],
                query_residual=(arrays.query_residual[chosen_rows] if arrays.query_residual is not None else None),
            )
        x, y, _ = self._candidate_features(arrays, fit=True)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("choice baseline requires both chosen and non-chosen candidates")
        if self.model_name == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                n_jobs=-1, random_state=self.seed,
            )
        elif self.model_name in {"hist_gbdt", "xgboost_proxy"}:
            self.model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1e-3, random_state=self.seed,
            )
        else:
            raise ValueError(f"unknown choice baseline: {self.model_name}")
        self.model.fit(x, y)
        return self

    def predict_proba(self, arrays):
        x, _, layout = self._candidate_features(arrays, fit=False, include_labels=False)
        if 1 not in self.model.classes_:
            raise ValueError("choice baseline did not learn a positive class")
        positive = self.model.predict_proba(x)[:, list(self.model.classes_).index(1)]
        output = np.zeros_like(arrays.option_mask, dtype=np.float32)
        for value, (row, option) in zip(positive, layout):
            output[row, option] = max(float(value), 1e-8)
        output *= arrays.option_mask
        return output / np.clip(output.sum(axis=1, keepdims=True), 1e-12, None)


class MatrixFactorizationChoice:
    """Warm-person/seen-item matrix-factorization baseline.

    It deliberately refuses unseen panels or questions; semantic HSRM is
    expected to cover those cells while this baseline measures the strength of
    ID memorization in the warm-person/seen-item regime.
    """

    def __init__(self, dimensions=32, max_options=11, seed=1701):
        self.dimensions = int(dimensions)
        self.max_options = int(max_options)
        self.seed = int(seed)

    def fit(self, rows, epochs=30, lr=0.03, batch_size=4096, device=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.panels = {value: i for i, value in enumerate(sorted(rows["panel_id"].astype(str).unique()))}
        self.items = {value: i for i, value in enumerate(sorted(rows["question_key"].astype(str).unique()))}
        torch.manual_seed(self.seed)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        class _MF(nn.Module):
            def __init__(inner, n_panels, n_items, dimensions, max_options):
                super().__init__()
                inner.user = nn.Embedding(n_panels, dimensions)
                inner.item_option = nn.Parameter(torch.randn(n_items, max_options, dimensions) * 0.05)
                inner.bias = nn.Parameter(torch.zeros(n_items, max_options))

            def forward(inner, p, q, n):
                logits = torch.einsum("bd,bcd->bc", inner.user(p), inner.item_option[q]) + inner.bias[q]
                mask = torch.arange(self.max_options, device=logits.device)[None] < n[:, None]
                return logits.masked_fill(~mask, -1e9)

        self.model = _MF(len(self.panels), len(self.items), self.dimensions, self.max_options).to(device)
        arrays = (
            np.asarray([self.panels[str(value)] for value in rows["panel_id"]]),
            np.asarray([self.items[str(value)] for value in rows["question_key"]]),
            rows["n_options"].to_numpy(np.int64), rows["answer_index"].to_numpy(np.int64),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        rng = np.random.default_rng(self.seed)
        for _ in range(int(epochs)):
            order = rng.permutation(len(rows))
            for start in range(0, len(order), int(batch_size)):
                idx = order[start : start + int(batch_size)]
                p, q, n, y = [torch.as_tensor(value[idx], device=device) for value in arrays]
                loss = F.cross_entropy(self.model(p, q, n), y)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        self.device = device
        return self

    def predict_proba(self, rows):
        import torch

        unknown_panels = set(rows["panel_id"].astype(str)) - set(self.panels)
        unknown_items = set(rows["question_key"].astype(str)) - set(self.items)
        if unknown_panels or unknown_items:
            raise ValueError("matrix factorization is undefined for unseen panels/items")
        p = torch.as_tensor([self.panels[str(v)] for v in rows["panel_id"]], device=self.device)
        q = torch.as_tensor([self.items[str(v)] for v in rows["question_key"]], device=self.device)
        n = torch.as_tensor(rows["n_options"].to_numpy(np.int64), device=self.device)
        with torch.no_grad():
            return torch.softmax(self.model(p, q, n), dim=-1).cpu().numpy().astype(np.float32)
