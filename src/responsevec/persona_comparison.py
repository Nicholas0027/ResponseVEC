"""Leakage-safe construction and calibration for persona comparisons."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from responsevec.persona_router import DEMO_COLS, stable_history


@dataclass
class HSRMArrays:
    """Lightweight choice-array container compatible with sklearn baselines."""
    rows: pd.DataFrame
    demographic_codes: np.ndarray
    history: np.ndarray
    history_mask: np.ndarray
    option_matrix: np.ndarray
    option_mask: np.ndarray
    log_prior: np.ndarray
    targets: np.ndarray
    ordinal_mask: np.ndarray
    query_residual: np.ndarray | None = None

    def __len__(self):
        return len(self.rows)


def inductive_pca_option_vectors(option_vectors: Mapping[str, np.ndarray],
                                 roles: Mapping[str, Sequence[str]], dimensions: int,
                                 seed: int = 1701):
    """Fit PCA on calibration/train item options and transform every item."""
    fit_keys = sorted((set(roles["calibration"]) | set(roles["train"])) & set(option_vectors))
    if not fit_keys:
        raise ValueError("no calibration/train option vectors available for PCA")
    fit = np.vstack([np.asarray(option_vectors[key]) for key in fit_keys])
    n_components = min(int(dimensions), fit.shape[0], fit.shape[1])
    pca = PCA(n_components=n_components, random_state=seed).fit(fit)
    transformed = {str(key): pca.transform(np.asarray(value)).astype(np.float32)
                   for key, value in option_vectors.items()}
    return transformed, pca, fit_keys


def build_choice_arrays(rows: pd.DataFrame, responses_for_same_split: pd.DataFrame,
                        option_vectors: Mapping[str, np.ndarray],
                        calibration_by_domain: Mapping[str, Sequence[str]], k: int,
                        seed: int, max_options: int | None = None) -> HSRMArrays:
    """Build target-independent choice arrays with calibration-only history."""
    rows = rows.reset_index(drop=True).copy()
    if responses_for_same_split.duplicated(["panel_id", "question_key"]).any():
        raise ValueError("duplicate respondent/question response")
    allowed_panels = set(responses_for_same_split.panel_id.astype(str))
    missing = set(rows.panel_id.astype(str)) - allowed_panels
    if missing:
        raise ValueError("target rows and history responses are from different respondent splits")
    lookup = {(str(r.panel_id), str(r.question_key)): int(r.answer_index)
              for r in responses_for_same_split.itertuples(index=False)}
    dim = next(iter(option_vectors.values())).shape[1]
    max_options = int(max_options or max(len(value) for value in option_vectors.values()))
    if max_options < max(len(option_vectors[str(q)]) for q in rows.question_key):
        raise ValueError("max_options is smaller than a target item's option count")
    option_matrix = np.zeros((len(rows), max_options, dim), np.float32)
    option_mask = np.zeros((len(rows), max_options), np.float32)
    history = np.zeros((len(rows), int(k), dim), np.float32)
    history_mask = np.zeros((len(rows), int(k)), np.float32)
    log_prior = np.full((len(rows), max_options), -np.inf, np.float32)
    for i, row in enumerate(rows.itertuples(index=False)):
        options = np.asarray(option_vectors[str(row.question_key)], np.float32)
        n = len(options)
        if int(row.n_options) != n:
            raise ValueError(f"option count mismatch for {row.question_key}")
        option_matrix[i, :n] = options
        option_mask[i, :n] = 1.0
        log_prior[i, :n] = -np.log(n)
        calibration = [q for q in calibration_by_domain.get(str(row.domain), ())
                       if str(q) != str(row.question_key)]
        for j, question in enumerate(stable_history(str(row.panel_id), calibration, k, seed)):
            answer = lookup.get((str(row.panel_id), str(question)))
            if answer is None:
                continue
            source = np.asarray(option_vectors[str(question)], np.float32)
            if answer < 0 or answer >= len(source):
                raise ValueError(f"invalid calibration answer for {question}")
            history[i, j] = source[answer]
            history_mask[i, j] = 1.0
    targets = rows.answer_index.to_numpy(np.int64) if "answer_index" in rows else np.zeros(len(rows), np.int64)
    ordinal = rows.is_ordinal.to_numpy(bool) if "is_ordinal" in rows else np.zeros(len(rows), bool)
    return HSRMArrays(rows, np.zeros((len(rows), 0), np.int64), history, history_mask,
                      option_matrix, option_mask, log_prior, targets, ordinal)


def fit_position_prior(rows: pd.DataFrame, demographic: bool = False,
                       alpha: float = 0.5, minimum_group_n: int = 25) -> dict[str, Any]:
    """Estimate smoothed padded-position counts from the supplied rows only."""
    required = {"answer_index", "n_options"}
    if not required <= set(rows):
        raise ValueError(f"position-prior rows lack {sorted(required - set(rows))}")
    width = int(rows.n_options.max())
    groups: dict[str, list[float]] = {}
    group_n: dict[str, int] = {}
    columns = [c for c in DEMO_COLS if c in rows] if demographic else []
    iterator = rows.groupby(columns, dropna=False, sort=True) if columns else [((), rows)]
    for key, frame in iterator:
        key = key if isinstance(key, tuple) else (key,)
        count = np.full(width, float(alpha))
        for answer in frame.answer_index.astype(int):
            if answer < 0 or answer >= width:
                raise ValueError("answer index outside padded position prior")
            count[answer] += 1
        encoded = "\x1f".join(map(str, key))
        groups[encoded] = count.tolist()
        group_n[encoded] = len(frame)
    fallback = np.full(width, float(alpha))
    for answer in rows.answer_index.astype(int):
        fallback[answer] += 1
    return {"width": width, "alpha": float(alpha), "columns": columns,
            "minimum_group_n": int(minimum_group_n), "groups": groups,
            "group_n": group_n, "fallback": fallback.tolist()}


def predict_position_prior(model: Mapping[str, Any], rows: pd.DataFrame) -> list[np.ndarray]:
    columns, width, alpha = model["columns"], int(model["width"]), float(model["alpha"])
    minimum = int(model.get("minimum_group_n", 0))
    output = []
    for row in rows.itertuples(index=False):
        n = int(row.n_options)
        key = "\x1f".join(str(getattr(row, col)) for col in columns)
        eligible = int(model.get("group_n", {}).get(key, 0)) >= minimum
        source = np.asarray(model["groups"].get(key, model["fallback"]) if eligible
                            else model["fallback"], float)
        count = np.full(n, alpha)
        count[:min(n, width)] = source[:min(n, width)]
        output.append(count / count.sum())
    return output


def log_opinion_pool(p_stat, p_llm, lam: float):
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    if isinstance(p_stat, np.ndarray) and p_stat.ndim == 1:
        logp = (1.0 - lam) * np.log(np.maximum(p_stat, 1e-12)) + lam * np.log(np.maximum(p_llm, 1e-12))
        logp -= logp.max(); out = np.exp(logp)
        return out / out.sum()
    rows = [log_opinion_pool(np.asarray(a), np.asarray(b), lam) for a, b in zip(p_stat, p_llm)]
    return np.stack(rows) if len({len(row) for row in rows}) <= 1 else rows


def fit_lambda(p_stat, p_llm, labels, grid=None) -> float:
    grid = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(grid, float)
    labels = np.asarray(labels, int)
    losses = []
    for lam in grid:
        p = log_opinion_pool(p_stat, p_llm, float(lam))
        losses.append(np.mean([-np.log(max(row[label], 1e-12)) for row, label in zip(p, labels)]))
    return float(grid[int(np.argmin(losses))])


def temperature_scale(probabilities, temperature: float):
    if isinstance(probabilities, np.ndarray) and probabilities.ndim == 1:
        p = np.power(np.maximum(probabilities.astype(float), 1e-12), 1.0 / float(temperature))
        return p / p.sum()
    rows = [temperature_scale(np.asarray(row), temperature) for row in probabilities]
    return np.stack(rows) if len({len(row) for row in rows}) <= 1 else rows


def fit_temperature(probabilities, labels) -> float:
    from scipy.optimize import minimize_scalar
    p, labels = list(probabilities), np.asarray(labels, int)
    def objective(log_t):
        scaled = temperature_scale(p, np.exp(log_t))
        return float(np.mean([-np.log(max(row[label], 1e-12)) for row, label in zip(scaled, labels)]))
    return float(np.exp(minimize_scalar(objective, bounds=(-3.0, 5.0), method="bounded").x))


def respondent_split(panel_ids: Sequence[str], seed: int) -> tuple[set[str], set[str]]:
    def side(panel):
        digest = hashlib.sha256(f"{seed}|{panel}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % 2
    ids = sorted(set(map(str, panel_ids)))
    return ({p for p in ids if side(p) == 0}, {p for p in ids if side(p) == 1})


def strict_align(left: pd.DataFrame, right: pd.DataFrame,
                 metadata=("panel_id", "question_key", "answer_index", "n_options")) -> pd.DataFrame:
    """One-to-one row_id join that rejects metadata disagreement."""
    for name, frame in (("left", left), ("right", right)):
        missing = ({"row_id"} | set(metadata)) - set(frame)
        if missing:
            raise ValueError(f"{name} lacks alignment columns {sorted(missing)}")
        if frame.row_id.astype(str).duplicated().any():
            raise ValueError(f"{name} has missing or duplicate row_id")
        if frame[["row_id", *metadata]].astype(str).duplicated().any():
            raise ValueError(f"{name} has duplicate alignment key")
    l, r = left.copy(), right.copy()
    l["row_id"] = l.row_id.astype(str); r["row_id"] = r.row_id.astype(str)
    if set(l.row_id) != set(r.row_id):
        raise ValueError("row_id coverage mismatch")
    joined = l.merge(r, on="row_id", how="inner", validate="one_to_one", suffixes=("_left", "_right"))
    for column in metadata:
        a, b = f"{column}_left", f"{column}_right"
        if a in joined and b in joined and not np.array_equal(joined[a].astype(str), joined[b].astype(str)):
            raise ValueError(f"alignment mismatch in {column}")
    return joined
