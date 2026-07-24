"""Leakage-safe item splits, persona banks, and Bayesian persona routing."""
from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

DEMO_COLS = ["country", "sex", "age_bin", "education", "income_quintile"]


def build_item_split(items: pd.DataFrame, option_table: Mapping[str, np.ndarray],
                     n_calibration: int = 8, n_folds: int = 6,
                     clusters_per_domain: int = 12, seed: int = 1701) -> dict[str, Any]:
    """Select whole semantic clusters for calibration, then balance targets."""
    out: dict[str, Any] = {"seed": seed, "n_folds": n_folds, "domains": {}}
    for domain, frame in items.groupby("domain", sort=True):
        keys = sorted(frame.question_key.astype(str).unique())
        if len(keys) < n_calibration + n_folds:
            raise ValueError(f"{domain}: need at least {n_calibration + n_folds} items")
        means = np.stack([np.asarray(option_table[k]).mean(0) for k in keys])
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
        nc = min(clusters_per_domain, len(keys) - n_folds + 1)
        labels = AgglomerativeClustering(n_clusters=nc, metric="cosine", linkage="average").fit_predict(means)
        groups = [sorted(keys[i] for i in np.flatnonzero(labels == c)) for c in range(nc)]
        choices = []
        for size in range(1, nc + 1):
            for chosen in combinations(range(nc), size):
                total = sum(len(groups[c]) for c in chosen)
                if len(keys) - total >= n_folds:
                    choices.append((abs(total - n_calibration), total > n_calibration, size, chosen))
        chosen = min(choices)[-1]
        calibration = sorted(k for c in chosen for k in groups[c])
        target = sorted(set(keys) - set(calibration))
        rng = np.random.default_rng(seed + sum(map(ord, domain)))
        ordered = [target[i] for i in rng.permutation(len(target))]
        folds = [sorted(ordered[f::n_folds]) for f in range(n_folds)]
        out["domains"][domain] = {
            "semantic_clusters": {str(i): g for i, g in enumerate(groups)},
            "calibration": calibration, "target_folds": folds,
        }
    return out


def fold_roles(split: Mapping[str, Any], fold: int) -> dict[str, list[str]]:
    roles = {"calibration": [], "train": [], "validation": [], "test": []}
    nf = int(split["n_folds"])
    for data in split["domains"].values():
        roles["calibration"] += data["calibration"]
        roles["test"] += data["target_folds"][fold]
        roles["validation"] += data["target_folds"][(fold + 1) % nf]
        roles["train"] += [q for i, fs in enumerate(data["target_folds"])
                           if i not in (fold, (fold + 1) % nf) for q in fs]
    return {k: sorted(v) for k, v in roles.items()}


def stable_history(panel_id: str, calibration: Sequence[str], k: int, seed: int) -> list[str]:
    import hashlib
    def score(q: str) -> bytes:
        return hashlib.sha256(f"{seed}|{panel_id}|{q}".encode()).digest()
    return sorted(calibration, key=score)[:max(0, k)]


def make_profiles(rows: pd.DataFrame, panels: Sequence[str], questions: Sequence[str],
                  max_options: int) -> np.ndarray:
    qi = {q: i for i, q in enumerate(questions)}
    pi = {p: i for i, p in enumerate(panels)}
    x = np.zeros((len(panels), len(questions) * max_options), np.float32)
    for row in rows[rows.question_key.isin(qi)].itertuples():
        if row.panel_id in pi and 0 <= int(row.answer_index) < max_options:
            x[pi[row.panel_id], qi[row.question_key] * max_options + int(row.answer_index)] = 1
    return x


def fit_persona_bank(train: pd.DataFrame, calibration: Sequence[str], m: int = 4,
                     alpha: float = 0.5, seed: int = 1701) -> dict[str, Any]:
    """Fit domain-local KMeans and smoothed P(answer | persona, question)."""
    bank: dict[str, Any] = {"m": m, "alpha": alpha, "domains": {}}
    for domain, rows in train.groupby("domain", sort=True):
        cal = sorted(set(calibration) & set(rows.question_key))
        panels = sorted(rows.panel_id.unique())
        maxopt = int(rows.n_options.max())
        x = make_profiles(rows, panels, cal, maxopt)
        if len(panels) < m:
            raise ValueError(f"{domain}: fewer than {m} train respondents")
        km = KMeans(n_clusters=m, n_init=20, random_state=seed).fit(x)
        labels = dict(zip(panels, map(int, km.labels_)))
        population: dict[str, list[float]] = {}
        probs: dict[str, dict[str, list[float]]] = {str(z): {} for z in range(m)}
        for q in cal:
            qr = rows[rows.question_key.eq(q)]
            nopt = int(qr.n_options.iloc[0])
            pop = np.bincount(qr.answer_index.astype(int), minlength=nopt) + alpha
            population[q] = (pop / pop.sum()).tolist()
            for z in range(m):
                zr = qr[qr.panel_id.map(labels).eq(z)]
                cnt = np.bincount(zr.answer_index.astype(int), minlength=nopt) + alpha
                probs[str(z)][q] = (cnt / cnt.sum()).tolist()
        text = _persona_text(rows, cal, probs, population, m)
        bank["domains"][domain] = {"calibration": cal, "max_options": maxopt,
            "centers": km.cluster_centers_.tolist(), "panel_cluster": labels,
            "response_prob": probs, "population_prob": population, "persona_text": text}
    return bank


def _persona_text(rows: pd.DataFrame, cal: Sequence[str], probs: Mapping[str, Any],
                  population: Mapping[str, Any], m: int) -> dict[str, str]:
    catalogue = rows[rows.question_key.isin(cal)].drop_duplicates("question_key").set_index("question_key")
    out = {}
    for z in range(m):
        ranked = sorted(cal, key=lambda q: (-float(np.max(np.abs(np.asarray(probs[str(z)][q]) - population[q]))), q))[:3]
        lines = []
        for q in ranked:
            row = catalogue.loc[q]
            options = json.loads(row.options_json) if isinstance(row.options_json, str) else row.options_json
            answer = options[int(np.argmax(probs[str(z)][q]))]
            lines.append(f"{row.question} -> {answer}")
        out[str(z)] = "Persona tendencies (calibration only): " + "; ".join(lines)
    return out


def assign_personas(rows: pd.DataFrame, domain_bank: Mapping[str, Any]) -> dict[str, int]:
    panels = sorted(rows.panel_id.unique())
    x = make_profiles(rows, panels, domain_bank["calibration"], int(domain_bank["max_options"]))
    centers = np.asarray(domain_bank["centers"])
    return dict(zip(panels, np.argmin(((x[:, None] - centers[None]) ** 2).sum(2), axis=1).astype(int)))


def fit_demographic_router(train_panels: pd.DataFrame, labels: Mapping[str, int]):
    frame = train_panels.drop_duplicates("panel_id").copy()
    y = frame.panel_id.map(labels).astype(int)
    enc = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), DEMO_COLS))
    x = enc.fit_transform(frame)
    model = LogisticRegression(max_iter=2000, random_state=0).fit(x, y)
    return enc, model


def router_prior(router, panels: pd.DataFrame, m: int) -> dict[str, np.ndarray]:
    enc, model = router
    frame = panels.drop_duplicates("panel_id")
    raw = model.predict_proba(enc.transform(frame))
    full = np.zeros((len(frame), m), float)
    full[:, model.classes_.astype(int)] = raw
    return dict(zip(frame.panel_id, full / full.sum(1, keepdims=True)))


def update_posterior(prior: Sequence[float], history: Sequence[tuple[str, int]],
                     response_prob: Mapping[str, Mapping[str, Sequence[float]]]) -> np.ndarray:
    logp = np.log(np.maximum(np.asarray(prior, float), 1e-300))
    for q, answer in history:
        for z in range(len(logp)):
            p = response_prob[str(z)][q]
            logp[z] += np.log(max(float(p[int(answer)]), 1e-300))
    logp -= logp.max()
    post = np.exp(logp)
    return post / post.sum()


def question_aligned_shuffle(test: pd.DataFrame, histories: Mapping[str, Sequence[str]],
                             seed: int = 1701) -> dict[str, list[tuple[str, int]]]:
    """Shuffle answers among test respondents, preserving every question key."""
    lookup = {(r.panel_id, r.question_key): int(r.answer_index) for r in test.itertuples()}
    demos = test.drop_duplicates("panel_id").set_index("panel_id")
    answerers = {q: sorted(g.panel_id.unique()) for q, g in test.groupby("question_key")}
    out = {}
    for pid, questions in histories.items():
        values = []
        for q in questions:
            pool = [p for p in answerers.get(q, []) if p != pid]
            matched = [p for p in pool if all(str(demos.loc[p, c]) == str(demos.loc[pid, c])
                                              for c in ("country", "sex", "age_bin"))]
            pool = matched or pool
            if pool:
                import hashlib
                donor = min(pool, key=lambda p: hashlib.sha256(f"{seed}|{pid}|{q}|{p}".encode()).digest())
                values.append((q, lookup[(donor, q)]))
            elif (pid, q) in lookup:
                values.append((q, lookup[(pid, q)]))
        out[pid] = values
    return out
