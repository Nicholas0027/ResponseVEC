"""Evaluation helpers for cold-item persona comparisons."""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def metrics(frame):
    probability = list(frame.probability)
    y = frame.answer_index.to_numpy(int)
    selected = np.asarray([p[a] for p, a in zip(probability, y)])
    nll = float(-np.log(np.maximum(selected, 1e-12)).mean())
    accuracy = float(np.mean([int(np.argmax(p) == a) for p, a in zip(probability, y)]))
    brier, rps = [], []
    for p, answer, ordinal in zip(probability, y, frame.is_ordinal):
        onehot = np.zeros(len(p)); onehot[answer] = 1
        brier.append(np.square(p - onehot).sum())
        if bool(ordinal) and len(p) > 1:
            rps.append(np.square(np.cumsum(p[:-1]) - np.cumsum(onehot[:-1])).sum() / (len(p) - 1))
    wd = []
    for _, group in frame.groupby("question_key"):
        if not bool(group.is_ordinal.iloc[0]):
            continue
        n = int(group.n_options.iloc[0])
        weights = group.survey_weight.to_numpy(float); weights /= weights.sum()
        predicted = np.average(np.stack(group.probability), axis=0, weights=weights)
        observed = np.zeros(n)
        for answer, weight in zip(group.answer_index.astype(int), weights):
            observed[answer] += weight
        position = np.linspace(0.0, 1.0, n)
        wd.append(wasserstein_distance(position, position, observed, predicted))
    return {"nll": nll, "brier": float(np.mean(brier)), "accuracy": accuracy,
            "rps": float(np.mean(rps)) if rps else np.nan,
            "population_wd": float(np.mean(wd)) if wd else np.nan, "rows": len(frame)}


def respondent_bootstrap_nll_gap(frame, left, right, seed=1701, draws=2000):
    subset = frame[frame.method.isin([left, right])]
    wide = subset.pivot(index=["panel_id", "row_id", "answer_index"],
                        columns="method", values="probability")
    if left not in wide or right not in wide:
        return np.nan, np.nan, np.nan
    values = wide.reset_index()
    values["gap"] = [-np.log(max(b[int(y)], 1e-12)) + np.log(max(a[int(y)], 1e-12))
                     for y, a, b in zip(values.answer_index, values[left], values[right])]
    person = values.groupby("panel_id").gap.mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(person, (draws, len(person)), replace=True).mean(axis=1)
    return float(person.mean()), *map(float, np.quantile(boot, [0.025, 0.975]))
