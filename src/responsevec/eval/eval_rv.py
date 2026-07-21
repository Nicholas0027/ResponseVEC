"""ResponseVec primary-claim evaluation (design §9).

The unseen-item claim generalizes over BOTH a population of respondents and a
population of items. A respondent-only bootstrap treats the item pool as fixed
and understates the CI when items are few (Protocol B holds out a small fold).
`crossed_bootstrap` resamples respondents AND items (two-way cluster bootstrap)
so the interval reflects both sources of variation (§9.1).

`primary_family` runs EXACTLY the three preregistered comparisons at the unseen
K=5 endpoint (§9.2):
    ResponseVec (LLM2Vec-Gen)  vs  best direct-output control
    ResponseVec                vs  raw causal hidden states
    ResponseVec                vs  input-centric LLM2Vec
each Holm-corrected at family alpha=0.05 AND required to clear a practical
effect of >=0.02 nats (NLL). A comparison counts as a win only if it is both
statistically significant and practically large — reported jointly.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .bootstrap import holm_correction


def _paired_cell_means(
    predictions: pd.DataFrame, method: str, reference: str, value_col: str
) -> pd.DataFrame:
    """Per (panel_id, question_key) paired difference candidate-reference. One
    row per target cell — the two-way clustering unit."""
    cell = predictions.groupby(
        ["method", "domain", "panel_id", "question_key"], as_index=False
    )[value_col].mean()
    left = cell[cell["method"].eq(method)].rename(columns={value_col: "candidate"})
    right = cell[cell["method"].eq(reference)].rename(columns={value_col: "reference"})
    paired = left.merge(
        right[["domain", "panel_id", "question_key", "reference"]],
        on=["domain", "panel_id", "question_key"],
    )
    if paired.empty:
        raise ValueError(f"No overlapping (panel_id, question_key) cells between {method!r} and {reference!r}")
    paired["difference"] = paired["candidate"] - paired["reference"]
    return paired


def crossed_bootstrap(
    predictions: pd.DataFrame,
    method: str,
    reference: str,
    metric: str = "nll",
    k: int | None = None,
    split: str = "test",
    item_pool: str | None = "unseen",
    replicates: int = 2000,
    seed: int = 1701,
) -> dict[str, float]:
    """Two-way (respondent x item) cluster bootstrap of the paired metric
    difference. Lower-is-better metrics (nll, rps, brier, mae) report
    difference = candidate - reference, so a NEGATIVE difference favors the
    candidate; accuracy is handled as higher-is-better via `correct`."""
    frame = predictions[predictions["split"].eq(split)]
    if k is not None:
        frame = frame[frame["k"].eq(k)]
    if item_pool is not None and "item_pool" in frame.columns:
        frame = frame[frame["item_pool"].eq(item_pool)]
    if frame.empty:
        raise ValueError(f"No rows for split={split} k={k} item_pool={item_pool}")

    value_col = "correct" if metric == "accuracy" else metric
    paired = _paired_cell_means(frame, method, reference, value_col)
    # Match respondent_macro_metrics: average respondent/item cells inside each
    # domain, then macro-average domains.
    observed = float(paired.groupby("domain")["difference"].mean().mean())

    matrices: list[np.ndarray] = []
    for _, group in paired.groupby("domain", sort=True):
        matrix = group.pivot_table(
            index="panel_id", columns="question_key", values="difference", aggfunc="mean"
        ).to_numpy(dtype=float)
        if matrix.size:
            matrices.append(matrix)
    if not matrices:
        raise ValueError("No domain matrices available for crossed bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        domain_means = []
        for matrix in matrices:
            respondent_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            item_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            # np.ix_ preserves BOTH respondent and item multiplicities. The old
            # implementation converted sampled items to a set and silently
            # destroyed the item bootstrap.
            sampled = matrix[np.ix_(respondent_indices, item_indices)]
            if np.isfinite(sampled).any():
                domain_means.append(float(np.nanmean(sampled)))
        draws[replicate] = float(np.mean(domain_means)) if domain_means else np.nan
    draws = draws[~np.isnan(draws)]
    if len(draws) < max(20, replicates // 2):
        raise ValueError("Too few valid crossed-bootstrap replicates")

    lower_is_better = metric != "accuracy"
    favors_candidate = observed < 0 if lower_is_better else observed > 0
    # Plus-one correction avoids impossible p=0 with a finite Monte Carlo
    # sample while retaining a valid [0,1] probability.
    p_left = (float(np.sum(draws <= 0)) + 1.0) / (len(draws) + 1.0)
    p_right = (float(np.sum(draws >= 0)) + 1.0) / (len(draws) + 1.0)
    return {
        "method": method, "reference": reference, "metric": metric,
        "k": -1 if k is None else int(k), "item_pool": item_pool or "all",
        "difference": observed,
        "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        "p_two_sided": float(min(1.0, 2.0 * min(p_left, p_right))),
        "n_panels": int(paired["panel_id"].nunique()), "n_items": int(paired["question_key"].nunique()),
        "favors_candidate": bool(favors_candidate),
    }


def primary_family(
    predictions: pd.DataFrame,
    responsevec_method: str,
    comparators: Sequence[str],
    metric: str = "nll",
    k: int = 5,
    item_pool: str = "unseen",
    practical_effect_nats: float = 0.02,
    replicates: int = 2000,
    seed: int = 1701,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """The preregistered primary family: ResponseVec vs each comparator at the
    unseen K endpoint. Holm across the family; a claim requires BOTH
    holm_reject_null AND |difference| >= practical_effect_nats in the favorable
    direction."""
    rows = []
    for reference in comparators:
        rows.append(
            crossed_bootstrap(
                predictions, responsevec_method, reference, metric=metric, k=k,
                item_pool=item_pool, replicates=replicates, seed=seed,
            )
        )
    table = pd.DataFrame(rows)
    holm = holm_correction(table["p_two_sided"].tolist(), alpha=alpha)
    table["holm_reject_null"] = [r["reject_null"] for r in holm]
    table["holm_threshold"] = [r["holm_threshold"] for r in holm]
    lower_is_better = metric != "accuracy"
    effect_ok = (-table["difference"] >= practical_effect_nats) if lower_is_better else (table["difference"] >= practical_effect_nats)
    table["practical_effect_met"] = effect_ok
    table["primary_claim_supported"] = table["holm_reject_null"] & effect_ok & table["favors_candidate"]
    return table


def evaluate_primary(
    predictions: pd.DataFrame,
    responsevec_method: str,
    comparators: Sequence[str],
    output_dir,
    metrics: Iterable[str] = ("nll", "rps"),
    k: int = 5,
    item_pool: str = "unseen",
    practical_effect_nats: float = 0.02,
    replicates: int = 2000,
    seed: int = 1701,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run the primary family for each primary metric (NLL primary, RPS
    co-primary) and write primary_family.csv."""
    from pathlib import Path

    tables = []
    for metric in metrics:
        table = primary_family(
            predictions, responsevec_method, comparators, metric=metric, k=k,
            item_pool=item_pool, practical_effect_nats=practical_effect_nats,
            replicates=replicates, seed=seed, alpha=alpha,
        )
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "primary_family.csv", index=False)
    return combined
