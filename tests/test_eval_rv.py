from __future__ import annotations

import numpy as np
import pandas as pd

from responsevec.eval.eval_rv import crossed_bootstrap, primary_family


def _synthetic_predictions(candidate_gain, n_panels=30, n_items=8, noise=0.05, seed=0):
    """Two methods over (panel, item) cells; candidate has lower NLL by
    candidate_gain on average. Returns a predictions frame with the columns the
    bootstrap needs."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_panels):
        for i in range(n_items):
            base = 1.0 + rng.normal(scale=noise)
            rows.append({"method": "reference", "panel_id": f"p{p}", "question_key": f"q{i}",
                         "domain": "D", "split": "test", "k": 5, "item_pool": "unseen",
                         "nll": base, "correct": 0.0})
            rows.append({"method": "cand", "panel_id": f"p{p}", "question_key": f"q{i}",
                         "domain": "D", "split": "test", "k": 5, "item_pool": "unseen",
                         "nll": base - candidate_gain + rng.normal(scale=noise), "correct": 0.0})
    return pd.DataFrame(rows)


def test_crossed_bootstrap_favors_better_candidate():
    preds = _synthetic_predictions(candidate_gain=0.10, seed=1)
    result = crossed_bootstrap(preds, "cand", "reference", metric="nll", k=5, replicates=500, seed=1)
    assert result["difference"] < 0                       # lower NLL is better
    assert result["favors_candidate"] is True
    assert result["ci_high"] < 0                          # whole CI below zero
    assert result["n_items"] == 8 and result["n_panels"] == 30


def test_crossed_bootstrap_wider_than_respondent_only():
    """The two-way bootstrap must not be narrower than resampling panels alone;
    adding item resampling injects additional variance."""
    preds = _synthetic_predictions(candidate_gain=0.08, noise=0.15, seed=2)
    crossed = crossed_bootstrap(preds, "cand", "reference", metric="nll", k=5, replicates=800, seed=2)

    # respondent-only reference: resample panels, keep all items
    paired = preds.pivot_table(index=["panel_id", "question_key"], columns="method", values="nll")
    diff = (paired["cand"] - paired["reference"]).groupby("panel_id").mean()
    rng = np.random.default_rng(2)
    panels = diff.index.to_numpy()
    draws = np.array([rng.choice(diff.to_numpy(), size=len(panels), replace=True).mean() for _ in range(800)])
    resp_width = np.quantile(draws, 0.975) - np.quantile(draws, 0.025)
    crossed_width = crossed["ci_high"] - crossed["ci_low"]
    assert crossed_width >= 0.9 * resp_width               # at least comparable, generally wider


def test_primary_family_requires_practical_effect():
    # gain of 0.10 nats: significant AND practically large -> claim supported
    big = _synthetic_predictions(candidate_gain=0.10, seed=3)
    table = primary_family(big, "cand", ["reference"], metric="nll", k=5,
                           practical_effect_nats=0.02, replicates=500, seed=3)
    assert bool(table["primary_claim_supported"].iloc[0]) is True

    # gain of 0.005 nats: may be significant with low noise but BELOW threshold
    tiny = _synthetic_predictions(candidate_gain=0.005, noise=0.01, seed=4)
    table_tiny = primary_family(tiny, "cand", ["reference"], metric="nll", k=5,
                                practical_effect_nats=0.02, replicates=500, seed=4)
    assert bool(table_tiny["practical_effect_met"].iloc[0]) is False
    assert bool(table_tiny["primary_claim_supported"].iloc[0]) is False


def test_primary_family_holm_across_three_comparisons():
    preds = _synthetic_predictions(candidate_gain=0.10, seed=5)
    # replicate reference into three named comparators
    frames = [preds]
    for name in ["hidden", "input_centric"]:
        clone = preds[preds["method"].eq("reference")].copy()
        clone["method"] = name
        frames.append(clone)
    full = pd.concat(frames, ignore_index=True)
    table = primary_family(full, "cand", ["reference", "hidden", "input_centric"],
                           metric="nll", k=5, replicates=500, seed=5)
    assert len(table) == 3
    assert "holm_threshold" in table.columns
    assert table["primary_claim_supported"].all()
