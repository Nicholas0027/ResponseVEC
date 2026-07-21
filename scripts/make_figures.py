#!/usr/bin/env python
"""Generate the compact preregistered main-result plots from prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from responsevec.config import load_and_prepare
from responsevec.eval.metrics import respondent_macro_metrics


METHOD_LABELS = {
    "direct_selected": "Direct output (selected)",
    "raw_selected": "Raw hidden state",
    "input_centric": "Input-centric",
    "response_centric": "ResponseVec",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    paths = sorted(Path(config["paths"]["metrics"]).glob("k_*/predictions_all_item_folds.parquet"))
    if not paths:
        raise FileNotFoundError("run evaluate_primary.py before make_figures.py")
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    macro = respondent_macro_metrics(predictions)
    data = macro[
        macro["domain"].eq("macro")
        & macro["item_pool"].eq("unseen")
        & macro["method"].isin(METHOD_LABELS)
    ].copy()
    figure_dir = Path(config["paths"]["figures"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method, group in data.groupby("method"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["nll"], marker="o", label=METHOD_LABELS[method])
    ax.set(xlabel="Observed history answers (K)", ylabel="Respondent-macro NLL", title="Unseen-item prediction")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure2_nll_vs_k.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "figure2_nll_vs_k.png", dpi=220, bbox_inches="tight")
    print(figure_dir / "figure2_nll_vs_k.pdf")


if __name__ == "__main__":
    main()
