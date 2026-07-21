#!/usr/bin/env python
"""R2 (cross-domain) and R3 (OOD-demographic-intersection) transfer evaluation.

Extends evaluate_primary.py — which owns the R1 unseen-item regime — with the
two transfer regimes of design §5.3/§5.4. Reuses the same frozen representation
caches (produced by extract_representations.py --protocol C|D), the same
option-aware decoder, and the same leakage-safe protocol builders; only the
target units and (for R2) the decoder's training-domain filter change.

Prerequisites (run once per family/k/option-seed before this script):
    # R2: per held-out domain, test split
    python scripts/extract_representations.py --protocol C \
        --held-out-domain <domain> --family response_centric \
        --split test --k 5 --option-seed 0 --synthetic-encoder   # smoke
    # R3: OOD-intersection split (split flag ignored, always ood_intersection)
    python scripts/extract_representations.py --protocol D \
        --family response_centric --split test --k 5 --option-seed 0 --synthetic-encoder
    # Shared training cache: Protocol A seen items of the ID train split
    python scripts/extract_representations.py --protocol A \
        --family response_centric --split train --k 5 --option-seed 0 --synthetic-encoder

Regime -> protocol mapping
--------------------------
R2 cross-domain        : build_protocol_c(store, held_out_domain, split="test")
                         decoder trained on the OTHER three domains' Protocol-A
                         seen-item train units (filtered by domain != held_out).
R3 OOD-intersection    : build_protocol_d(store)   # split="ood_intersection"
                         decoder trained on ID-respondent Protocol-A train split.

Outputs
-------
metrics/k_{k}/transfer_r2_<domain>.csv   per held-out domain
metrics/k_{k}/transfer_r3_ood.csv        OOD-intersection summary
metrics/k_{k}/transfer_cost.csv          per-method GPU-hours/latency (Figure 2)

This script does NOT gate G1 (the R1 primary claim). It reports transfer
degradation and cost; the gate lives in evaluate_primary.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from responsevec.cache import RepresentationCache
from responsevec.config import load_and_prepare
from responsevec.data import PanelStore
from responsevec.encode import load_option_table
from responsevec.eval.cost import CostRecord, save_cost_table, timed_extraction
from responsevec.eval.metrics import add_probability_scores, compute_metric_tables
from responsevec.pipeline import protocol_cache_directory
from responsevec.prior import PopulationPrior
from responsevec.protocols import (
    ItemFolds,
    build_protocol_c,
    build_protocol_d,
)
from responsevec.training import (
    arrays_from_cache,
    predict_head,
    prediction_frame,
    train_option_head,
)
from responsevec.utils import write_json


def _load_arrays(config, protocol, split, k, option_seed, family, option_table,
                 prior, train_keys, held_out_domain=None):
    """Load a regime cache and build HeadArrays (mirrors train_primary.load_arrays
    but for protocol C/D caches instead of the shared Protocol-B cache)."""
    directory = protocol_cache_directory(
        config["paths"]["cache"], protocol=protocol, respondent_split=split,
        k=k, option_seed=option_seed, family=family,
        held_out_domain=held_out_domain,
    )
    cache = RepresentationCache.load(directory)
    return arrays_from_cache(
        cache, option_table, prior, train_keys,
        max_options=int(config["decoder"].get("max_options", 11)),
    )


def _filter_by_domain(arrays, keep_domains):
    """Subset HeadArrays to rows whose domain is in keep_domains (R2 training
    excludes the held-out domain)."""
    keep = arrays.rows["domain"].astype(str).isin(set(str(d) for d in keep_domains)).to_numpy()
    if not keep.any():
        raise ValueError("domain filter removed every training row")
    from responsevec.training import HeadArrays
    return HeadArrays(
        rows=arrays.rows.loc[keep].reset_index(drop=True),
        z=arrays.z[keep], option_matrix=arrays.option_matrix[keep],
        option_mask=arrays.option_mask[keep], log_prior=arrays.log_prior[keep],
        targets=arrays.targets[keep], ordinal_mask=arrays.ordinal_mask[keep],
        direct_probabilities=(arrays.direct_probabilities[keep] if arrays.direct_probabilities is not None else None),
    )


def _run_one_method(method_name, cache_family, config, train_arrays, test_arrays,
                    regime, k, n_test_respondents, n_forward_passes, decoder_common):
    """Train one representation family's decoder on train_arrays, predict on
    test_arrays, and return (predictions_df, cost_record). Wrapped in
    timed_extraction so the cost table reflects real extraction+head time."""
    primary = config["decoder"]["primary"]
    with timed_extraction(method_name, regime, k, n_test_respondents, n_forward_passes) as rec:
        fit = train_option_head(
            train_arrays, test_arrays,  # validation = test here for transfer (no fold val split)
            temperature_init=float(primary["temperature_init"]),
            rps_lambda=float(primary["rps_lambda"]),
            seed=int(config["seed"]), **decoder_common,
        )
        probabilities = predict_head(fit.model, test_arrays, device=decoder_common.get("device"))
    predictions = prediction_frame(test_arrays, probabilities, method_name)
    return predictions, rec


def _run_regime(methods, regime, test_arrays_by_family, train_arrays_by_family, config,
                metric_dir, k, n_test_respondents, n_forward_passes_per_family):
    """Score a set of methods on one regime. Returns (predictions_df, cost_records)."""
    decoder_common = dict(
        projection_dim=int(config["decoder"]["projection_dim"]),
        dropout=float(config["decoder"]["dropout"]),
        prior_eta=float(config["decoder"]["primary"]["prior_eta"]), learnable_eta=False,
        lr=float(config["decoder"]["primary"]["lr"]),
        weight_decay=float(config["decoder"]["primary"]["weight_decay"]),
        epochs=int(config["decoder"]["epochs"]),
        patience=int(config["decoder"]["early_stopping_patience"]),
        batch_size=int(config["decoder"]["batch_size"]),
        gradient_clip=float(config["decoder"]["gradient_clip"]),
        device=None,
    )
    all_predictions = []
    cost_records = []
    for method, family in methods:
        if family not in test_arrays_by_family:
            continue
        preds, rec = _run_one_method(
            method, family, config, train_arrays_by_family[family],
            test_arrays_by_family[family], regime, k, n_test_respondents,
            n_forward_passes_per_family.get(family, 0), decoder_common,
        )
        all_predictions.append(preds)
        cost_records.append(rec)
    return (pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()), cost_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--option-seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    store = PanelStore.from_dir(config["paths"]["processed"])
    folds = ItemFolds.load(Path(config["paths"]["processed"]) / "item_folds.json")
    metric_dir = Path(config["paths"]["metrics"]) / f"k_{args.k}"
    metric_dir.mkdir(parents=True, exist_ok=True)

    prior_config = config["prior"]
    prior = PopulationPrior(prior_config["country_shrinkage"], prior_config["laplace"])
    # For transfer regimes the prior is fit on ALL training items (no fold holdout).
    prior.fit(store.responses[store.responses["split"].eq("train")])
    train_keys = frozenset(store.responses[store.responses["split"].eq("train")
                            & ~store.responses["is_unseen_item"]]["question_key"].astype(str))
    option_table = load_option_table(Path(config["paths"]["cache"]) / "option_table.npz")

    # Representation families to evaluate (mirrors train_primary's set).
    families = {
        "causal_final": "causal_final", "raw_mean": "raw_mean",
        "input_centric": "input_centric", "response_centric": "response_centric",
        "sentence": "sentence",
    }
    method_names = {
        "causal_final": "raw_selected", "raw_mean": "raw_mean",
        "input_centric": "input_centric", "response_centric": "response_centric",
        "sentence": "sentence",
    }

    all_costs = []
    all_predictions = []

    # Shared Protocol-A training cache (seen items of ID train split).
    train_arrays_by_family = {}
    for family in families.values():
        try:
            train_arrays_by_family[family] = _load_arrays(
                config, "A", "train", args.k, args.option_seed, family,
                option_table, prior, train_keys,
            )
        except FileNotFoundError:
            print(f"[skip] no Protocol-A train cache for family={family}; "
                  f"run extract_representations.py --protocol A --family {family} --split train first")

    domains = list(store.responses["domain"].unique())

    # R2: leave-one-domain-out.
    for held in domains:
        test_arrays_by_family = {}
        for family in families.values():
            try:
                arr = _load_arrays(
                    config, "C", "test", args.k, args.option_seed, family,
                    option_table, prior, train_keys, held_out_domain=held,
                )
                test_arrays_by_family[family] = arr
            except FileNotFoundError:
                print(f"[skip] no Protocol-C cache for family={family} domain={held}")
        if not test_arrays_by_family:
            continue
        keep_domains = [d for d in domains if d != held]
        train_filtered = {
            f: _filter_by_domain(a, keep_domains) for f, a in train_arrays_by_family.items() if f in test_arrays_by_family
        }
        n_resp = int(test_arrays_by_family[next(iter(test_arrays_by_family))].rows["panel_id"].nunique())
        methods = [(method_names[f], f) for f in test_arrays_by_family.keys()]
        n_fwd = {f: len(test_arrays_by_family[f]) for f in test_arrays_by_family}
        preds, costs = _run_regime(
            methods, "R2", test_arrays_by_family, train_filtered, config,
            metric_dir, args.k, n_resp, n_fwd,
        )
        if not preds.empty:
            preds.to_parquet(metric_dir / f"transfer_r2_{held}.parquet", index=False)
        all_costs.extend(costs)
        all_predictions.append(preds)
        write_json(metric_dir / f"transfer_r2_{held}.json",
                   {"held_out_domain": held, "n_test_respondents": n_resp, "n_methods": len(methods)})

    # R3: OOD-demographic-intersection.
    test_arrays_by_family = {}
    for family in families.values():
        try:
            test_arrays_by_family[family] = _load_arrays(
                config, "D", "ood_intersection", args.k, args.option_seed, family,
                option_table, prior, train_keys,
            )
        except FileNotFoundError:
            print(f"[skip] no Protocol-D cache for family={family}; "
                  f"run extract_representations.py --protocol D --family {family} first")
    if test_arrays_by_family:
        n_resp = int(test_arrays_by_family[next(iter(test_arrays_by_family))].rows["panel_id"].nunique())
        methods = [(method_names[f], f) for f in test_arrays_by_family.keys()]
        n_fwd = {f: len(test_arrays_by_family[f]) for f in test_arrays_by_family}
        preds, costs = _run_regime(
            methods, "R3", test_arrays_by_family, train_arrays_by_family, config,
            metric_dir, args.k, n_resp, n_fwd,
        )
        if not preds.empty:
            preds.to_parquet(metric_dir / "transfer_r3_ood.parquet", index=False)
        all_costs.extend(costs)
        all_predictions.append(preds)
        write_json(metric_dir / "transfer_r3_ood.json",
                   {"regime": "R3", "n_test_respondents": n_resp, "n_methods": len(methods)})

    # M3 mixed-allocation cost (post-hoc, no new extraction).
    from responsevec.eval.cost import mixed_allocation_cost
    m1 = next((c for c in all_costs if c.method == "response_centric"), None)
    m2 = next((c for c in all_costs if c.method == "input_centric"), None)
    if m1 and m2:
        for frac in (0.25, 0.5, 0.75):
            all_costs.append(CostRecord(
                method="mixed_allocation", regime=m1.regime, k=args.k,
                n_respondents=m1.n_respondents,
                n_forward_passes=int(frac * m1.n_forward_passes + (1 - frac) * m2.n_forward_passes),
                elapsed_seconds=mixed_allocation_cost(m1.gpu_hours_per_1k, m2.gpu_hours_per_1k, frac) * 3600.0,
            ))

    save_cost_table(all_costs, metric_dir / "transfer_cost.csv")
    if all_predictions:
        combined = pd.concat([p for p in all_predictions if not p.empty], ignore_index=True)
        if not combined.empty:
            combined = add_probability_scores(combined)
            compute_metric_tables(combined, metric_dir / "transfer")
    print({"cost_records": len(all_costs), "prediction_rows": sum(len(p) for p in all_predictions if not p.empty),
           "path": str(metric_dir / "transfer_cost.csv")})


if __name__ == "__main__":
    main()
