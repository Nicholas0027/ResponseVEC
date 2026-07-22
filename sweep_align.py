#!/usr/bin/env python
"""Hyperparameter sweep for ResponseVec-Align on the REAL SocioBench fold-0
cache. For each config we fit the aligner on the response_centric train/val
cache, report its held-out contrastive validation loss AND the downstream
option-aware decoder validation NLL on g_phi(z), against the raw response_centric
and sentence baselines. Goal: find a config where response_centric_aligned's
decoder val NLL clearly beats sentence (and, ideally, raw_mean_aligned).

Runs on the already-extracted caches only -- no 8B forward passes.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from responsevec.config import load_and_prepare
from responsevec.cache import RepresentationCache
from responsevec.data import PanelStore
from responsevec.encode import load_option_table
from responsevec.pipeline import fit_fold_prior, protocol_b_item_keys, shared_cache_directory
from responsevec.prior import PopulationPrior
from responsevec.protocols import ItemFolds
from responsevec.training import (
    apply_aligner, arrays_from_cache, subset_arrays,
    train_aligner, train_option_head,
)

CONFIG = "configs/responsevec.yaml"
FOLD, K, SEED = 0, 5, 1701


def load_family(config, family, split, prior, train_keys, role_keys, role, option_table):
    directory = shared_cache_directory(config["paths"]["cache"], respondent_split=split, k=K, option_seed=0, family=family)
    shared = arrays_from_cache(RepresentationCache.load(directory), option_table, prior, train_keys,
                              max_options=int(config["decoder"].get("max_options", 11)))
    return subset_arrays(shared, role_keys[role])


def decoder_val_nll(train_arrays, val_arrays):
    fit = train_option_head(train_arrays, val_arrays, projection_dim=256, dropout=0.1,
                            temperature_init=0.1, rps_lambda=0.1, lr=1e-3,
                            epochs=100, patience=10, batch_size=256, seed=SEED)
    return fit.best_validation_nll


def main():
    config = load_and_prepare(CONFIG)
    store = PanelStore.from_dir(config["paths"]["processed"])
    folds = ItemFolds.load(Path(config["paths"]["processed"]) / "item_folds.json")
    pc = config["prior"]
    prior, train_keys = fit_fold_prior(store, folds, FOLD, PopulationPrior(pc["country_shrinkage"], pc["laplace"]))
    role_keys = protocol_b_item_keys(folds, FOLD)
    option_table = load_option_table(Path(config["paths"]["cache"]) / "option_table.npz")

    caches = {}
    for fam in ["response_centric", "sentence", "raw_mean"]:
        caches[fam] = {
            "train": load_family(config, fam, "train", prior, train_keys, role_keys, "train", option_table),
            "val": load_family(config, fam, "validation", prior, train_keys, role_keys, "validation", option_table),
        }

    # Baselines (no alignment).
    print("=== BASELINES (decoder on raw z) ===", flush=True)
    base = {}
    for fam in ["response_centric", "sentence", "raw_mean"]:
        base[fam] = decoder_val_nll(caches[fam]["train"], caches[fam]["val"])
        print(f"  {fam:18s} val_nll={base[fam]:.4f}", flush=True)

    # Sweep grid on response_centric alignment.
    grid = {
        "temperature_init": [0.1, 0.3, 0.5],
        "lr": [3e-4, 1e-3],
        "cross_neg": [False, True],
        "weight_decay": [1e-4, 1e-3],
    }
    keys = list(grid)
    print("\n=== ALIGN SWEEP (response_centric) ===", flush=True)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        af = train_aligner(
            caches["response_centric"]["train"], caches["response_centric"]["val"],
            projection_dim=256, hidden_dim=512, dropout=0.1,
            temperature_init=cfg["temperature_init"], residual_alpha_init=0.0,
            cross_respondent_negatives=cfg["cross_neg"],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            epochs=80, patience=12, batch_size=256, seed=SEED,
        )
        at = apply_aligner(af.aligner, caches["response_centric"]["train"])
        av = apply_aligner(af.aligner, caches["response_centric"]["val"])
        dec = decoder_val_nll(at, av)
        beat_sentence = dec < base["sentence"]
        results.append((dec, af.best_validation_loss, af.best_epoch, cfg, beat_sentence))
        print(f"  t={cfg['temperature_init']} lr={cfg['lr']} xneg={cfg['cross_neg']} wd={cfg['weight_decay']} "
              f"-> dec_nll={dec:.4f} align_val={af.best_validation_loss:.3f} ep={af.best_epoch} "
              f"{'BEATS sentence' if beat_sentence else ''}", flush=True)

    results.sort(key=lambda r: r[0])
    print("\n=== TOP 5 configs by decoder val NLL ===", flush=True)
    for dec, aval, ep, cfg, beat in results[:5]:
        print(f"  dec_nll={dec:.4f} (sentence={base['sentence']:.4f}) cfg={cfg} {'BEATS' if beat else ''}", flush=True)
    Path("artifacts").mkdir(exist_ok=True)
    json.dump({"baselines": base, "best": {"dec_nll": results[0][0], "cfg": results[0][3]}},
              open("artifacts/align_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
