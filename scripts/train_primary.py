#!/usr/bin/env python
"""Train all primary heads for one Protocol-B outer fold and K.

Resumable per-method: if predictions_all_replicates.parquet from a prior run
of this exact (fold, k) already contains every decoder-seed x option-seed row
for a given base method, that method is loaded from disk (predictions +
per-seed best_validation_nll from its saved fit.json) instead of retrained.
This means adding a new family (e.g. --include-respondent-vec, or a config
change that adds "sentence" to representation_families) only trains the NEW
method on a rerun -- it does not re-train already-completed methods and burn
GPU/CPU time repeating work that has already produced a result. Pass
--force-retrain to disable this and retrain everything from scratch.
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
from responsevec.direct import DirectCalibrator, nll
from responsevec.encode import load_option_table
from responsevec.pipeline import (
    fit_fold_prior,
    protocol_b_item_keys,
    respondentvec_cache_directory,
    shared_cache_directory,
)
from responsevec.prior import PopulationPrior
from responsevec.protocols import ItemFolds
from responsevec.training import (
    apply_aligner,
    arrays_from_cache,
    arrays_from_respondent_cache,
    average_decoder_seeds,
    average_option_seeds,
    predict_head,
    prediction_frame,
    save_aligner_fit,
    save_head_fit,
    train_aligner,
    train_direct_head,
    train_option_head,
    subset_arrays,
)
from responsevec.utils import write_json


def load_arrays(config, fold, role, split, k, option_seed, family, option_table, prior, train_keys, role_keys):
    directory = shared_cache_directory(
        config["paths"]["cache"], respondent_split=split, k=k,
        option_seed=option_seed, family=family,
    )
    cache = RepresentationCache.load(directory)
    shared = arrays_from_cache(
        cache, option_table, prior, train_keys,
        max_options=int(config["decoder"].get("max_options", 11)),
    )
    return subset_arrays(shared, role_keys[role])


def load_existing_predictions(output_dir: Path) -> pd.DataFrame | None:
    path = output_dir / "predictions_all_replicates.parquet"
    return pd.read_parquet(path) if path.exists() else None


def method_is_complete(existing: pd.DataFrame | None, methods, option_seeds) -> bool:
    """`methods` names (e.g. ["direct_calibrated_seed1701", ...] or the
    seedless ["direct_raw"]) are complete for this run's option-seed request
    iff every one of them has a row for every requested option seed.
    Requesting MORE option seeds than a prior run covered forces a retrain
    (nothing is silently short-changed)."""
    if existing is None:
        return False
    for method in methods:
        rows = existing[existing["method"].eq(method)]
        if rows.empty or set(rows["option_seed"].astype(int)) < set(int(s) for s in option_seeds):
            return False
    return True


def load_saved_validation_nll(heads_dir: Path, fold: int, k: int, method: str, seed: int) -> float:
    fit_path = heads_dir / f"fold_{fold:02d}" / f"k_{k}" / f"{method}_seed{seed}" / "fit.json"
    if not fit_path.exists():
        raise FileNotFoundError(
            f"predictions for {method}_seed{seed} exist but {fit_path} is missing -- "
            "cannot resume without the saved validation NLL; pass --force-retrain."
        )
    return float(json.loads(fit_path.read_text())["best_validation_nll"])


def load_respondent_arrays(config, split, k):
    """RespondentVec (query-independent, design §2.3.D): one z per
    (panel_id, domain), broadcast onto input_centric's already-correct
    target-row arrays for the identical rows (see arrays_from_respondent_cache).
    Loaded once per split (not per fold/role) since RespondentVec's cache has
    no option_seed axis and is fold-independent, matching every other
    shared Protocol-B cache."""
    target_directory = shared_cache_directory(
        config["paths"]["cache"], respondent_split=split, k=k, option_seed=0, family="input_centric",
    )
    respondent_directory = respondentvec_cache_directory(config["paths"]["cache"], respondent_split=split, k=k)
    return target_directory, respondent_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--option-seeds", default="0", help="comma-separated cached prompt permutations")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override for smoke/debug")
    parser.add_argument("--seeds", default=None, help="override decoder seeds, comma-separated")
    parser.add_argument(
        "--include-respondent-vec", action="store_true",
        help="also train/evaluate the query-independent RespondentVec ablation (design section 2.3.D); "
             "requires scripts/extract_respondentvec.py to have been run for this split and K",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="retrain every method from scratch even if predictions_all_replicates.parquet "
             "already has complete results for it (disables the default resume-by-method behavior)",
    )
    parser.add_argument(
        "--align-families", default="",
        help="comma-separated representation families to ALSO train a task-aligned "
             "variant for (ResponseVec-Align, design section 2.5). For each name X in this "
             "list an extra method 'X_aligned' is produced: a frozen-encoder "
             "option-anchored supervised-contrastive projection g_phi is fit on X's "
             "train/val cache, then the SAME option-aware decoder is trained on g_phi(z). "
             "Applying it to raw_mean/sentence too keeps the capacity-matched fairness "
             "argument (e.g. --align-families response_centric,raw_mean,sentence).",
    )
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    option_seeds = [int(value) for value in args.option_seeds.split(",") if value.strip()]
    decoder_seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds else [int(value) for value in config["decoder"]["seeds"]]
    )
    store = PanelStore.from_dir(config["paths"]["processed"])
    folds = ItemFolds.load(Path(config["paths"]["processed"]) / "item_folds.json")
    prior_config = config["prior"]
    prior, train_keys = fit_fold_prior(
        store, folds, args.fold,
        PopulationPrior(prior_config["country_shrinkage"], prior_config["laplace"]),
    )
    fold_prior_dir = Path(config["paths"]["heads"]) / f"fold_{args.fold:02d}" / "prior"
    prior.save(fold_prior_dir)
    role_keys = protocol_b_item_keys(folds, args.fold)
    option_table = load_option_table(Path(config["paths"]["cache"]) / "option_table.npz")
    heads_dir = Path(config["paths"]["heads"])
    output_dir = Path(config["paths"]["predictions"]) / f"fold_{args.fold:02d}" / f"k_{args.k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = None if args.force_retrain else load_existing_predictions(output_dir)
    primary = config["decoder"]["primary"]
    common = dict(
        projection_dim=int(config["decoder"]["projection_dim"]),
        dropout=float(config["decoder"]["dropout"]),
        prior_eta=float(primary["prior_eta"]), learnable_eta=False,
        lr=float(primary["lr"]), weight_decay=float(primary["weight_decay"]),
        epochs=int(args.epochs or config["decoder"]["epochs"]),
        patience=int(config["decoder"]["early_stopping_patience"]),
        batch_size=int(config["decoder"]["batch_size"]),
        gradient_clip=float(config["decoder"]["gradient_clip"]),
        device=args.device,
    )

    # Heads are trained on the canonical order only. All cached permutations
    # are evaluated and averaged after semantic remapping.
    train_causal = load_arrays(config, args.fold, "train", "train", args.k, 0, "causal_final", option_table, prior, train_keys, role_keys)
    val_causal = load_arrays(config, args.fold, "validation", "validation", args.k, 0, "causal_final", option_table, prior, train_keys, role_keys)
    scalar = DirectCalibrator().fit(
        val_causal.direct_probabilities, val_causal.log_prior,
        val_causal.option_mask, val_causal.targets,
        max_options=val_causal.option_mask.shape[1],
    )
    raw_validation_nll = nll(val_causal.direct_probabilities, val_causal.targets)
    scalar_validation_probability = scalar.predict(
        val_causal.direct_probabilities, val_causal.log_prior, val_causal.option_mask
    )
    scalar_validation_nll = nll(scalar_validation_probability, val_causal.targets)

    all_predictions: list[pd.DataFrame] = []
    direct_calibrated_nlls: list[float] = []
    if method_is_complete(existing, [f"direct_calibrated_seed{s}" for s in decoder_seeds], option_seeds):
        print("[skip] direct_calibrated already complete for this fold/k -- reusing cached predictions")
        for seed in decoder_seeds:
            direct_calibrated_nlls.append(load_saved_validation_nll(heads_dir, args.fold, args.k, "direct_calibrated", seed))
        all_predictions.append(existing[existing["method"].str.startswith("direct_calibrated_seed")])
    else:
        for seed in decoder_seeds:
            fit = train_direct_head(
                train_causal, val_causal, temperature_init=1.0, seed=seed, **common
            )
            direct_calibrated_nlls.append(fit.best_validation_nll)
            save_head_fit(
                fit,
                heads_dir / f"fold_{args.fold:02d}" / f"k_{args.k}" / f"direct_calibrated_seed{seed}",
                {"family": "direct_calibrated", "seed": seed, "fold": args.fold, "k": args.k},
            )
            for option_seed in option_seeds:
                test = load_arrays(config, args.fold, "test", "test", args.k, option_seed, "causal_final", option_table, prior, train_keys, role_keys)
                probability = predict_head(fit.model, test, device=args.device, direct=True)
                all_predictions.append(prediction_frame(test, probability, f"direct_calibrated_seed{seed}"))

    # Raw and scalar controls need no decoder seed, so "complete" just means the
    # option-seed rows already exist.
    if method_is_complete(existing, ["direct_raw", "direct_scalar"], option_seeds):
        print("[skip] direct_raw/direct_scalar already complete for this fold/k -- reusing cached predictions")
        all_predictions.append(existing[existing["method"].isin(["direct_raw", "direct_scalar"])])
    else:
        for option_seed in option_seeds:
            test = load_arrays(config, args.fold, "test", "test", args.k, option_seed, "causal_final", option_table, prior, train_keys, role_keys)
            all_predictions.append(prediction_frame(test, test.direct_probabilities, "direct_raw"))
            scalar_probability = scalar.predict(test.direct_probabilities, test.log_prior, test.option_mask)
            all_predictions.append(prediction_frame(test, scalar_probability, "direct_scalar"))

    representation_families = {
        "raw_final": "causal_final",
        "raw_mean": "raw_mean",
        "input_centric": "input_centric",
        "response_centric": "response_centric",
        # Conventional text embedder (BGE) baseline, mirroring LLMGeovec's
        # Bert-whitening/GTE controls: does using the TASK model's own hidden
        # states beat a generic off-the-shelf sentence encoder on the same
        # prompt, or does the LLM add nothing over any embedder?
        "sentence": "sentence",
    }
    representation_validation: dict[str, list[float]] = {}
    for method, cache_family in representation_families.items():
        if method_is_complete(existing, [f"{method}_seed{s}" for s in decoder_seeds], option_seeds):
            print(f"[skip] {method} already complete for this fold/k -- reusing cached predictions")
            representation_validation[method] = [
                load_saved_validation_nll(heads_dir, args.fold, args.k, method, seed) for seed in decoder_seeds
            ]
            all_predictions.append(existing[existing["method"].str.startswith(f"{method}_seed")])
            continue
        train = load_arrays(config, args.fold, "train", "train", args.k, 0, cache_family, option_table, prior, train_keys, role_keys)
        validation = load_arrays(config, args.fold, "validation", "validation", args.k, 0, cache_family, option_table, prior, train_keys, role_keys)
        representation_validation[method] = []
        for seed in decoder_seeds:
            fit = train_option_head(
                train, validation, temperature_init=float(primary["temperature_init"]),
                rps_lambda=float(primary["rps_lambda"]), seed=seed, **common,
            )
            representation_validation[method].append(fit.best_validation_nll)
            save_head_fit(
                fit,
                heads_dir / f"fold_{args.fold:02d}" / f"k_{args.k}" / f"{method}_seed{seed}",
                {"family": method, "seed": seed, "fold": args.fold, "k": args.k},
            )
            for option_seed in option_seeds:
                test = load_arrays(config, args.fold, "test", "test", args.k, option_seed, cache_family, option_table, prior, train_keys, role_keys)
                probability = predict_head(fit.model, test, device=args.device)
                all_predictions.append(prediction_frame(test, probability, f"{method}_seed{seed}"))

    # ------------------------------------------------------------------ #
    # ResponseVec-Align (design section 2.5): task-aligned projection.     #
    # For each requested family X, fit a frozen-encoder contrastive        #
    # projection g_phi on X's train/val cache, project train/val/test z,   #
    # then train the SAME option-aware decoder on g_phi(z) -> 'X_aligned'. #
    # ------------------------------------------------------------------ #
    align_families = [name.strip() for name in args.align_families.split(",") if name.strip()]
    align_config = config.get("align", {}) if isinstance(config, dict) else {}
    for family in align_families:
        method = f"{family}_aligned"
        cache_family = representation_families.get(family, family)
        if method_is_complete(existing, [f"{method}_seed{s}" for s in decoder_seeds], option_seeds):
            print(f"[skip] {method} already complete for this fold/k -- reusing cached predictions")
            representation_validation[method] = [
                load_saved_validation_nll(heads_dir, args.fold, args.k, method, seed) for seed in decoder_seeds
            ]
            all_predictions.append(existing[existing["method"].str.startswith(f"{method}_seed")])
            continue
        base_train = load_arrays(config, args.fold, "train", "train", args.k, 0, cache_family, option_table, prior, train_keys, role_keys)
        base_validation = load_arrays(config, args.fold, "validation", "validation", args.k, 0, cache_family, option_table, prior, train_keys, role_keys)
        representation_validation[method] = []
        for seed in decoder_seeds:
            # 1) Fit the task-alignment projection on frozen z (encoder untouched).
            aligner_fit = train_aligner(
                base_train, base_validation,
                projection_dim=int(align_config.get("projection_dim", config["decoder"]["projection_dim"])),
                hidden_dim=int(align_config.get("hidden_dim", 512)),
                dropout=float(align_config.get("dropout", config["decoder"]["dropout"])),
                temperature_init=float(align_config.get("temperature_init", 0.07)),
                residual_alpha_init=float(align_config.get("residual_alpha_init", 0.0)),
                cross_respondent_negatives=bool(align_config.get("cross_respondent_negatives", True)),
                lr=float(align_config.get("lr", 1e-3)),
                weight_decay=float(align_config.get("weight_decay", 1e-5)),
                epochs=int(args.epochs or align_config.get("epochs", 60)),
                patience=int(align_config.get("early_stopping_patience", 8)),
                batch_size=int(align_config.get("batch_size", config["decoder"]["batch_size"])),
                seed=seed, device=args.device,
            )
            save_aligner_fit(
                aligner_fit,
                heads_dir / f"fold_{args.fold:02d}" / f"k_{args.k}" / f"{method}_seed{seed}",
                {"family": method, "base_family": family, "seed": seed, "fold": args.fold, "k": args.k},
            )
            # 2) Project train/val z, then train the standard option-aware decoder.
            aligned_train = apply_aligner(aligner_fit.aligner, base_train, device=args.device)
            aligned_validation = apply_aligner(aligner_fit.aligner, base_validation, device=args.device)
            fit = train_option_head(
                aligned_train, aligned_validation, temperature_init=float(primary["temperature_init"]),
                rps_lambda=float(primary["rps_lambda"]), seed=seed, **common,
            )
            representation_validation[method].append(fit.best_validation_nll)
            save_head_fit(
                fit,
                heads_dir / f"fold_{args.fold:02d}" / f"k_{args.k}" / f"{method}_seed{seed}",
                {"family": method, "base_family": family, "seed": seed, "fold": args.fold, "k": args.k},
            )
            # 3) Evaluate: project each option-permuted test cache with the SAME
            #    aligner, then decode. Identical option-seed averaging as every
            #    query-conditioned family.
            for option_seed in option_seeds:
                base_test = load_arrays(config, args.fold, "test", "test", args.k, option_seed, cache_family, option_table, prior, train_keys, role_keys)
                aligned_test = apply_aligner(aligner_fit.aligner, base_test, device=args.device)
                probability = predict_head(fit.model, aligned_test, device=args.device)
                all_predictions.append(prediction_frame(aligned_test, probability, f"{method}_seed{seed}"))

    if args.include_respondent_vec:
        # RespondentVec (query-independent, design §2.3.D): reuse input_centric's
        # already-correct target-row arrays (option_matrix/log_prior/targets are
        # identical rows, same fold/k) and replace z by joining the respondent-
        # level cache on (panel_id, domain). option_seed=0 only: there is no
        # target here to permute, so RespondentVec is never option-seed-averaged.
        if method_is_complete(existing, [f"respondent_vec_seed{s}" for s in decoder_seeds], [0]):
            print("[skip] respondent_vec already complete for this fold/k -- reusing cached predictions")
            representation_validation["respondent_vec"] = [
                load_saved_validation_nll(heads_dir, args.fold, args.k, "respondent_vec", seed) for seed in decoder_seeds
            ]
            all_predictions.append(existing[existing["method"].str.startswith("respondent_vec_seed")])
        else:
            def _respondent_vec_arrays(role, split):
                target_dir, respondent_dir = load_respondent_arrays(config, split, args.k)
                target = arrays_from_cache(
                    RepresentationCache.load(target_dir), option_table, prior, train_keys,
                    max_options=int(config["decoder"].get("max_options", 11)),
                )
                target = subset_arrays(target, role_keys[role])
                return arrays_from_respondent_cache(target, RepresentationCache.load(respondent_dir))

            rv_train = _respondent_vec_arrays("train", "train")
            rv_validation = _respondent_vec_arrays("validation", "validation")
            rv_test = _respondent_vec_arrays("test", "test")

            representation_validation["respondent_vec"] = []
            for seed in decoder_seeds:
                fit = train_option_head(
                    rv_train, rv_validation, temperature_init=float(primary["temperature_init"]),
                    rps_lambda=float(primary["rps_lambda"]), seed=seed, **common,
                )
                representation_validation["respondent_vec"].append(fit.best_validation_nll)
                save_head_fit(
                    fit,
                    heads_dir / f"fold_{args.fold:02d}" / f"k_{args.k}" / f"respondent_vec_seed{seed}",
                    {"family": "respondent_vec", "seed": seed, "fold": args.fold, "k": args.k},
                )
                probability = predict_head(fit.model, rv_test, device=args.device)
                all_predictions.append(prediction_frame(rv_test, probability, f"respondent_vec_seed{seed}"))

    raw_predictions = pd.concat(all_predictions, ignore_index=True)
    raw_predictions.to_parquet(output_dir / "predictions_all_replicates.parquet", index=False)
    option_averaged = average_option_seeds(raw_predictions)
    seed_averaged = average_decoder_seeds(option_averaged)

    direct_validation = {
        "direct_raw": float(raw_validation_nll),
        "direct_scalar": float(scalar_validation_nll),
        "direct_calibrated": float(np.mean(direct_calibrated_nlls)),
    }
    selected = min(direct_validation, key=direct_validation.get)
    selected_rows = seed_averaged[seed_averaged["method"].eq(selected)].copy()
    selected_rows["method"] = "direct_selected"
    seed_averaged = pd.concat([seed_averaged, selected_rows], ignore_index=True)
    raw_validation = {
        method: float(np.mean(representation_validation[method]))
        for method in ("raw_final", "raw_mean")
    }
    selected_raw = min(raw_validation, key=raw_validation.get)
    raw_selected_rows = seed_averaged[seed_averaged["method"].eq(selected_raw)].copy()
    raw_selected_rows["method"] = "raw_selected"
    seed_averaged = pd.concat([seed_averaged, raw_selected_rows], ignore_index=True)
    seed_averaged.to_parquet(output_dir / "predictions_seed_averaged.parquet", index=False)
    write_json(output_dir / "direct_selection.json", {
        "selected": selected, "validation_nll": direct_validation,
        "selection_used_test_labels": False,
    })
    write_json(output_dir / "raw_selection.json", {
        "selected": selected_raw, "validation_nll": raw_validation,
        "selection_used_test_labels": False,
    })
    print({"output": str(output_dir), "selected_direct": selected, "selected_raw": selected_raw, "rows": len(seed_averaged)})


if __name__ == "__main__":
    main()
