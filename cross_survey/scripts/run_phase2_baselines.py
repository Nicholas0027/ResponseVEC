#!/usr/bin/env python3
"""Population and demographics baselines for the CES Phase 2 target set."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kceiling", HERE / "run_k_ceiling.py")
kceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kceiling)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv")
    parser.add_argument("--manifest", default="cross_survey/metadata/ces2020_phase2_manifest.json")
    parser.add_argument("--catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--output", default="cross_survey/results/phase1/phase2_baselines.json")
    parser.add_argument("--trees", type=int, default=80)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    targets = manifest["target_items"]
    catalog = pd.read_csv(args.catalog)
    sources = catalog[
        catalog.wave.eq("source_pre") & catalog.found
        & catalog.item_specific_text.fillna(False)
    ].copy()
    if manifest.get("source_allowed_families"):
        sources = sources[sources.family.isin(manifest["source_allowed_families"])]
    sources = sources.item.tolist()
    columns = ["caseid", "starttime_post"] + kceiling.DEMOGRAPHICS + sources + targets
    frame = pd.read_csv(args.input, usecols=columns, low_memory=False)
    frame = frame[frame.starttime_post.notna()].dropna(subset=targets).reset_index(drop=True)
    buckets = frame.caseid.map(lambda x: kceiling.bucket(x, manifest["seed"]))
    train, test = frame[buckets <= 5].copy(), frame[buckets >= 8].copy()
    y_train, y_test = train[targets].to_numpy(), test[targets].to_numpy()

    p, classes = kceiling.population_probabilities(y_train, y_test)
    population, _ = kceiling.metrics(p, classes, y_test)
    imputer = SimpleImputer(strategy="most_frequent")
    x_train = imputer.fit_transform(train[kceiling.DEMOGRAPHICS])
    x_test = imputer.transform(test[kceiling.DEMOGRAPHICS])
    model = ExtraTreesClassifier(
        n_estimators=args.trees, max_depth=16, min_samples_leaf=20,
        max_features="sqrt", n_jobs=-1, random_state=manifest["seed"],
    ).fit(x_train, y_train)
    p, classes = kceiling.predict_proba_multi(model, x_test)
    demographics, _ = kceiling.metrics(p, classes, y_test)

    full_features = kceiling.DEMOGRAPHICS + sources
    full_imputer = SimpleImputer(strategy="most_frequent")
    full_train = full_imputer.fit_transform(train[full_features])
    full_test = full_imputer.transform(test[full_features])
    full_model = ExtraTreesClassifier(
        n_estimators=args.trees, max_depth=16, min_samples_leaf=20,
        max_features="sqrt", n_jobs=-1, random_state=manifest["seed"],
    ).fit(full_train, y_train)
    p, classes = kceiling.predict_proba_multi(full_model, full_test)
    full_history, full_loss = kceiling.metrics(p, classes, y_test)
    shuffled = kceiling.matched_shuffle(test, sources, manifest["seed"] + 10000)
    shuffled_frame = pd.concat(
        [test[kceiling.DEMOGRAPHICS].reset_index(drop=True),
         shuffled.reset_index(drop=True)], axis=1
    )
    p_shuffled, _ = kceiling.predict_proba_multi(
        full_model, full_imputer.transform(shuffled_frame[full_features])
    )
    shuffled_metrics, shuffled_loss = kceiling.metrics(p_shuffled, classes, y_test)
    full_identity = kceiling.bootstrap_gap(full_loss, shuffled_loss, manifest["seed"])
    payload = {
        "dataset": "ces2020", "phase": "coverage_ablation_baselines",
        "post_freeze": True, "targets": targets,
        "split_counts": {"train": len(train), "test": len(test)},
        "population": population, "demographics_extratrees": demographics,
        "full_text_audited_history": full_history,
        "full_matched_wrong_history": shuffled_metrics,
        "full_identity_gap": full_identity,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
