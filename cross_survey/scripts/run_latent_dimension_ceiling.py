#!/usr/bin/env python3
"""Does cross-construct transfer require a multidimensional respondent profile?

Source answers are one-hot encoded and a TruncatedSVD basis is fitted on TRAIN
respondents only. The same fixed basis is used for test and matched-wrong-person
histories. ExtraTrees predicts the ten post-wave targets from demographics plus
the first d respondent factors for d in {0,1,2,4,8,16,24}.

This is a preliminary low-rank ceiling, not the final ordinal IRT model. In
particular, d=1 is a generic dominant behavioral/political axis, not a formally
identified response-style intercept. The question is narrower: does adding
dimensions beyond the first improve held-out same-person prediction?
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kceiling", HERE / "run_k_ceiling.py")
kceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kceiling)


def respondent_arrays(probabilities, classes, y):
    losses, correct = [], []
    for j, (p, labels) in enumerate(zip(probabilities, classes)):
        lookup = {value: index for index, value in enumerate(labels)}
        indices = np.asarray([lookup[value] for value in y[:, j]], int)
        losses.append(-np.log(np.maximum(p[np.arange(len(y)), indices], 1e-12)))
        correct.append(labels[np.argmax(p, axis=1)] == y[:, j])
    return np.column_stack(losses).mean(axis=1), np.column_stack(correct).mean(axis=1)


def bootstrap_difference(a, b, seed=1701, draws=5000):
    difference = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for draw in range(draws):
        pick = rng.integers(0, len(difference), len(difference))
        boot[draw] = difference[pick].mean()
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"difference": float(difference.mean()),
            "ci": [float(low), float(high)], "draws": draws}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv")
    parser.add_argument("--catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--manifest", default="cross_survey/metadata/ces2020_cross_construct_manifest.json")
    parser.add_argument("--output", default="cross_survey/results/phase1/latent_dimension_ceiling.json")
    parser.add_argument("--dimensions", nargs="*", type=int,
                        default=[0, 1, 2, 4, 8, 16, 24])
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    catalog = pd.read_csv(args.catalog)
    source_frame = catalog[
        catalog.wave.eq("source_pre") & catalog.found
        & catalog.item_specific_text.fillna(False)
        & catalog.family.isin(manifest["source_allowed_families"])
    ].copy()
    sources = source_frame.item.tolist()
    targets = manifest["target_items"]
    columns = ["caseid", "starttime_post"] + kceiling.DEMOGRAPHICS + sources + targets
    frame = pd.read_csv(args.input, usecols=columns, low_memory=False)
    frame = frame[frame.starttime_post.notna()].dropna(subset=targets).reset_index(drop=True)
    buckets = frame.caseid.map(lambda x: kceiling.bucket(x, manifest["seed"]))
    train = frame[buckets <= 5].copy()
    validation = frame[(buckets >= 6) & (buckets <= 7)].copy()
    test = frame[buckets >= 8].copy()
    y_train, y_test = train[targets].to_numpy(), test[targets].to_numpy()
    print(f"cross-construct sources={len(sources)} train={len(train)} val={len(validation)} test={len(test)}")

    # Categorical source encoder. Missingness is itself a category because CES
    # branching may carry information, but target missingness remains excluded.
    source_encoder = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), sources),
    ], remainder="drop")
    source_train = source_encoder.fit_transform(train[sources])
    source_test = source_encoder.transform(test[sources])
    shuffled_source = kceiling.matched_shuffle(test, sources, manifest["seed"] + 10000)
    source_shuffled = source_encoder.transform(shuffled_source[sources])

    max_dimension = max(args.dimensions)
    if max_dimension >= min(source_train.shape):
        max_dimension = min(source_train.shape) - 1
    svd = TruncatedSVD(n_components=max_dimension, n_iter=7,
                       random_state=manifest["seed"])
    z_train = svd.fit_transform(source_train)
    z_test = svd.transform(source_test)
    z_shuffled = svd.transform(source_shuffled)

    demo_imputer = SimpleImputer(strategy="most_frequent")
    demo_train = demo_imputer.fit_transform(train[kceiling.DEMOGRAPHICS])
    demo_test = demo_imputer.transform(test[kceiling.DEMOGRAPHICS])
    results = []
    arrays = {}
    for dimension in args.dimensions:
        x_train = demo_train if dimension == 0 else np.column_stack(
            [demo_train, z_train[:, :dimension]])
        x_test = demo_test if dimension == 0 else np.column_stack(
            [demo_test, z_test[:, :dimension]])
        x_shuffled = demo_test if dimension == 0 else np.column_stack(
            [demo_test, z_shuffled[:, :dimension]])
        model = ExtraTreesClassifier(
            n_estimators=args.trees, max_depth=16, min_samples_leaf=20,
            max_features="sqrt", n_jobs=-1, random_state=manifest["seed"],
        ).fit(x_train, y_train)
        p, classes = kceiling.predict_proba_multi(model, x_test)
        metric, loss = kceiling.metrics(p, classes, y_test)
        respondent_nll, respondent_accuracy = respondent_arrays(p, classes, y_test)
        p_wrong, _ = kceiling.predict_proba_multi(model, x_shuffled)
        wrong_metric, wrong_loss = kceiling.metrics(p_wrong, classes, y_test)
        identity = kceiling.bootstrap_gap(loss, wrong_loss, manifest["seed"])
        arrays[dimension] = (respondent_nll, respondent_accuracy)
        record = {
            "dimension": dimension,
            "true_history": metric,
            "matched_wrong_history": wrong_metric,
            "identity_gap": identity,
            "svd_explained_variance_cumulative": (
                float(svd.explained_variance_ratio_[:dimension].sum())
                if dimension else 0.0
            ),
        }
        results.append(record)
        print(f"d={dimension:2d}: NLL={metric['nll']:.4f} acc={metric['accuracy']:.4f} "
              f"identity={identity['gap_nats']:+.4f}")

    comparisons = {}
    if 1 in arrays:
        for dimension in args.dimensions:
            if dimension <= 1:
                continue
            comparisons[f"d{dimension}_vs_d1"] = {
                "nll": bootstrap_difference(arrays[dimension][0], arrays[1][0],
                                             manifest["seed"]),
                "accuracy": bootstrap_difference(arrays[dimension][1], arrays[1][1],
                                                  manifest["seed"]),
            }
    payload = {
        "dataset": "ces2020", "phase": "latent_dimension_ceiling",
        "post_freeze": True, "exploratory": True,
        "source_items": sources, "target_items": targets,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "onehot_features": int(source_train.shape[1]),
        "results": results, "comparisons": comparisons,
        "caveat": "TruncatedSVD factors are preliminary, not identified ordinal IRT traits.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
