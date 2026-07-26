#!/usr/bin/env python3
"""Preliminary CPU ceiling: does pre-wave history identify post-wave answers?

This is deliberately non-LLM and runs only the initial K grid declared in the
split manifest. It compares a population prior, demographics, and an ExtraTrees
predictor over demographics plus K source answers. The decisive diagnostic is
true same-person history versus a demographic-stratum-matched wrong person's
history on the identical target rows.

This is a screening experiment, not the final paper result: question wording,
construct labels, ordinal status, repeated concepts, and branch logic still
require codebook adjudication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

DEMOGRAPHICS = [
    "birthyr", "gender", "educ", "race", "hispanic", "marstat", "region",
    "faminc_new",
]
EPS = 1e-12


def bucket(caseid: object, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{caseid}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 10


def choose_source(items: list[str], k: int | str, seed: int) -> list[str]:
    if k == "full" or int(k) >= len(items):
        return list(items)
    if int(k) == 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(items, int(k), replace=False).tolist())


def matched_shuffle(frame: pd.DataFrame, source: list[str], seed: int) -> pd.DataFrame:
    """Shuffle source histories within coarse demographic strata."""
    if not source:
        return frame[source].copy()
    rng = np.random.default_rng(seed)
    strata = pd.DataFrame({
        "gender": frame.gender.fillna(-1),
        "age_decade": ((2020 - frame.birthyr.fillna(2020)) // 10).clip(0, 12),
        "educ": frame.educ.fillna(-1),
    }, index=frame.index)
    out = frame[source].copy()
    for _, indices in strata.groupby(list(strata.columns)).groups.items():
        idx = np.asarray(list(indices))
        if len(idx) > 1:
            out.loc[idx, source] = frame.loc[rng.permutation(idx), source].to_numpy()
    return out


def population_probabilities(y_train: np.ndarray, y_test: np.ndarray) -> tuple[list, list]:
    probabilities, classes = [], []
    for j in range(y_train.shape[1]):
        labels = np.unique(y_train[:, j])
        counts = np.asarray([(y_train[:, j] == value).sum() for value in labels], float) + 0.5
        p = counts / counts.sum()
        probabilities.append(np.tile(p, (len(y_test), 1)))
        classes.append(labels)
    return probabilities, classes


def metrics(probabilities: list[np.ndarray], classes: list[np.ndarray],
            y: np.ndarray) -> tuple[dict, np.ndarray]:
    losses, correct = [], []
    for j, (p, labels) in enumerate(zip(probabilities, classes)):
        lookup = {value: index for index, value in enumerate(labels)}
        indices = np.asarray([lookup[value] for value in y[:, j]], int)
        losses.append(-np.log(np.maximum(p[np.arange(len(y)), indices], EPS)))
        predictions = labels[np.argmax(p, axis=1)]
        correct.append(predictions == y[:, j])
    loss = np.column_stack(losses)
    accuracy = np.column_stack(correct)
    return {
        "nll": float(loss.mean()),
        "accuracy": float(accuracy.mean()),
        "rows": int(loss.size),
        "respondents": int(len(y)),
        "targets": int(y.shape[1]),
    }, loss


def bootstrap_gap(true_loss: np.ndarray, shuffled_loss: np.ndarray,
                  seed: int, draws: int = 2000) -> dict:
    # Positive = matched-wrong history is worse, hence true identity helps.
    per_person = shuffled_loss.mean(axis=1) - true_loss.mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for draw in range(draws):
        pick = rng.integers(0, len(per_person), len(per_person))
        boot[draw] = per_person[pick].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "gap_nats": float(per_person.mean()),
        "ci": [float(lo), float(hi)],
        "respondents": int(len(per_person)),
        "draws": draws,
        "true_history_better": bool(lo > 0),
    }


def predict_proba_multi(model: ExtraTreesClassifier, x: np.ndarray) -> tuple[list, list]:
    probabilities = model.predict_proba(x)
    if not isinstance(probabilities, list):
        probabilities = [probabilities]
    classes = model.classes_
    if not isinstance(classes, list):
        classes = [classes]
    return probabilities, classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv"
    )
    parser.add_argument(
        "--manifest", default="cross_survey/metadata/ces2020_split_manifest.json"
    )
    parser.add_argument(
        "--output", default="cross_survey/results/phase1/initial_k_ceiling.json"
    )
    parser.add_argument("--trees", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument(
        "--k-grid", nargs="*", default=None,
        help="Post-initial extension only, e.g. 1 3 5 10 20 40 full",
    )
    parser.add_argument(
        "--history-seeds", nargs="*", type=int, default=None,
        help="Repeat random history selection; model seed remains manifest seed",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    source = manifest["source_items"]
    targets = manifest["target_items"]
    seed = int(manifest["seed"])
    k_grid = args.k_grid or manifest["initial_k_grid"]
    history_seeds = args.history_seeds or manifest["initial_history_seeds"]
    usecols = ["caseid", "starttime_post"] + DEMOGRAPHICS + source + targets
    frame = pd.read_csv(args.input, usecols=usecols, low_memory=False)
    frame = frame[frame.starttime_post.notna()].copy()
    frame = frame.dropna(subset=targets).reset_index(drop=True)
    buckets = frame.caseid.map(lambda value: bucket(value, seed))
    train = frame[buckets <= 5].copy()
    validation = frame[(buckets >= 6) & (buckets <= 7)].copy()
    test = frame[buckets >= 8].copy()
    # Validation remains untouched in this first ceiling; it is reserved for
    # later hyperparameter selection. Fixed model settings come from the script.
    print(f"paired complete respondents: train={len(train)} val={len(validation)} test={len(test)}")

    y_train = train[targets].to_numpy()
    y_test = test[targets].to_numpy()
    pop_p, pop_classes = population_probabilities(y_train, y_test)
    population, _ = metrics(pop_p, pop_classes, y_test)
    print(f"population: NLL={population['nll']:.4f} acc={population['accuracy']:.4f}")

    results = []
    for k_value in k_grid:
      for history_seed in (history_seeds if str(k_value) not in ("0",) else [history_seeds[0]]):
        selected = choose_source(source, k_value, history_seed)
        features = DEMOGRAPHICS + selected
        imputer = SimpleImputer(strategy="most_frequent")
        x_train = imputer.fit_transform(train[features])
        x_test = imputer.transform(test[features])
        model = ExtraTreesClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        true_p, classes = predict_proba_multi(model, x_test)
        true_metrics, true_loss = metrics(true_p, classes, y_test)

        if selected:
            shuffled_source = matched_shuffle(test, selected, history_seed + 10000)
            # Concatenate once; repeated column insertion fragments the frame
            # badly at K=full and can dominate CPU time.
            shuffled_frame = pd.concat(
                [test[DEMOGRAPHICS].reset_index(drop=True),
                 shuffled_source.reset_index(drop=True)], axis=1
            )
            x_shuffled = imputer.transform(shuffled_frame[features])
            shuffled_p, _ = predict_proba_multi(model, x_shuffled)
            shuffled_metrics, shuffled_loss = metrics(shuffled_p, classes, y_test)
            identity = bootstrap_gap(true_loss, shuffled_loss, history_seed)
        else:
            shuffled_metrics = true_metrics.copy()
            identity = {
                "gap_nats": 0.0, "ci": [0.0, 0.0],
                "respondents": len(test), "draws": 0,
                "true_history_better": False,
            }
        record = {
            "k": k_value,
            "history_seed": history_seed,
            "selected_source_items": selected,
            "model": "ExtraTreesClassifier",
            "true_history": true_metrics,
            "matched_wrong_history": shuffled_metrics,
            "identity_gap": identity,
        }
        results.append(record)
        print(
            f"K={str(k_value):>4s} seed={history_seed}: NLL={true_metrics['nll']:.4f} "
            f"acc={true_metrics['accuracy']:.4f} "
            f"identity_gap={identity['gap_nats']:+.4f} "
            f"CI=[{identity['ci'][0]:+.4f},{identity['ci'][1]:+.4f}]"
        )

    payload = {
        "dataset": "ces2020",
        "post_freeze": True,
        "preliminary": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": seed,
        "history_seeds": history_seeds,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "target_items": targets,
        "population": population,
        "results": results,
        "caveat": (
            "Preliminary coding-only ceiling. Targets are not yet adjudicated "
            "for wording, ordinal status, repeated concepts, or branch logic."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
