#!/usr/bin/env python3
"""Matched-K history selection ablation on CES 2020.

Selectors see the same auditable source-text pool and the same 10 manually
reviewed targets. Random and family-balanced selectors repeat over three history
seeds. TF-IDF, BGE, and supervised mutual-information selection use a greedy
coverage objective: each selected source item should improve the best available
alignment to at least one target, rather than repeatedly selecting near-duplicate
items for the same construct.

The MI selector reads target answers from TRAIN respondents only and is labelled
an oracle. TF-IDF and BGE read target question text but no target answers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kceiling", HERE / "run_k_ceiling.py")
kceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kceiling)


def greedy_cover(items: list[str], scores: np.ndarray, k: int) -> tuple[list[str], list[float]]:
    """Greedily maximize sum_t max_{selected s} score(s,t)."""
    scores = np.asarray(scores, float)
    if scores.shape[0] != len(items):
        raise ValueError("score rows must align with items")
    current = np.full(scores.shape[1], -np.inf)
    available = set(range(len(items)))
    selected, gains = [], []
    for _ in range(min(k, len(items))):
        best_index, best_gain = None, -np.inf
        for index in available:
            updated = np.maximum(current, scores[index])
            before = np.where(np.isfinite(current), current, 0.0).sum()
            gain = float(updated.sum() - before)
            if gain > best_gain or (gain == best_gain and items[index] < items[best_index]):
                best_index, best_gain = index, gain
        selected.append(items[best_index])
        gains.append(best_gain)
        current = np.maximum(current, scores[best_index])
        available.remove(best_index)
    return selected, gains


def random_select(items: list[str], k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(items, min(k, len(items)), replace=False).tolist())


def family_balanced_select(catalog: pd.DataFrame, k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    groups = {name: frame.item.tolist() for name, frame in catalog.groupby("family")}
    families = list(groups)
    rng.shuffle(families)
    selected = []
    # One per family first; if K exceeds family count, fill from remaining items.
    for name in families:
        selected.append(str(rng.choice(groups[name])))
        if len(selected) == k:
            return sorted(selected)
    remaining = sorted(set(catalog.item) - set(selected))
    if len(selected) < k:
        selected += rng.choice(remaining, min(k - len(selected), len(remaining)),
                               replace=False).tolist()
    return sorted(selected)


def tfidf_scores(source_text: list[str], target_text: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(source_text + target_text)
    source = matrix[: len(source_text)]
    target = matrix[len(source_text):]
    return (source @ target.T).toarray()


def bge_scores(source_text: list[str], target_text: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    vectors = model.encode(source_text + target_text, batch_size=64,
                           normalize_embeddings=True, show_progress_bar=False)
    source = vectors[: len(source_text)]
    target = vectors[len(source_text):]
    return np.asarray(source @ target.T, float)


def mi_scores(train: pd.DataFrame, sources: list[str], targets: list[str]) -> np.ndarray:
    matrix = np.zeros((len(sources), len(targets)), float)
    for i, source in enumerate(sources):
        x = train[source].fillna(-999).astype(str)
        for j, target in enumerate(targets):
            y = train[target].astype(str)
            value = mutual_info_score(x, y)
            # Normalize by target entropy so binary and five-way targets are
            # comparable in the greedy coverage objective.
            entropy = mutual_info_score(y, y)
            matrix[i, j] = value / max(entropy, 1e-12)
    return matrix


def evaluate(train: pd.DataFrame, test: pd.DataFrame, targets: list[str],
             selected: list[str], model_seed: int, history_seed: int,
             trees: int, max_depth: int, min_leaf: int) -> dict:
    features = kceiling.DEMOGRAPHICS + selected
    imputer = SimpleImputer(strategy="most_frequent")
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    y_train = train[targets].to_numpy()
    y_test = test[targets].to_numpy()
    model = ExtraTreesClassifier(
        n_estimators=trees, max_depth=max_depth, min_samples_leaf=min_leaf,
        max_features="sqrt", n_jobs=-1, random_state=model_seed,
    ).fit(x_train, y_train)
    p, classes = kceiling.predict_proba_multi(model, x_test)
    true_metrics, true_loss = kceiling.metrics(p, classes, y_test)
    correct = []
    for j, (probability, labels) in enumerate(zip(p, classes)):
        predictions = labels[np.argmax(probability, axis=1)]
        correct.append(predictions == y_test[:, j])
    correct = np.column_stack(correct)

    shuffled_source = kceiling.matched_shuffle(test, selected, history_seed + 10000)
    shuffled_frame = pd.concat(
        [test[kceiling.DEMOGRAPHICS].reset_index(drop=True),
         shuffled_source.reset_index(drop=True)], axis=1
    )
    x_shuffled = imputer.transform(shuffled_frame[features])
    shuffled_p, _ = kceiling.predict_proba_multi(model, x_shuffled)
    shuffled_metrics, shuffled_loss = kceiling.metrics(shuffled_p, classes, y_test)
    identity = kceiling.bootstrap_gap(true_loss, shuffled_loss, history_seed)
    return {
        "true_history": true_metrics,
        "matched_wrong_history": shuffled_metrics,
        "identity_gap": identity,
        "respondent_nll": true_loss.mean(axis=1).tolist(),
        "respondent_accuracy": correct.mean(axis=1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv")
    parser.add_argument("--catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--manifest", default="cross_survey/metadata/ces2020_phase2_manifest.json")
    parser.add_argument("--output", default="cross_survey/results/phase1/coverage_ablation.json")
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--k-grid", nargs="*", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    catalog = pd.read_csv(args.catalog)
    source_catalog = catalog[
        catalog.wave.eq("source_pre")
        & catalog.found
        & catalog.item_specific_text.fillna(False)
    ].copy().sort_values("item")
    if manifest.get("source_allowed_families"):
        source_catalog = source_catalog[
            source_catalog.family.isin(manifest["source_allowed_families"])
        ].copy()
    sources = source_catalog.item.tolist()
    source_text = source_catalog.proxy_text.fillna("").tolist()
    targets = manifest["target_items"]
    target_text = [manifest["target_text"][item] for item in targets]
    usecols = ["caseid", "starttime_post"] + kceiling.DEMOGRAPHICS + sources + targets
    frame = pd.read_csv(args.input, usecols=usecols, low_memory=False)
    frame = frame[frame.starttime_post.notna()].dropna(subset=targets).reset_index(drop=True)
    buckets = frame.caseid.map(lambda value: kceiling.bucket(value, manifest["seed"]))
    train = frame[buckets <= 5].copy()
    validation = frame[(buckets >= 6) & (buckets <= 7)].copy()
    test = frame[buckets >= 8].copy()
    print(f"text-audited source pool={len(sources)}; train={len(train)} val={len(validation)} test={len(test)}")

    print("building selector score matrices")
    score_matrices = {
        "tfidf_target_aware": tfidf_scores(source_text, target_text),
        "bge_target_aware": bge_scores(source_text, target_text, args.bge_model),
        "supervised_mi_oracle": mi_scores(train, sources, targets),
    }

    results = []
    for k in (args.k_grid or manifest["k_grid"]):
        plans = []
        for seed in manifest["history_seeds"]:
            plans.append(("random", seed, random_select(sources, k, seed), None))
            plans.append(("family_balanced", seed,
                          family_balanced_select(source_catalog, k, seed), None))
        for name, scores in score_matrices.items():
            selected, gains = greedy_cover(sources, scores, k)
            plans.append((name, manifest["seed"], selected, gains))

        for name, history_seed, selected, selection_gains in plans:
            evaluated = evaluate(
                train, test, targets, selected, manifest["seed"], history_seed,
                args.trees, args.max_depth, args.min_leaf,
            )
            record = {
                "selector": name, "k": k, "history_seed": history_seed,
                "selected_source_items": selected,
                "selection_gains": selection_gains,
                **evaluated,
            }
            results.append(record)
            metric = evaluated["true_history"]
            identity = evaluated["identity_gap"]
            print(
                f"{name:22s} K={k:2d} seed={history_seed}: "
                f"NLL={metric['nll']:.4f} acc={metric['accuracy']:.4f} "
                f"identity={identity['gap_nats']:+.4f}"
            )

    payload = {
        "dataset": "ces2020", "phase": "coverage_ablation",
        "post_freeze": True, "exploratory": True,
        "manifest": str(args.manifest),
        "source_pool_count": len(sources),
        "source_pool_rule": manifest["source_text_pool_rule"],
        "target_items": targets,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "bge_model": args.bge_model,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
