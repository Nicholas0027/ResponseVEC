#!/usr/bin/env python3
"""Map unseen target option text to loadings in a 2D respondent space.

For each target item, a warm OVR logistic model is fit on TRAIN respondents to
obtain one coefficient vector (intercept + two respondent slopes) per option.
Family-disjoint folds then learn to predict those coefficient vectors from
option metadata, TF-IDF text, or frozen BGE text. Held-out family coefficients
are never used by deployable arms. Test respondents are used only for scoring.

This is the first true cold-target experiment in the cross-survey programme.
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kceiling", HERE / "run_k_ceiling.py")
kceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kceiling)

EPS = 1e-12


def fit_option_coefficients(z: np.ndarray, y: pd.Series,
                            codes: list[int]) -> np.ndarray:
    mask = y.notna().to_numpy()
    z_fit = z[mask]
    values = y[mask].astype(int).to_numpy()
    coefficients = []
    for code in codes:
        binary = (values == code).astype(int)
        if binary.min() == binary.max():
            prevalence = float(np.clip(binary.mean(), 1e-5, 1 - 1e-5))
            coefficients.append([np.log(prevalence / (1 - prevalence)), 0.0, 0.0])
            continue
        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        model.fit(z_fit, binary)
        coefficients.append([float(model.intercept_[0]),
                             float(model.coef_[0, 0]),
                             float(model.coef_[0, 1])])
    return np.asarray(coefficients, float)


def position_features(frame: pd.DataFrame) -> np.ndarray:
    n = frame.n_options.to_numpy(float)
    position = frame.option_position.to_numpy(float)
    normalized = np.divide(position, np.maximum(n - 1, 1))
    return np.column_stack([
        np.ones(len(frame)), normalized, normalized ** 2,
        1.0 / n, position == 0, position == (n - 1),
        n == 2, n == 4, n == 5, n >= 6,
    ]).astype(float)


def group_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                alphas=(0.01, 0.1, 1.0, 10.0, 100.0)) -> tuple[Ridge, float, dict]:
    unique = sorted(set(groups.tolist()))
    scores = {}
    for alpha in alphas:
        losses = []
        for held in unique:
            train = groups != held
            test = ~train
            if train.sum() < y.shape[1] + 2:
                continue
            model = Ridge(alpha=alpha).fit(x[train], y[train])
            losses.append(float(np.mean((model.predict(x[test]) - y[test]) ** 2)))
        scores[str(alpha)] = float(np.mean(losses)) if losses else float("inf")
    best = min(alphas, key=lambda a: scores[str(a)])
    return Ridge(alpha=best).fit(x, y), float(best), scores


def predictions_from_coefficients(z: np.ndarray, option_frame: pd.DataFrame,
                                  coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = option_frame.option_code.astype(int).to_numpy()
    logits = coefficients[:, 0][None, :] + z @ coefficients[:, 1:].T
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, codes


def append_scores(records: list[dict], method: str, item: str, family: str,
                  respondents: np.ndarray, answers: np.ndarray,
                  probabilities: np.ndarray, codes: np.ndarray) -> None:
    lookup = {code: index for index, code in enumerate(codes)}
    for panel, answer, probability in zip(respondents, answers, probabilities):
        if int(answer) not in lookup:
            continue
        index = lookup[int(answer)]
        records.append({
            "method": method, "item": item, "family": family,
            "panel_id": str(panel),
            "nll": float(-np.log(max(probability[index], EPS))),
            "correct": int(codes[int(np.argmax(probability))] == int(answer)),
        })


def clustered_comparison(frame: pd.DataFrame, left: str, right: str,
                         unit: str, seed: int, draws=5000) -> dict:
    key = ["panel_id", "item"]
    a = frame[frame.method.eq(left)][key + ["nll", "correct"]]
    b = frame[frame.method.eq(right)][key + ["nll", "correct"]]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    cluster_col = "panel_id" if unit == "respondent" else "item"
    # Build cluster means explicitly; pandas named aggregation cannot express
    # paired differences without temporary columns.
    merged["nll_gap"] = merged.nll_a - merged.nll_b
    merged["acc_gap"] = merged.correct_a - merged.correct_b
    grouped = list(merged.groupby(cluster_col))
    clusters = np.asarray([g[["nll_gap", "acc_gap"]].mean().to_numpy()
                           for _, g in grouped])
    sizes = np.asarray([len(g) for _, g in grouped], float)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, 2))
    for draw in range(draws):
        pick = rng.integers(0, len(clusters), len(clusters))
        boot[draw] = np.average(clusters[pick], axis=0, weights=sizes[pick])
    return {
        "left_minus_right_nll": float(merged.nll_gap.mean()),
        "nll_ci": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
        "left_minus_right_accuracy": float(merged.acc_gap.mean()),
        "accuracy_ci": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
        "clusters": int(len(clusters)), "unit": unit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv")
    parser.add_argument("--source-catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--source-manifest", default="cross_survey/metadata/ces2020_cross_construct_manifest.json")
    parser.add_argument("--options", default="cross_survey/metadata/ces2020_all_item_options.csv")
    parser.add_argument("--folds", default="cross_survey/metadata/ces2020_cold_item_folds.json")
    parser.add_argument("--output", default="cross_survey/results/phase1/cold_item_loading.json")
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    source_manifest = json.loads(Path(args.source_manifest).read_text())
    fold_manifest = json.loads(Path(args.folds).read_text())
    source_catalog = pd.read_csv(args.source_catalog)
    sources = source_catalog[
        source_catalog.wave.eq("source_pre") & source_catalog.found
        & source_catalog.item_specific_text.fillna(False)
        & source_catalog.family.isin(source_manifest["source_allowed_families"])
    ].item.tolist()
    options = pd.read_csv(args.options)
    items = fold_manifest["retained_items"]
    # Extra pre-wave items not used to infer z provide supervised examples of
    # how question/option text maps to respondent slopes. They enlarge the item
    # bank without leaking any post-wave target answer.
    loading_source_items = options[
        options.wave.eq("source_pre") & ~options.item.isin(sources)
    ].item.drop_duplicates().tolist()
    option_bank_items = loading_source_items + items
    options = options[options.item.isin(option_bank_items)].copy()
    columns = ["caseid", "starttime_post"] + sorted(
        set(sources + loading_source_items + items)
    )
    frame = pd.read_csv(args.input, usecols=columns, low_memory=False)
    frame = frame[frame.starttime_post.notna()].reset_index(drop=True)
    buckets = frame.caseid.map(lambda x: kceiling.bucket(x, args.seed))
    train, test = frame[buckets <= 5].copy(), frame[buckets >= 8].copy()

    encoder = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), sources)
    ])
    x_train = encoder.fit_transform(train[sources])
    x_test = encoder.transform(test[sources])
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=args.seed)
    z_train = svd.fit_transform(x_train)
    z_test = svd.transform(x_test)
    mean, std = z_train.mean(axis=0), z_train.std(axis=0)
    z_train = (z_train - mean) / np.maximum(std, 1e-8)
    z_test = (z_test - mean) / np.maximum(std, 1e-8)

    coefficient_rows = []
    warm_coefficients = {}
    for item in option_bank_items:
        option_frame = options[options.item.eq(item)].sort_values("option_position")
        codes = option_frame.option_code.astype(int).tolist()
        coefficients = fit_option_coefficients(z_train, train[item], codes)
        warm_coefficients[item] = coefficients
        for row, coefficient in zip(option_frame.itertuples(index=False), coefficients):
            coefficient_rows.append({
                "item": item, "family": row.family,
                "option_code": int(row.option_code),
                "option_text": row.option_text,
                "option_position": int(row.option_position),
                "n_options": int(row.n_options),
                "coef_intercept": coefficient[0],
                "coef_z1": coefficient[1], "coef_z2": coefficient[2],
            })
    coefficient_frame = pd.DataFrame(coefficient_rows)

    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer(args.bge_model, device="cpu")
    all_text = coefficient_frame.option_text.tolist()
    bge_vectors = bge.encode(all_text, batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False)
    score_records = []
    fold_records = []
    for fold in range(fold_manifest["n_folds"]):
        test_families = {family for family, value in fold_manifest["family_to_fold"].items()
                         if value == fold}
        test_mask = (
            coefficient_frame.item.isin(items)
            & coefficient_frame.family.isin(test_families)
        ).to_numpy()
        train_mask = ~test_mask
        if not test_mask.any():
            continue
        train_options = coefficient_frame[train_mask]
        test_options = coefficient_frame[test_mask]
        y_coeff = coefficient_frame[["coef_intercept", "coef_z1", "coef_z2"]].to_numpy()
        groups = train_options.family.to_numpy()

        position = position_features(coefficient_frame)
        position_model, position_alpha, _ = group_ridge(
            position[train_mask], y_coeff[train_mask], groups)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_train = vectorizer.fit_transform(train_options.option_text).toarray()
        tfidf_test = vectorizer.transform(test_options.option_text).toarray()
        tfidf_model, tfidf_alpha, _ = group_ridge(
            tfidf_train, y_coeff[train_mask], groups)

        bge_model, bge_alpha, _ = group_ridge(
            np.asarray(bge_vectors)[train_mask], y_coeff[train_mask], groups)

        predicted = {
            "position": position_model.predict(position[test_mask]),
            "tfidf": tfidf_model.predict(tfidf_test),
            "bge": bge_model.predict(np.asarray(bge_vectors)[test_mask]),
        }
        fold_records.append({
            "fold": fold, "test_families": sorted(test_families),
            "train_option_rows": int(train_mask.sum()),
            "test_option_rows": int(test_mask.sum()),
            "alphas": {"position": position_alpha, "tfidf": tfidf_alpha,
                       "bge": bge_alpha},
        })
        test_indices = np.flatnonzero(test_mask)
        index_lookup = {global_index: local for local, global_index in enumerate(test_indices)}
        for item in test_options.item.unique():
            option_frame = options[options.item.eq(item)].sort_values("option_position")
            family = str(option_frame.family.iloc[0])
            global_indices = coefficient_frame.index[
                coefficient_frame.item.eq(item)
            ].to_numpy()
            item_rows = test[item].notna().to_numpy()
            respondents = test.loc[item_rows, "caseid"].to_numpy()
            answers = test.loc[item_rows, item].astype(int).to_numpy()
            z = z_test[item_rows]
            # True cold floor.
            uniform = np.full((len(z), len(option_frame)), 1 / len(option_frame))
            append_scores(score_records, "uniform", item, family, respondents,
                          answers, uniform, option_frame.option_code.to_numpy())
            # Non-deployable warm ceiling.
            p, codes = predictions_from_coefficients(
                z, option_frame, warm_coefficients[item])
            append_scores(score_records, "warm_oracle", item, family, respondents,
                          answers, p, codes)
            population_coefficients = warm_coefficients[item].copy()
            population_coefficients[:, 1:] = 0.0
            p, codes = predictions_from_coefficients(
                z, option_frame, population_coefficients)
            append_scores(score_records, "population_oracle", item, family,
                          respondents, answers, p, codes)
            for method, matrix in predicted.items():
                local_indices = [index_lookup[index] for index in global_indices]
                coefficients = matrix[local_indices]
                p, codes = predictions_from_coefficients(z, option_frame, coefficients)
                append_scores(score_records, method, item, family, respondents,
                              answers, p, codes)
                # Diagnostic used by Ku-style designs that provide the target
                # population distribution: hold base-rate intercepts at their
                # warm values and test only whether text maps PERSON slopes.
                with_prior = coefficients.copy()
                with_prior[:, 0] = warm_coefficients[item][:, 0]
                p, codes = predictions_from_coefficients(z, option_frame, with_prior)
                append_scores(score_records, f"{method}_oracle_intercept", item,
                              family, respondents, answers, p, codes)

    scores = pd.DataFrame(score_records)
    summary = scores.groupby("method").agg(
        nll=("nll", "mean"), accuracy=("correct", "mean"),
        rows=("nll", "size"), respondents=("panel_id", "nunique"),
        items=("item", "nunique"), families=("family", "nunique"),
    ).reset_index().to_dict(orient="records")
    comparisons = {}
    for left, right in [("tfidf", "position"), ("bge", "position"),
                        ("bge", "tfidf"), ("warm_oracle", "tfidf")]:
        comparisons[f"{left}_vs_{right}"] = {
            "respondent": clustered_comparison(scores, left, right, "respondent",
                                                args.seed),
            "item": clustered_comparison(scores, left, right, "item", args.seed),
        }
    for method in ("position", "tfidf", "bge"):
        left = f"{method}_oracle_intercept"
        comparisons[f"{left}_vs_population_oracle"] = {
            "respondent": clustered_comparison(scores, left, "population_oracle",
                                                "respondent", args.seed),
            "item": clustered_comparison(scores, left, "population_oracle",
                                          "item", args.seed),
        }
    payload = {
        "dataset": "ces2020", "phase": "cold_item_loading",
        "post_freeze": True, "exploratory": True,
        "respondent_dimensions": 2,
        "respondent_source_items": sources,
        "loading_bank_source_items": loading_source_items,
        "target_items": items,
        "item_folds": fold_manifest["family_to_fold"],
        "split_counts": {"train": len(train), "test": len(test)},
        "summary": summary, "comparisons": comparisons,
        "folds": fold_records,
        "caveat": "Preliminary text proxies and only nine target families; item-clustered inference is low-power.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    scores.to_parquet(output.with_suffix(".rows.parquet"), index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
