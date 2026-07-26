#!/usr/bin/env python3
"""Calibrate structured LLM item features to respondent-space loadings.

Reads the Qwen3-32B structured item records, converts them to per-option
feature vectors, and maps those features to warm (intercept, z1, z2)
coefficients under the same family-disjoint fold framework as
``run_cold_item_loading.py``.  Compares the structured LLM arm against
TF-IDF, BGE, position, and warm oracle.

The structured features per option row are:
  - stance_direction, behavior_intensity, is_neutral, uncertainty  (option-level)
  - global_uncertainty                                              (item-level)
  - item_type one-hot, scale_type one-hot                          (categorical)
  - domain multi-hot                                               (top-3)
  - construct relevance-weighted multi-hot                        (top-4)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("cold", HERE / "run_cold_item_loading.py")
cold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cold)

spec2 = importlib.util.spec_from_file_location("kceiling", HERE / "run_k_ceiling.py")
kceiling = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(kceiling)

ITEM_TYPES = ["attitude", "behavior", "factual_knowledge", "perception",
              "identity", "demographic", "other"]
SCALE_TYPES = ["binary", "ordinal", "nominal", "multi_select_binary",
               "numeric", "other"]


def load_structured(path: Path) -> dict[tuple[str, str], dict]:
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("valid"):
            continue
        key = (record["item"], record["template"])
        records[key] = record["data"]
    return records


def build_domain_vocabulary(structured: dict) -> list[str]:
    domains = set()
    for data in structured.values():
        domains.update(data.get("domains", []))
    return sorted(domains)


def build_construct_vocabulary(structured: dict) -> list[str]:
    constructs = set()
    for data in structured.values():
        for construct in data.get("constructs", []):
            constructs.add(construct["name"])
    return sorted(constructs)


def option_features(record_data: dict, domain_vocab: list[str],
                    construct_vocab: list[str]) -> np.ndarray:
    item_type_vec = np.zeros(len(ITEM_TYPES))
    if record_data["item_type"] in ITEM_TYPES:
        item_type_vec[ITEM_TYPES.index(record_data["item_type"])] = 1.0

    scale_vec = np.zeros(len(SCALE_TYPES))
    if record_data["scale_type"] in SCALE_TYPES:
        scale_vec[SCALE_TYPES.index(record_data["scale_type"])] = 1.0

    domain_vec = np.zeros(len(domain_vocab))
    for domain in record_data.get("domains", []):
        if domain in domain_vocab:
            domain_vec[domain_vocab.index(domain)] = 1.0

    construct_vec = np.zeros(len(construct_vocab))
    for construct in record_data.get("constructs", []):
        name = construct["name"]
        if name in construct_vocab:
            construct_vec[construct_vocab.index(name)] = float(construct["relevance"])

    item_level = np.array([
        float(record_data.get("global_uncertainty", 0.0)),
    ])
    return np.concatenate([item_type_vec, scale_vec, domain_vec,
                           construct_vec, item_level])


def structured_option_matrix(coefficient_frame: pd.DataFrame,
                             structured: dict,
                             domain_vocab: list[str],
                             construct_vocab: list[str],
                             template: str = "v1") -> np.ndarray:
    rows = []
    missing = 0
    for row in coefficient_frame.itertuples(index=False):
        data = structured.get((row.item, template))
        if data is None:
            data = structured.get((row.item, "v2"))
        item_feat = option_features(data, domain_vocab, construct_vocab) if data else None
        if data is not None:
            option_map = {int(opt["code"]): opt for opt in data["options"]}
            opt = option_map.get(int(row.option_code))
        else:
            opt = None
        if opt is not None:
            opt_level = np.array([
                float(opt["stance_direction"]),
                float(opt["behavior_intensity"]),
                float(opt.get("is_neutral", False)),
                float(opt["uncertainty"]),
            ])
            rows.append(np.concatenate([opt_level, item_feat]))
        else:
            missing += 1
            dim = 4 + len(ITEM_TYPES) + len(SCALE_TYPES) + len(domain_vocab) + len(construct_vocab) + 1
            rows.append(np.zeros(dim))
    return np.asarray(rows), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv")
    parser.add_argument("--source-catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--source-manifest", default="cross_survey/metadata/ces2020_cross_construct_manifest.json")
    parser.add_argument("--options", default="cross_survey/metadata/ces2020_all_item_options.csv")
    parser.add_argument("--folds", default="cross_survey/metadata/ces2020_cold_item_folds.json")
    parser.add_argument("--structured", default="cross_survey/results/structured_items_qwen32b.jsonl")
    parser.add_argument("--output", default="cross_survey/results/phase1/structured_loading.json")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    structured = load_structured(Path(args.structured))
    print(f"loaded {len(structured)} valid structured records "
          f"({len({k[0] for k in structured})} items)")

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

    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import OneHotEncoder

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
        coefficients = cold.fit_option_coefficients(z_train, train[item], codes)
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

    domain_vocab = build_domain_vocabulary(structured)
    construct_vocab = build_construct_vocabulary(structured)
    print(f"domain vocab: {len(domain_vocab)}; construct vocab: {len(construct_vocab)}")

    struct_features, n_missing = structured_option_matrix(
        coefficient_frame, structured, domain_vocab, construct_vocab, "v1")
    print(f"structured feature matrix: {struct_features.shape}; missing options: {n_missing}")

    from sklearn.feature_extraction.text import TfidfVectorizer

    score_records = []
    fold_records = []
    for fold_idx in range(fold_manifest["n_folds"]):
        test_families = {family for family, value in fold_manifest["family_to_fold"].items()
                         if value == fold_idx}
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

        position = cold.position_features(coefficient_frame)
        position_model, position_alpha, _ = cold.group_ridge(
            position[train_mask], y_coeff[train_mask], groups)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_train = vectorizer.fit_transform(train_options.option_text).toarray()
        tfidf_test = vectorizer.transform(test_options.option_text).toarray()
        tfidf_model, tfidf_alpha, _ = cold.group_ridge(
            tfidf_train, y_coeff[train_mask], groups)

        struct_model, struct_alpha, _ = cold.group_ridge(
            struct_features[train_mask], y_coeff[train_mask], groups)

        struct_v2_features, _ = structured_option_matrix(
            coefficient_frame, structured, domain_vocab, construct_vocab, "v2")
        struct_v2_model, struct_v2_alpha, _ = cold.group_ridge(
            struct_v2_features[train_mask], y_coeff[train_mask], groups)

        predicted = {
            "position": position_model.predict(position[test_mask]),
            "tfidf": tfidf_model.predict(tfidf_test),
            "structured_llm": struct_model.predict(struct_features[test_mask]),
            "structured_llm_v2": struct_v2_model.predict(struct_v2_features[test_mask]),
        }
        fold_records.append({
            "fold": fold_idx, "test_families": sorted(test_families),
            "train_option_rows": int(train_mask.sum()),
            "test_option_rows": int(test_mask.sum()),
            "alphas": {"position": position_alpha, "tfidf": tfidf_alpha,
                       "structured_llm": struct_alpha,
                       "structured_llm_v2": struct_v2_alpha},
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
            uniform = np.full((len(z), len(option_frame)), 1 / len(option_frame))
            cold.append_scores(score_records, "uniform", item, family, respondents,
                               answers, uniform, option_frame.option_code.to_numpy())
            p, codes = cold.predictions_from_coefficients(
                z, option_frame, warm_coefficients[item])
            cold.append_scores(score_records, "warm_oracle", item, family, respondents,
                               answers, p, codes)
            population_coefficients = warm_coefficients[item].copy()
            population_coefficients[:, 1:] = 0.0
            p, codes = cold.predictions_from_coefficients(
                z, option_frame, population_coefficients)
            cold.append_scores(score_records, "population_oracle", item, family,
                               respondents, answers, p, codes)
            for method, matrix in predicted.items():
                local_indices = [index_lookup[index] for index in global_indices]
                coefficients = matrix[local_indices]
                p, codes = cold.predictions_from_coefficients(z, option_frame, coefficients)
                cold.append_scores(score_records, method, item, family, respondents,
                                   answers, p, codes)
                with_prior = coefficients.copy()
                with_prior[:, 0] = warm_coefficients[item][:, 0]
                p, codes = cold.predictions_from_coefficients(z, option_frame, with_prior)
                cold.append_scores(score_records, f"{method}_oracle_intercept", item,
                                   family, respondents, answers, p, codes)

    scores = pd.DataFrame(score_records)
    summary = scores.groupby("method").agg(
        nll=("nll", "mean"), accuracy=("correct", "mean"),
        rows=("nll", "size"), respondents=("panel_id", "nunique"),
        items=("item", "nunique"), families=("family", "nunique"),
    ).reset_index().to_dict(orient="records")

    comparisons = {}
    pairs = [
        ("structured_llm", "tfidf"),
        ("structured_llm", "position"),
        ("structured_llm", "bge"),
        ("structured_llm_v2", "structured_llm"),
        ("tfidf", "position"),
        ("warm_oracle", "structured_llm"),
        ("warm_oracle", "tfidf"),
    ]
    for left, right in pairs:
        if left in scores.method.values and right in scores.method.values:
            comparisons[f"{left}_vs_{right}"] = {
                "respondent": cold.clustered_comparison(scores, left, right, "respondent",
                                                         args.seed),
                "family": cold.clustered_comparison(scores, left, right, "item",
                                                     args.seed),
            }
    for method in ("structured_llm", "tfidf", "position"):
        left = f"{method}_oracle_intercept"
        if left in scores.method.values and "population_oracle" in scores.method.values:
            comparisons[f"{left}_vs_population_oracle"] = {
                "respondent": cold.clustered_comparison(scores, left, "population_oracle",
                                                         "respondent", args.seed),
                "family": cold.clustered_comparison(scores, left, "population_oracle",
                                                     "item", args.seed),
            }

    tfidf_headroom = None
    warm_rows = scores[scores.method.eq("warm_oracle")]
    tfidf_rows = scores[scores.method.eq("tfidf")]
    struct_rows = scores[scores.method.eq("structured_llm")]
    if not warm_rows.empty and not tfidf_rows.empty and not struct_rows.empty:
        tfidf_headroom = {
            "tfidf_nll_recovery_pct": float(
                (tfidf_rows.nll.mean() - warm_rows.nll.mean())
                / (1.3310 - warm_rows.nll.mean()) * 100
            ) if (1.3310 - warm_rows.nll.mean()) != 0 else None,
            "structured_nll_recovery_pct": float(
                (struct_rows.nll.mean() - warm_rows.nll.mean())
                / (1.3310 - warm_rows.nll.mean()) * 100
            ) if (1.3310 - warm_rows.nll.mean()) != 0 else None,
        }

    payload = {
        "dataset": "ces2020", "phase": "structured_loading",
        "post_freeze": True, "exploratory": True,
        "model": "Qwen3-32B-AWQ",
        "structured_records_loaded": len(structured),
        "structured_items_covered": len({k[0] for k in structured}),
        "domain_vocab_size": len(domain_vocab),
        "construct_vocab_size": len(construct_vocab),
        "feature_dim": int(struct_features.shape[1]),
        "missing_option_features": int(n_missing),
        "respondent_dimensions": 2,
        "respondent_source_items": sources,
        "loading_bank_source_items": loading_source_items,
        "target_items": items,
        "item_folds": fold_manifest["family_to_fold"],
        "split_counts": {"train": len(train), "test": len(test)},
        "summary": summary,
        "comparisons": comparisons,
        "folds": fold_records,
        "tfidf_headroom_reference": tfidf_headroom,
        "caveat": "Preliminary structured LLM features; only nine target families; "
                  "family-clustered inference is low-power.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    scores.to_parquet(output.with_suffix(".rows.parquet"), index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
