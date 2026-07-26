#!/usr/bin/env python3
"""Inspect the top one-hot loadings of CES cross-construct respondent factors."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kceiling", HERE / "run_k_ceiling.py")
kceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kceiling)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cross_survey/data/raw/ces2020/CES20_Common_OUTPUT_vv.csv")
    parser.add_argument("--catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--manifest", default="cross_survey/metadata/ces2020_cross_construct_manifest.json")
    parser.add_argument("--output", default="cross_survey/results/phase1/latent_factor_loadings.json")
    parser.add_argument("--dimensions", type=int, default=8)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    catalog = pd.read_csv(args.catalog)
    sources = catalog[
        catalog.wave.eq("source_pre") & catalog.found
        & catalog.item_specific_text.fillna(False)
        & catalog.family.isin(manifest["source_allowed_families"])
    ].item.tolist()
    frame = pd.read_csv(args.input, usecols=["caseid", "starttime_post"] + sources,
                        low_memory=False)
    frame = frame[frame.starttime_post.notna()].reset_index(drop=True)
    buckets = frame.caseid.map(lambda x: kceiling.bucket(x, manifest["seed"]))
    train = frame[buckets <= 5]
    encoder = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), sources)
    ])
    matrix = encoder.fit_transform(train[sources])
    svd = TruncatedSVD(n_components=args.dimensions, n_iter=7,
                       random_state=manifest["seed"]).fit(matrix)
    names = encoder.get_feature_names_out()
    factors = []
    for index, component in enumerate(svd.components_):
        positive = np.argsort(component)[-args.top:][::-1]
        negative = np.argsort(component)[:args.top]
        factors.append({
            "factor": index + 1,
            "explained_variance_ratio": float(svd.explained_variance_ratio_[index]),
            "top_positive": [
                {"feature": str(names[i]), "loading": float(component[i])}
                for i in positive
            ],
            "top_negative": [
                {"feature": str(names[i]), "loading": float(component[i])}
                for i in negative
            ],
        })
    payload = {"dataset": "ces2020", "source_items": sources,
               "onehot_features": len(names), "factors": factors,
               "warning": "SVD signs are arbitrary and factors are not psychometrically identified."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"features": len(names), "variance":
                      [round(x, 4) for x in svd.explained_variance_ratio_]}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
