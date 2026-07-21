#!/usr/bin/env python
"""Extract and atomically cache RespondentVec (design §2.3.D): a query-independent
respondent representation r_i^K = E_LLM2Vec(T_r(d_i, H_i^K)). One vector per
(panel_id, domain) — NOT per target item, since the target question and its
options are deliberately absent from this prompt. Shared across all six outer
item folds and every target item for that respondent (no option_seed axis:
there is no option scoring here to permute).

    python scripts/extract_respondentvec.py --config configs/responsevec.yaml \
        --split test --k 5

Train the corresponding head with:
    python scripts/train_primary.py --config configs/responsevec.yaml \
        --fold 0 --k 5 --include-respondent-vec
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from responsevec.cache import RepresentationCache
from responsevec.config import load_and_prepare
from responsevec.data import PanelStore
from responsevec.llm_rv import load_llm2vec_encoder
from responsevec.pipeline import materialize_respondent_prompts, respondentvec_cache_directory
from responsevec.protocols import ItemFolds, build_respondentvec_units
from responsevec.utils import stable_hash


def fake_vectors(prompts, dim: int):
    vectors = []
    for prompt in prompts:
        rng = np.random.default_rng(stable_hash("smoke", "respondent_vec", prompt))
        vectors.append(rng.normal(size=dim))
    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--synthetic-encoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    store = PanelStore.from_dir(config["paths"]["processed"])
    folds = ItemFolds.load(Path(config["paths"]["processed"]) / "item_folds.json")
    units = build_respondentvec_units(store, folds, args.split)
    if not units:
        raise RuntimeError("respondentvec protocol produced no units")

    rep = config["representation"]
    revisions = rep.get("revisions", {})
    rows, prompts = materialize_respondent_prompts(
        store, units, k=args.k, history_seed=int(config["data"]["calibration_seed"]),
    )

    if args.synthetic_encoder:
        resolved_quantization = "__unset__"
        encoder = None
    else:
        encoder = load_llm2vec_encoder(
            rep["input_centric"], rep["dtype"], base_checkpoint=rep.get("input_centric_base"),
            max_length=int(rep["max_length"]), batch_size=int(rep["batch_size"]),
            checkpoint_revision=revisions.get("input_centric"),
            base_revision=revisions.get("input_centric_base"),
            foundation_revision=revisions.get("backbone"),
        )
        resolved_quantization = None  # LLM2Vec encoders are never quantized in this pipeline.

    settings = {
        "max_length": int(rep["max_length"]),
        "history_selection": "all",
        "dtype": rep["dtype"],
        "model_revisions": dict(revisions),
        "resolved_quantization": resolved_quantization,
        "candidate_rows_hash": f"{stable_hash(sorted(rows['row_id'].astype(str))):016x}",
    }

    directory = respondentvec_cache_directory(config["paths"]["cache"], respondent_split=args.split, k=args.k)
    cache = RepresentationCache.create(
        directory, family="respondent_vec", checkpoint=rep["input_centric"],
        item_split=f"respondentvec/split={args.split}", k=args.k, option_seed=0,
        has_logits=False, settings=settings, overwrite=args.overwrite,
    )
    done = cache.already_done_row_ids()
    needed = [index for index, row_id in enumerate(rows["row_id"].astype(str)) if row_id not in done]
    for start in range(0, len(needed), args.shard_size):
        indices = needed[start : start + args.shard_size]
        chunk_prompts = [prompts[index] for index in indices]
        vectors = fake_vectors(chunk_prompts, 48) if encoder is None else encoder.encode(chunk_prompts)
        cache.append(rows.iloc[indices].reset_index(drop=True), vectors)
    summary = cache.validate()
    print({"rows": len(rows), "family": "respondent_vec", "summary": summary})
    gc.collect()


if __name__ == "__main__":
    main()
