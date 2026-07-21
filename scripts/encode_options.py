#!/usr/bin/env python
"""Encode every question-option pair once with the shared frozen E_o."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from responsevec.config import load_and_prepare
from responsevec.data import PanelStore
from responsevec.encode import build_option_table, save_option_table
from responsevec.llm_rv import LLM2VecEncoder, load_llm2vec_encoder
from responsevec.utils import stable_hash, write_json


def deterministic_smoke_encoder(texts):
    vectors = []
    for text in texts:
        seed = int.from_bytes(__import__("hashlib").sha256(text.encode()).digest()[:8], "little")
        vectors.append(np.random.default_rng(seed).normal(size=32))
    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--synthetic-encoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    output = Path(config["paths"]["cache"]) / "option_table.npz"
    metadata_path = output.with_suffix(".meta.json")
    rep = config["representation"]
    revisions = rep.get("revisions", {})
    identity = {
        "checkpoint": "synthetic-option-encoder" if args.synthetic_encoder else rep["option_encoder"],
        "checkpoint_revision": None if args.synthetic_encoder else revisions.get("input_centric"),
        "base_checkpoint": None if args.synthetic_encoder else rep.get("input_centric_base"),
        "base_revision": None if args.synthetic_encoder else revisions.get("input_centric_base"),
        "foundation_revision": None if args.synthetic_encoder else revisions.get("backbone"),
    }
    identity["fingerprint"] = f"{stable_hash(identity):016x}"
    if output.exists() and not args.overwrite:
        if not metadata_path.exists():
            raise RuntimeError(f"option table has no identity metadata: {metadata_path}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("option-table identity changed; rerun with --overwrite into a clean artifact root")
        print(f"reuse {output}")
        return
    store = PanelStore.from_dir(config["paths"]["processed"])
    catalogue = store.responses[["question_key", "question", "options_json"]].drop_duplicates("question_key")
    if args.synthetic_encoder:
        encoder = LLM2VecEncoder(deterministic_smoke_encoder, name="synthetic-option-encoder")
    else:
        encoder = load_llm2vec_encoder(
            rep["option_encoder"], dtype=rep["dtype"],
            base_checkpoint=rep.get("input_centric_base"),
            max_length=int(rep["max_length"]), batch_size=int(rep["batch_size"]),
            checkpoint_revision=revisions.get("input_centric"),
            base_revision=revisions.get("input_centric_base"),
            foundation_revision=revisions.get("backbone"),
        )
    table = build_option_table(catalogue, encoder)
    save_option_table(table, output)
    write_json(metadata_path, identity)
    print({"option_table": str(output), "items": len(table), "dim": next(iter(table.values())).shape[1]})


if __name__ == "__main__":
    main()
