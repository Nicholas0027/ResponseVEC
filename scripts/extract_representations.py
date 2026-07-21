#!/usr/bin/env python
"""Extract and atomically cache one representation family for a protocol.

Default protocol is B (unseen-item R1); --protocol C/D extends to the
cross-domain (R2) and OOD-demographic-intersection (R3) transfer regimes of
design §5.3/§5.4. Protocol C requires --held-out-domain; Protocol D scores the
``ood_intersection`` split.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from responsevec.cache import RepresentationCache
from responsevec.config import load_and_prepare
from responsevec.data import PanelStore
from responsevec.llm_rv import (
    CausalExtractor,
    LLM2VecEncoder,
    choose_device,
    load_causal_backbone,
    load_llm2vec_encoder,
    load_llm2vec_gen_encoder,
    load_sentence_encoder,
)
from responsevec.pipeline import (
    CachedRetriever,
    extraction_settings,
    materialize_prompts,
    protocol_cache_directory,
    shared_cache_directory,
)
from responsevec.protocols import (
    ItemFolds,
    assert_no_leakage,
    build_protocol_a,
    build_protocol_b_candidates,
    build_protocol_c,
    build_protocol_d,
)
from responsevec.utils import stable_hash


def fake_vectors(prompts, dim: int, family: str):
    vectors = []
    for prompt in prompts:
        rng = np.random.default_rng(stable_hash("smoke", family, prompt))
        vectors.append(rng.normal(size=dim))
    return np.asarray(vectors, dtype=np.float32)


def build_units(protocol: str, store, folds, split: str, held_out_domain: str | None):
    """Select the protocol's unit builder. Returns (units, folds_or_none)."""
    if protocol == "B":
        units = build_protocol_b_candidates(store, folds, split)
        return units, folds
    if protocol == "A":
        units = build_protocol_a(store, split)
        return units, None
    if protocol == "C":
        if not held_out_domain:
            raise ValueError("Protocol C requires --held-out-domain")
        units = build_protocol_c(store, held_out_domain, split)
        return units, None
    if protocol == "D":
        # Protocol D always scores the ood_intersection split; --split is ignored.
        units = build_protocol_d(store, split="ood_intersection")
        return units, None
    raise ValueError(f"Unknown protocol: {protocol!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--family", choices=["causal", "input_centric", "response_centric", "sentence"], required=True)
    parser.add_argument("--protocol", choices=["A", "B", "C", "D"], default="B",
                        help="A=seen-item, B=unseen-item (R1, default), C=cross-domain (R2), D=OOD-intersection (R3)")
    parser.add_argument("--held-out-domain", default=None, help="required for Protocol C")
    parser.add_argument("--fold", type=int, default=0, help="deprecated: shared caches are fold-independent")
    parser.add_argument("--role", choices=["train", "validation", "test"], default=None, help="deprecated")
    parser.add_argument("--split", choices=["train", "validation", "test"], default=None,
                        help="respondent split; Protocol D ignores this (always ood_intersection)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--option-seed", type=int, default=0)
    parser.add_argument("--selection", choices=["semantic", "random", "all"], default="semantic")
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--synthetic-encoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    store = PanelStore.from_dir(config["paths"]["processed"])
    folds = ItemFolds.load(Path(config["paths"]["processed"]) / "item_folds.json")
    # Protocol D always uses the ood_intersection split regardless of --split.
    split = "ood_intersection" if args.protocol == "D" else args.split
    if not split:
        raise ValueError(f"--split is required for protocol {args.protocol}")
    units, leak_folds = build_units(args.protocol, store, folds, split, args.held_out_domain)
    assert_no_leakage(units, leak_folds)
    if not units:
        raise RuntimeError(f"protocol {args.protocol} produced no units")

    retriever = None
    rep = config["representation"]
    revisions = rep.get("revisions", {})
    if args.k > 0 and args.selection == "semantic":
        if args.synthetic_encoder:
            encode = lambda texts: fake_vectors(texts, 24, "retriever")
            retriever = CachedRetriever(LLM2VecEncoder(encode, "synthetic-retriever"))
        else:
            retriever = CachedRetriever(load_sentence_encoder(
                config["history"]["retriever"], revision=revisions.get("sentence_encoder")
            ))
        retriever.prime(store.responses["question"].astype(str).unique())
    rows, prompts = materialize_prompts(
        store, units, k=args.k, retriever=retriever, selection=args.selection,
        history_seed=int(config["data"]["calibration_seed"]), option_seed=args.option_seed,
    )
    permutations = [json.loads(value) for value in rows["label_to_semantic_json"]]
    item_split = f"{args.protocol}/shared/split={split}"
    # Fold the HARDWARE-RESOLVED precision into the fingerprint. Synthetic (smoke)
    # runs never touch a GPU, so their fingerprint stays quantization-free and
    # matches across machines; real runs resolve nf4-vs-bf16 exactly as the
    # extractor will, so an L4 4-bit cache can never masquerade as an A100 bf16
    # cache (or vice versa).
    if args.synthetic_encoder:
        resolved_quantization = "__unset__"
    else:
        from responsevec.llm_rv import resolve_quantization

        resolved_quantization = resolve_quantization(rep.get("quantization"))
    settings = extraction_settings(config, args.selection, resolved_quantization)
    # Shared caches are intentionally independent of the outer fold, but they
    # must never survive a data or item-partition change unnoticed.  Keep these
    # deterministic hashes in the cache fingerprint rather than in row metadata.
    # Protocol B carries the item-fold partition; A/C/D (seen-item protocols)
    # carry only the candidate-row hash, since they do not use the fold split.
    if args.protocol == "B":
        partition_payload = {
            "calibration_pool": sorted(folds.calibration_pool),
            "target_folds": {
                domain: [sorted(items) for items in domain_folds]
                for domain, domain_folds in sorted(folds.target_folds.items())
            },
        }
        settings["item_partition_hash"] = f"{stable_hash(partition_payload):016x}"
    settings["candidate_rows_hash"] = f"{stable_hash(sorted(rows['row_id'].astype(str))):016x}"
    settings["protocol"] = args.protocol
    if args.held_out_domain:
        settings["held_out_domain"] = args.held_out_domain
    root = Path(config["paths"]["cache"])

    def _cache_dir(family: str) -> Path:
        return protocol_cache_directory(
            root, protocol=args.protocol, respondent_split=split, k=args.k,
            option_seed=args.option_seed, family=family,
            held_out_domain=args.held_out_domain,
        )

    if args.family == "causal":
        checkpoints = {"causal_final": rep["backbone"], "raw_mean": rep["backbone"]}
        paths = {family: _cache_dir(family) for family in checkpoints}
        caches = {
            family: RepresentationCache.create(
                paths[family], family=family, checkpoint=checkpoint,
                item_split=item_split, k=args.k, option_seed=args.option_seed,
                has_logits=family == "causal_final", settings=settings, overwrite=args.overwrite,
            ) for family, checkpoint in checkpoints.items()
        }
        if args.synthetic_encoder:
            extractor = None
        else:
            model, tokenizer = load_causal_backbone(
                rep["backbone"], rep["dtype"], rep.get("quantization"),
                revision=revisions.get("backbone"),
            )
            extractor = CausalExtractor(
                model, tokenizer, choose_device(), max_length=int(rep["max_length"]),
                batch_size=int(rep["batch_size"]),
            )
        done = {family: cache.already_done_row_ids() for family, cache in caches.items()}
        global_max_options = int(rows["n_options"].max())
        needed = [index for index, row_id in enumerate(rows["row_id"].astype(str)) if any(row_id not in values for values in done.values())]
        for start in range(0, len(needed), args.shard_size):
            indices = needed[start : start + args.shard_size]
            chunk_prompts = [prompts[index] for index in indices]
            chunk_n = rows.iloc[indices]["n_options"].astype(int).tolist()
            chunk_perm = [permutations[index] for index in indices]
            if extractor is None:
                final = fake_vectors(chunk_prompts, 48, "causal_final")
                mean = fake_vectors(chunk_prompts, 48, "raw_mean")
                logits = np.zeros((len(indices), max(chunk_n)), dtype=np.float32)
                for i, n in enumerate(chunk_n):
                    logits[i, :n] = final[i, :n]
                output = {"final": final, "mean": mean, "logits": logits}
            else:
                output = extractor.extract(chunk_prompts, chunk_n, chunk_perm)
            if output["logits"].shape[1] < global_max_options:
                padded = np.full((len(indices), global_max_options), -np.inf, dtype=np.float32)
                padded[:, : output["logits"].shape[1]] = output["logits"]
                output["logits"] = padded
            chunk_rows = rows.iloc[indices].reset_index(drop=True)
            for family, vector_key in (("causal_final", "final"), ("raw_mean", "mean")):
                keep = np.asarray([str(row_id) not in done[family] for row_id in chunk_rows["row_id"]])
                if keep.any():
                    caches[family].append(
                        chunk_rows.loc[keep].reset_index(drop=True), output[vector_key][keep],
                        logits=output["logits"][keep] if family == "causal_final" else None,
                    )
                    done[family].update(chunk_rows.loc[keep, "row_id"].astype(str))
        summary = {family: cache.validate() for family, cache in caches.items()}
    else:
        if args.family == "input_centric":
            checkpoint = rep["input_centric"]
            encoder = None if args.synthetic_encoder else load_llm2vec_encoder(
                checkpoint, rep["dtype"], base_checkpoint=rep.get("input_centric_base"),
                max_length=int(rep["max_length"]), batch_size=int(rep["batch_size"]),
                checkpoint_revision=revisions.get("input_centric"),
                base_revision=revisions.get("input_centric_base"),
                foundation_revision=revisions.get("backbone"),
            )
        elif args.family == "response_centric":
            checkpoint = rep["response_centric"]
            encoder = None if args.synthetic_encoder else load_llm2vec_gen_encoder(
                checkpoint, rep["dtype"], max_length=int(rep["max_length"]),
                batch_size=int(rep["batch_size"]), revision=revisions.get("response_centric"),
            )
        else:
            checkpoint = rep["sentence_encoder"]
            encoder = None if args.synthetic_encoder else load_sentence_encoder(
                checkpoint, revision=revisions.get("sentence_encoder")
            )
        directory = _cache_dir(args.family)
        cache = RepresentationCache.create(
            directory, family=args.family, checkpoint=checkpoint, item_split=item_split,
            k=args.k, option_seed=args.option_seed, has_logits=False, settings=settings,
            overwrite=args.overwrite,
        )
        done = cache.already_done_row_ids()
        needed = [index for index, row_id in enumerate(rows["row_id"].astype(str)) if row_id not in done]
        for start in range(0, len(needed), args.shard_size):
            indices = needed[start : start + args.shard_size]
            chunk_prompts = [prompts[index] for index in indices]
            vectors = fake_vectors(chunk_prompts, 48, args.family) if encoder is None else encoder.encode(chunk_prompts)
            cache.append(rows.iloc[indices].reset_index(drop=True), vectors)
        summary = {args.family: cache.validate()}
    print({"rows": len(rows), "family": args.family, "summary": summary})
    gc.collect()


if __name__ == "__main__":
    main()
