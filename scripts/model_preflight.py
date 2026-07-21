#!/usr/bin/env python
"""Fail fast on model/API incompatibilities before the expensive extraction.

The full check downloads and runs the 0.6B versions of all three model paths.
It is intentionally separate from the scientific cache and produces no paper
predictions.  ``--metadata-only`` checks checkpoint visibility and dependency
versions without requiring CUDA or downloading weights.
"""

from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
from pathlib import Path

import numpy as np

from responsevec.config import load_and_prepare
from responsevec.llm_rv import (
    CausalExtractor,
    choose_device,
    load_causal_backbone,
    load_llm2vec_encoder,
    load_llm2vec_gen_encoder,
)
from responsevec.utils import write_json


PROMPTS = [
    "Task: Predict this respondent's answer.\nTarget question: Agree?\n"
    "Options:\nA. Agree\nB. Disagree\n\nAnswer:",
    "Task: Predict this respondent's answer.\nTarget question: Concerned?\n"
    "Options:\nA. Concerned\nB. Not concerned\n\nAnswer:",
]


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/responsevec.yaml")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    config = load_and_prepare(args.config)
    rep = config["representation"]

    from huggingface_hub import model_info

    checkpoints = {
        "causal_small": rep["small_backbone"],
        "input_mntp_small": rep["small_input_centric_base"],
        "input_supervised_small": rep["small_input_centric"],
        "response_small": rep["small_response_centric"],
        "causal_primary": rep["backbone"],
        "input_mntp_primary": rep["input_centric_base"],
        "input_supervised_primary": rep["input_centric"],
        "response_primary": rep["response_centric"],
    }
    revision_keys = {
        "causal_small": "small_backbone",
        "input_mntp_small": "small_input_centric_base",
        "input_supervised_small": "small_input_centric",
        "response_small": "small_response_centric",
        "causal_primary": "backbone",
        "input_mntp_primary": "input_centric_base",
        "input_supervised_primary": "input_centric",
        "response_primary": "response_centric",
    }
    pinned = rep.get("revisions", {})
    revisions = {
        name: model_info(identifier, revision=pinned.get(revision_keys[name])).sha
        for name, identifier in checkpoints.items()
    }
    expected = {name: pinned.get(key) for name, key in revision_keys.items()}
    if revisions != expected:
        raise RuntimeError(f"checkpoint revision mismatch: resolved={revisions}, expected={expected}")
    result = {
        "status": "METADATA_PASS" if args.metadata_only else "PENDING",
        "versions": {
            package: version(package)
            for package in ("torch", "transformers", "peft")
        },
        "checkpoint_revisions": revisions,
        "scientific_results": False,
    }
    if args.metadata_only:
        output = Path(config["paths"]["metrics"]) / "model_preflight.json"
        write_json(output, result)
        print(result)
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("full model preflight requires a CUDA runtime")
    if version("transformers") != "4.56.2":
        raise RuntimeError("Qwen3 MNTP remote code is validated against transformers==4.56.2")

    causal, tokenizer = load_causal_backbone(
        rep["small_backbone"], rep["dtype"], None,
        revision=pinned.get("small_backbone"),
    )
    extractor = CausalExtractor(causal, tokenizer, choose_device(), max_length=128, batch_size=2)
    direct = extractor.extract(PROMPTS, [2, 2])
    if direct["final"].shape[0] != 2 or not np.isfinite(direct["probabilities"]).all():
        raise RuntimeError("small causal extraction returned invalid arrays")
    del extractor, tokenizer, causal
    release_memory()

    input_encoder = load_llm2vec_encoder(
        rep["small_input_centric"], rep["dtype"],
        base_checkpoint=rep["small_input_centric_base"], max_length=128, batch_size=2,
        checkpoint_revision=pinned.get("small_input_centric"),
        base_revision=pinned.get("small_input_centric_base"),
        foundation_revision=pinned.get("small_backbone"),
    )
    input_vectors = input_encoder.encode(PROMPTS)
    if input_vectors.shape[0] != 2 or not np.isfinite(input_vectors).all():
        raise RuntimeError("small input-centric extraction returned invalid arrays")
    del input_encoder
    release_memory()

    response_encoder = load_llm2vec_gen_encoder(
        rep["small_response_centric"], rep["dtype"], max_length=128, batch_size=2,
        revision=pinned.get("small_response_centric"),
    )
    response_vectors = response_encoder.encode(PROMPTS)
    if response_vectors.shape[0] != 2 or not np.isfinite(response_vectors).all():
        raise RuntimeError("small response-centric extraction returned invalid arrays")
    del response_encoder
    release_memory()

    result.update({
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "shapes": {
            "causal": list(direct["final"].shape),
            "input_centric": list(input_vectors.shape),
            "response_centric": list(response_vectors.shape),
        },
    })
    output = Path(config["paths"]["metrics"]) / "model_preflight.json"
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    main()
