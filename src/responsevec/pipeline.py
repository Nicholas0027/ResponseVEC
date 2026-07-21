"""Shared orchestration primitives for scripts and notebooks.

This module deliberately contains no CLI and no global state. It converts
protocol units into canonical prompts, builds fold-safe priors, and standardizes
cache paths so a Colab restart can resume without guessing filenames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .prompting_rv import (
    build_canonical_prompt,
    build_respondent_prompt,
    deterministic_permutation,
    select_history,
)
from .protocols import EvalUnit, RespondentUnit
from .utils import stable_hash


class CachedRetriever:
    """Cache question embeddings in memory and batch-prime all known texts."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.cache: dict[str, np.ndarray] = {}

    def prime(self, texts: Iterable[str]) -> None:
        missing = sorted({str(text) for text in texts} - self.cache.keys())
        if not missing:
            return
        vectors = np.asarray(self.encoder.encode(missing), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(missing):
            raise ValueError(f"retriever returned invalid shape {vectors.shape}")
        self.cache.update({text: vector for text, vector in zip(missing, vectors)})

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        self.prime(texts)
        return np.stack([self.cache[str(text)] for text in texts])


def _target_lookup(store) -> dict[tuple[str, str], dict[str, Any]]:
    duplicated = store.responses.duplicated(["panel_id", "question_key"])
    if duplicated.any():
        raise ValueError("responses contain duplicate (panel_id, question_key) rows")
    return {
        (str(row.panel_id), str(row.question_key)): row._asdict()
        for row in store.responses.itertuples(index=False)
    }


def materialize_prompts(
    store,
    units: Sequence[EvalUnit],
    *,
    k: int,
    retriever=None,
    selection: str = "semantic",
    history_seed: int = 1701,
    option_seed: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one prompt per fixed target unit.

    ``option_seed=0`` is identity order. Positive seeds use deterministic
    per-row permutations. Rows store hashes and semantic mappings, not prompt
    text, reducing accidental exposure of respondent context in caches.
    """
    lookup = _target_lookup(store)
    if selection == "semantic" and retriever is None and k > 0:
        raise ValueError("semantic history selection requires a retriever")
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    for unit in units:
        key = (str(unit.panel_id), str(unit.question_key))
        if key not in lookup:
            raise KeyError(f"protocol unit target missing from responses: {key}")
        row = lookup[key]
        panel = store.by_panel[unit.panel_id]
        sources = panel[panel["question_key"].astype(str).isin(unit.eligible_source_keys)].to_dict("records")
        history = select_history(
            str(row["question"]), sources, int(k), retriever,
            selection=selection, random_seed=int(history_seed), panel_id=str(unit.panel_id),
        )
        if option_seed == 0:
            permutation = list(range(int(row["n_options"])))
        else:
            permutation = deterministic_permutation(
                int(row["n_options"]), int(option_seed), str(row["row_id"])
            ).tolist()
        prompt, correct_label, label_to_semantic = build_canonical_prompt(row, history, permutation)
        prompts.append(prompt)
        record = dict(row)
        record.update({
            "protocol": unit.protocol,
            "fold": int(unit.fold),
            "target_role": unit.target_role,
            "item_pool": unit.item_pool,
            "k": int(k),
            "history_seed": int(history_seed),
            "option_seed": int(option_seed),
            "history_question_keys_json": json.dumps([str(item["question_key"]) for item in history]),
            "label_to_semantic_json": json.dumps(label_to_semantic),
            "correct_label_index": int(correct_label),
            "prompt_hash": f"{stable_hash(prompt):016x}",
        })
        records.append(record)
    return pd.DataFrame(records), prompts


def materialize_respondent_prompts(
    store, units: Sequence[RespondentUnit], *, k: int, history_seed: int = 1701,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one query-independent prompt per (panel_id, domain) unit (design
    §2.3.D). History selection is deliberately selection='all' (deterministic,
    target-agnostic, capped at k) — there is no target question here to rank
    against semantically, so 'semantic' selection is not meaningful and no
    retriever is needed, unlike materialize_prompts."""
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    for unit in units:
        panel = store.by_panel[unit.panel_id]
        demographic_text = str(panel["demographic_text"].iloc[0])
        sources = panel[panel["question_key"].astype(str).isin(unit.eligible_source_keys)].to_dict("records")
        history = select_history(
            "", sources, int(k), None, selection="all",
            random_seed=int(history_seed), panel_id=str(unit.panel_id),
        )
        prompt = build_respondent_prompt(demographic_text, history)
        prompts.append(prompt)
        records.append({
            "row_id": f"{unit.panel_id}::respondentvec::{unit.domain}::k{int(k)}",
            "panel_id": unit.panel_id,
            "domain": unit.domain,
            "k": int(k),
            "history_seed": int(history_seed),
            "history_question_keys_json": json.dumps([str(item["question_key"]) for item in history]),
            "prompt_hash": f"{stable_hash(prompt):016x}",
        })
    return pd.DataFrame(records), prompts


def extraction_settings(
    config: Mapping[str, Any],
    selection: str,
    resolved_quantization: str | None = "__unset__",
) -> dict[str, Any]:
    representation = config["representation"]
    settings = {
        "max_length": int(representation["max_length"]),
        "history_selection": selection,
        "history_retriever": config["history"]["retriever"],
        "dtype": representation["dtype"],
        "model_revisions": dict(representation.get("revisions", {})),
    }
    # The RESOLVED precision (nf4 vs bf16) is hardware-dependent — nf4 on a
    # <30GB card, bf16 on an A100/CPU — and changes the numerical value of every
    # hidden state and logit. It MUST enter the cache fingerprint, or a causal
    # cache built in 4-bit on an L4 would be silently reused (or appended to) as
    # bf16 on an A100 under an identical fingerprint, corrupting the H1/H3
    # tables (design §5.7.7: the fingerprint captures everything that changes an
    # embedding). "__unset__" preserves the pre-fix fingerprint for the encoder
    # families where the caller has no quantization to resolve.
    if resolved_quantization != "__unset__":
        settings["resolved_quantization"] = resolved_quantization
    return settings


def cache_directory(
    cache_root: str | Path,
    *,
    protocol: str,
    outer_fold: int,
    target_role: str,
    respondent_split: str,
    k: int,
    option_seed: int,
    family: str,
) -> Path:
    return (
        Path(cache_root)
        / f"protocol_{protocol}"
        / f"fold_{int(outer_fold):02d}"
        / target_role
        / respondent_split
        / f"k_{int(k)}"
        / f"option_{int(option_seed)}"
        / family
    )


def shared_cache_directory(
    cache_root: str | Path,
    *,
    respondent_split: str,
    k: int,
    option_seed: int,
    family: str,
) -> Path:
    """Protocol-B cache shared by all six outer item folds."""
    return (
        Path(cache_root) / "protocol_B" / "shared" / respondent_split
        / f"k_{int(k)}" / f"option_{int(option_seed)}" / family
    )


def protocol_cache_directory(
    cache_root: str | Path,
    *,
    protocol: str,
    respondent_split: str,
    k: int,
    option_seed: int,
    family: str,
    held_out_domain: str | None = None,
) -> Path:
    """General cache directory for any protocol (A/B/C/D). Protocol B delegates
    to ``shared_cache_directory`` for backward compatibility. Protocols C and D
    keep their own subdirectory so a cross-domain or OOD-intersection cache can
    never be confused with the R1 unseen-item cache."""
    if protocol == "B":
        return shared_cache_directory(
            cache_root, respondent_split=respondent_split, k=k,
            option_seed=option_seed, family=family,
        )
    suffix = f"_{held_out_domain}" if held_out_domain else ""
    return (
        Path(cache_root) / f"protocol_{protocol}{suffix}" / respondent_split
        / f"k_{int(k)}" / f"option_{int(option_seed)}" / family
    )


def respondentvec_cache_directory(cache_root: str | Path, *, respondent_split: str, k: int) -> Path:
    """RespondentVec cache: one vector per (panel_id, domain), shared across all
    six outer item folds and all target items (query-independent by design),
    so there is no option_seed axis (no target labels are ever scored here)."""
    return Path(cache_root) / "respondent_vec" / respondent_split / f"k_{int(k)}"


def protocol_b_item_keys(folds, outer_fold: int) -> dict[str, frozenset[str]]:
    """Union role items across domains for fold-safe fitting."""
    result: dict[str, frozenset[str]] = {}
    for role in ("train", "validation", "test"):
        result[role] = frozenset().union(
            *(folds.role_items(domain, outer_fold, role) for domain in folds.target_folds)
        )
    return result


def fit_fold_prior(store, folds, outer_fold: int, prior) -> tuple[Any, frozenset[str]]:
    train_keys = protocol_b_item_keys(folds, outer_fold)["train"]
    train_rows = store.responses[store.responses["split"].eq("train")]
    prior.fit(train_rows, allowed_question_keys=train_keys)
    return prior, train_keys
