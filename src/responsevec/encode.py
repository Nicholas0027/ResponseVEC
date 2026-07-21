"""Frozen option encoder E_o and the option table it produces (design §2.4).

Every option's TEXT is encoded once, by a single frozen encoder shared across
ALL compared methods, into a vector o_c. This is the pivot of the fairness
argument: the decoder never sees option identities as free embeddings it could
memorize, so a method can only win by carrying respondent signal in z, not by
overfitting option indices. The same E_o serves the bilinear decoder, the
direct-logit calibrator, and every representation baseline.

At K=0 with an uninformative z, the decoder reduces to the prior (see
decoder.py); this module just builds and caches the (question -> option-matrix)
table so training/eval never re-encode the same option text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .prompting_rv import truncate_text


def option_encoding_text(question: str, option: str) -> str:
    """Encode an option in the context of its question so identical option
    strings ('Agree') under different questions get distinct vectors."""
    return f"{truncate_text(question, 200)} :: {truncate_text(option, 120)}"


def build_option_table(
    catalogue: pd.DataFrame,
    encoder,
) -> dict[str, np.ndarray]:
    """Encode every (question, option) once. `catalogue` has columns
    question_key, question, options_json. `encoder` exposes encode(texts)->(n,d).
    Returns {question_key: (n_options, dim) float32}."""
    texts: list[str] = []
    layout: list[tuple[str, int, int]] = []  # (question_key, start, n_options)
    for row in catalogue.itertuples(index=False):
        options = json.loads(row.options_json) if isinstance(row.options_json, str) else list(row.options_json)
        layout.append((row.question_key, len(texts), len(options)))
        texts.extend(option_encoding_text(row.question, opt) for opt in options)

    vectors = np.asarray(encoder.encode(texts), dtype=np.float32) if texts else np.zeros((0, 1), np.float32)
    table: dict[str, np.ndarray] = {}
    for question_key, start, n_options in layout:
        table[question_key] = vectors[start : start + n_options]
    return table


def save_option_table(table: Mapping[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: value for key, value in table.items()})


def load_option_table(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(Path(path), allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def stack_option_matrix(
    table: Mapping[str, np.ndarray],
    question_keys: Sequence[str],
    max_options: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-row option matrices into a padded (n_rows, max_options, dim)
    tensor plus a (n_rows, max_options) validity mask, for batched decoding."""
    if max_options <= 0:
        raise ValueError("max_options must be positive")
    if not table and question_keys:
        raise ValueError("option table is empty")
    if not question_keys:
        dim = next(iter(table.values())).shape[1] if table else 0
        return (
            np.zeros((0, max_options, dim), dtype=np.float32),
            np.zeros((0, max_options), dtype=np.float32),
        )
    dim = next(iter(table.values())).shape[1]
    matrix = np.zeros((len(question_keys), max_options, dim), dtype=np.float32)
    mask = np.zeros((len(question_keys), max_options), dtype=np.float32)
    for row, key in enumerate(question_keys):
        if key not in table:
            raise KeyError(f"missing option encoding for {key}")
        options = table[key]
        n = options.shape[0]
        if n > max_options:
            raise ValueError(f"{key} has {n} options, exceeds max_options={max_options}")
        matrix[row, :n] = options
        mask[row, :n] = 1.0
    return matrix, mask
