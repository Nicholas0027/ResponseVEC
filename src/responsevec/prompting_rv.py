"""Canonical query-focused respondent prompt (design §2.1) and the leakage-safe
history selector R_q.

Every compared representation family — direct logits, raw causal hidden states,
input-centric LLM2Vec, response-centric LLM2Vec-Gen — sees EXACTLY this prompt.
No generated persona summary is used in the primary method, so there is no
opaque teacher and no evidence asymmetry between methods.

The history selector ranks a respondent's eligible answered items by semantic
similarity of their QUESTION TEXT to the target question, using a frozen
retriever. It never reads target labels, target-item statistics, or test
outcomes. Two controls (design §2.1) share this code path:
  - random-K:  deterministic random order instead of semantic ranking;
  - all:       every eligible item up to the context budget.

Option-label utilities (single-token labels, deterministic permutations,
semantic re-mapping) are shared with the extraction path in models/.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .utils import stable_int

OPTION_LABELS = list("ABCDEFGHIJK")

# A retriever encodes an iterable of question strings into an (n, d) float array.
Retriever = Callable[[Sequence[str]], np.ndarray]


def truncate_text(text: str, characters: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= characters else text[: characters - 1].rstrip() + "…"


def deterministic_permutation(n_options: int, seed: int, row_id: str) -> np.ndarray:
    rng = np.random.default_rng(stable_int(seed, row_id))
    return rng.permutation(n_options)


def option_token_ids(
    tokenizer,
    max_options: int,
    *,
    continuation_prefix: str | None = None,
    label_prefix: str = "",
) -> list[int]:
    """Return the one-token continuations for option labels.

    BPE token IDs depend on the preceding text. For a raw prompt ending in
    ``Answer:``, the continuation is usually ``" A"``; inside a chat template
    the assistant starts a new turn and the continuation is usually ``"A"``.
    Encoding labels in isolation can therefore read the wrong vocabulary
    entries. When ``continuation_prefix`` is supplied, IDs are derived from the
    actual prefix+completion tokenization and the one-token contract is checked.
    """
    token_ids: list[int] = []
    for label in OPTION_LABELS[:max_options]:
        completion = f"{label_prefix}{label}"
        if continuation_prefix is None:
            encoded = tokenizer.encode(completion, add_special_tokens=False)
        else:
            prefix_ids = tokenizer.encode(continuation_prefix, add_special_tokens=False)
            full_ids = tokenizer.encode(continuation_prefix + completion, add_special_tokens=False)
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError("tokenizer changed prefix tokens when appending an option label")
            encoded = full_ids[len(prefix_ids) :]
        if len(encoded) != 1:
            raise ValueError(
                f"Option continuation {completion!r} is not one token for "
                f"{tokenizer.name_or_path}: {encoded}. Use sequence scoring for this checkpoint."
            )
        token_ids.append(int(encoded[0]))
    return token_ids


def semantic_probabilities(label_probabilities: np.ndarray, label_to_semantic: Sequence[int]) -> np.ndarray:
    """Map a distribution over PRESENTED label positions back to semantic option
    order (design §5.7.6 — permutations must be undone before scoring)."""
    semantic = np.zeros_like(label_probabilities)
    for label_index, semantic_index in enumerate(label_to_semantic):
        semantic[int(semantic_index)] = label_probabilities[label_index]
    return semantic


# ---------------------------------------------------------------------------
# History selection (R_q)
# ---------------------------------------------------------------------------


def _cosine_ranking(target_vector: np.ndarray, source_vectors: np.ndarray) -> np.ndarray:
    """Descending cosine-similarity order (indices into source_vectors)."""
    target = target_vector / max(np.linalg.norm(target_vector), 1e-12)
    norms = np.linalg.norm(source_vectors, axis=1)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    sims = (source_vectors @ target) / norms
    # Negate for descending; stable sort keeps input order on exact ties, but the
    # caller passes a text-sorted pool so ties break deterministically by text.
    return np.argsort(-sims, kind="stable")


def select_history(
    target_question: str,
    source_items: Sequence[Mapping[str, Any]],
    k: int,
    retriever: Retriever | None,
    selection: str = "semantic",
    *,
    random_seed: int = 0,
    panel_id: str = "",
) -> list[Mapping[str, Any]]:
    """Pick up to K history items for one (respondent, target) pair.

    `source_items` are the respondent's ELIGIBLE answered items — the caller
    (protocol layer) must have already removed the target item, all evaluation
    targets, and any held-out-fold items, so nothing here can leak. Each item is
    a mapping with at least 'question' and 'answer_text' (and typically
    'question_key', 'answer_index').

    selection:
      - 'semantic': rank by cosine similarity of question text to the target
                    (frozen retriever), take top-K;
      - 'random':   deterministic random order (control);
      - 'all':      every eligible item, capped at K.

    Returned items are ordered most-relevant-first for semantic selection.
    """
    if k <= 0 or not source_items:
        return []
    # Deterministic base order so every mode is reproducible and ties are broken
    # by question text rather than input/DataFrame order.
    pool = sorted(source_items, key=lambda item: str(item.get("question_key", item["question"])))

    if selection == "all":
        return list(pool[:k])
    if selection == "random":
        rng = np.random.default_rng(stable_int(random_seed, panel_id, "history"))
        order = rng.permutation(len(pool))
        return [pool[i] for i in order[:k]]
    if selection == "semantic":
        if retriever is None:
            raise ValueError("semantic history selection requires a retriever")
        texts = [target_question] + [str(item["question"]) for item in pool]
        vectors = np.asarray(retriever(texts), dtype=np.float64)
        if vectors.shape[0] != len(pool) + 1:
            raise ValueError("retriever must return one vector per input question")
        ranking = _cosine_ranking(vectors[0], vectors[1:])
        return [pool[int(i)] for i in ranking[:k]]
    raise ValueError(f"Unknown history selection mode: {selection!r}")


# ---------------------------------------------------------------------------
# Canonical prompt
# ---------------------------------------------------------------------------


def _history_block(history_rows: Sequence[Mapping[str, Any]]) -> str:
    if not history_rows:
        return "No relevant previous responses are available."
    lines = []
    for index, row in enumerate(history_rows, start=1):
        question = truncate_text(row["question"], 160)
        answer = truncate_text(row["answer_text"], 90)
        lines.append(f"{index}. {question} -> {answer}")
    return "\n".join(lines)


def build_respondent_prompt(
    demographic_text: str,
    history_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Query-independent respondent prompt T_r(d_i, H_i^K) (design §2.3.D).

    The target question and its options are REMOVED entirely — this is the
    only structural difference from build_canonical_prompt. Encoding this with
    the input-centric LLM2Vec encoder yields RespondentVec r_i^K, the
    query-independent ablation used only for H9 and the reusable-vector
    transfer tables. Because there is no target here, history is selected with
    selection='all' (deterministic, target-agnostic) upstream — a semantic
    ranking against "the target question" is meaningless when there is none.
    """
    return (
        "Task: Describe this real survey respondent's demographic background "
        "and known attitudes, independent of any specific target question.\n\n"
        f"Respondent background:\n{truncate_text(demographic_text, 600)}\n\n"
        f"Known previous responses:\n{_history_block(history_rows)}"
    )


def build_canonical_prompt(
    row: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    permutation: Sequence[int] | None = None,
    *,
    include_demographic: bool = True,
) -> tuple[str, int, list[int]]:
    """Build the one canonical query-focused prompt for a (respondent, target)
    pair (design §2.1). Returns (prompt, correct_label_index, label_to_semantic).

    `history_rows` is the output of select_history (already leakage-safe and
    ordered). `permutation` maps presented label positions -> semantic option
    index; identity if None. The correct label index is the presented position
    of the respondent's true answer, so the extraction path can read the option
    logit even under permutation. `include_demographic=False` blanks the
    demographic block for the −demographic ablation (design §6.4); the block
    label is retained so the prompt structure is stable and only the content is
    removed, isolating the demographic signal.
    """
    options = json.loads(row["options_json"]) if isinstance(row["options_json"], str) else list(row["options_json"])
    n_options = len(options)
    if n_options > len(OPTION_LABELS):
        raise ValueError(f"{n_options} options exceed supported labels: {row['row_id']}")
    if permutation is None:
        permutation = list(range(n_options))
    permutation = [int(v) for v in permutation]
    if sorted(permutation) != list(range(n_options)):
        raise ValueError("permutation must contain every semantic option exactly once")

    option_lines = [
        f"{OPTION_LABELS[label_index]}. {truncate_text(options[semantic_index], 180)}"
        for label_index, semantic_index in enumerate(permutation)
    ]
    answer_index = int(row["answer_index"])
    correct_label = permutation.index(answer_index)

    demographic_block = truncate_text(row['demographic_text'], 600) if include_demographic else "Withheld for ablation."

    prompt = (
        "Task: Predict how this real survey respondent is likely to answer the target question.\n\n"
        f"Respondent background:\n{demographic_block}\n\n"
        f"Relevant previous responses:\n{_history_block(history_rows)}\n\n"
        f"Target question:\n{truncate_text(row['question'], 600)}\n\n"
        "Options:\n" + "\n".join(option_lines) + "\n\nAnswer:"
    )
    return prompt, correct_label, permutation
