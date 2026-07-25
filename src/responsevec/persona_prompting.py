"""Persona Effect Prompting (Hu & Collier, ACL 2024), ADAPTED to the frozen
doubly-cold persona protocol (Table A).

Reproduces Hu & Collier's test of whether *prompting* an LLM with persona
variables improves prediction of a real respondent's answer, via five prompt-only
variants plus the protocol's information-conserving shuffle control:
  pe_no_persona           target question + options only (persona withheld);
  pe_demographic          + demographic description;
  pe_demographic_behavior + the respondent's assigned behaviour-persona text;
  pe_history              demographic + K calibration (question, answer) lines;
  pe_shuffled_behavior    pe_history structure, answers from the frozen
                          question-aligned shuffle (falsification control).

Leakage guards: history and behaviour-persona text come ONLY from the
respondent's own calibration answers; behaviour text is the frozen persona-bank
``persona_text`` selected by ``assign_personas`` (train-fit); the target answer
never enters any prompt (used only for validation-temperature fit and scoring).

ADAPTED (not EXACT): reader is Qwen3-8B option-token probabilities on SocioBench,
cold-item/cold-respondent, personas from the frozen per-domain bank, no linear
oracle. torch/transformers import lazily (CPU-safe); reader signature matches the
Persona-DB experiment.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from responsevec.persona_router import (DEMO_COLS, assign_personas,
    question_aligned_shuffle, stable_history)
from responsevec.prompting_rv import OPTION_LABELS, truncate_text

# reader signature: (prompts, n_options, label_to_semantic) -> (n, max_opt) probs
OptionProbFn = Callable[[Sequence[str], Sequence[int], Sequence[Sequence[int]]], np.ndarray]

VARIANTS = ("pe_no_persona", "pe_demographic", "pe_demographic_behavior",
            "pe_history", "pe_shuffled_behavior")
# Variants whose prompt carries a demographic / history block, respectively.
_WITH_DEMOGRAPHIC = set(VARIANTS) - {"pe_no_persona"}
_WITH_HISTORY = {"pe_history", "pe_shuffled_behavior"}


def _options_of(row) -> list[str]:
    value = row.options_json
    return json.loads(value) if isinstance(value, str) else list(value)


def demographic_text(row) -> str:
    """Prefer a precomputed ``demographic_text`` column; else render DEMO_COLS."""
    text = getattr(row, "demographic_text", None)
    if isinstance(text, str) and text.strip():
        return text
    parts = [f"{column}={getattr(row, column)}" for column in DEMO_COLS if hasattr(row, column)]
    return "; ".join(parts) if parts else "Unknown."


def build_behavior_persona_index(train_calibration: pd.DataFrame,
                                 persona_bank: Mapping[str, Any]
                                 ) -> dict[tuple[str, str], str]:
    """Map (domain, panel_id) -> behaviour-persona text from the frozen bank.

    Each held-out respondent is routed to the nearest train-fit KMeans centre via
    ``assign_personas`` (own calibration answers only); the frozen per-persona
    ``persona_text`` for that domain is returned. No val/test label ever enters.
    """
    index: dict[tuple[str, str], str] = {}
    for domain, domain_bank in persona_bank["domains"].items():
        rows = train_calibration[train_calibration.domain.eq(domain)]
        if rows.empty:
            continue
        assignments = assign_personas(rows, domain_bank)
        texts = domain_bank.get("persona_text", {})
        for panel_id, persona in assignments.items():
            index[(str(domain), str(panel_id))] = str(texts.get(str(int(persona)), ""))
    return index


def _calibration_history(panel_id: str, domain: str, k: int, seed: int,
                         calibration_by_domain: Mapping[str, Sequence[str]],
                         answer_lookup: Mapping[tuple[str, str], int]) -> list[tuple[str, int]]:
    """Deterministic K calibration (question_key, answer_index) for one
    respondent, restricted to their own split's calibration answers."""
    questions = stable_history(str(panel_id),
                               list(calibration_by_domain.get(str(domain), ())), k, seed)
    history: list[tuple[str, int]] = []
    for question in questions:
        answer = answer_lookup.get((str(panel_id), str(question)))
        if answer is not None:
            history.append((str(question), int(answer)))
    return history


def build_shuffled_history(target: pd.DataFrame, k: int, seed: int,
                           calibration_by_domain: Mapping[str, Sequence[str]]
                           ) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """Question-aligned shuffle mirroring ``pe_history`` structure exactly.

    For every (respondent, domain) in ``target`` take the SAME K calibration
    question keys ``pe_history`` uses, then replace each real answer with a
    demographically matched donor's answer via the frozen
    ``question_aligned_shuffle``. Question keys and demographic strata are
    preserved; only the respondent-answer correspondence breaks. Keyed by
    ``(panel_id, domain)`` so scoring aligns it per target row.
    """
    if target.empty:
        return {}
    histories: dict[str, list[str]] = {}
    domain_of: dict[tuple[str, str], list[str]] = {}
    for row in target.drop_duplicates(["panel_id", "domain"]).itertuples(index=False):
        questions = stable_history(str(row.panel_id),
                                   list(calibration_by_domain.get(str(row.domain), ())),
                                   k, seed)
        domain_of[(str(row.panel_id), str(row.domain))] = questions
        histories.setdefault(str(row.panel_id), [])
        histories[str(row.panel_id)] += questions
    shuffled = question_aligned_shuffle(target, histories, seed)
    answer_of = {(pid, q): a for pid, pairs in shuffled.items() for q, a in pairs}
    out: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for (panel_id, domain), questions in domain_of.items():
        pairs = [(q, answer_of[(panel_id, q)]) for q in questions
                 if (panel_id, q) in answer_of]
        out[(panel_id, domain)] = pairs
    return out


def _history_lines(history: Sequence[tuple[str, int]],
                   question_text: Mapping[str, str],
                   options_by_question: Mapping[str, Sequence[str]]) -> str:
    if not history:
        return "None"
    lines = []
    for index, (question_key, answer) in enumerate(history, start=1):
        question = truncate_text(question_text.get(question_key, question_key), 160)
        options = options_by_question.get(question_key, [])
        answer_text = truncate_text(str(options[answer]) if 0 <= answer < len(options)
                                    else str(answer), 90)
        lines.append(f"{index}. {question} -> {answer_text}")
    return "\n".join(lines)


def build_prompt(row, options: Sequence[str], variant: str,
                 label_to_semantic: Sequence[int], *,
                 persona_text: str = "", history_block: str = "None") -> str:
    """Canonical persona-effect prompt (matches the frozen LLM prompt structure).
    ``label_to_semantic[label]`` = semantic option index (identity in the primary
    protocol). The target answer is never written into the prompt."""
    demographic = demographic_text(row) if variant in _WITH_DEMOGRAPHIC else "Withheld."
    persona_block = ""
    if variant == "pe_demographic_behavior" and persona_text:
        persona_block = "Persona tendencies:\n" + truncate_text(persona_text, 600) + "\n\n"
    history = history_block if variant in _WITH_HISTORY else "None"
    option_lines = [f"{OPTION_LABELS[label_index]}. {truncate_text(str(options[int(semantic)]), 180)}"
                    for label_index, semantic in enumerate(label_to_semantic)]
    return (
        "Task: Predict how this real survey respondent is likely to answer the "
        "target question. Return exactly one option letter.\n\n"
        f"Respondent background:\n{truncate_text(demographic, 600)}\n\n"
        f"{persona_block}"
        f"Relevant previous responses:\n{history}\n\n"
        f"Target question:\n{truncate_text(str(row.question), 600)}\n\n"
        "Options:\n" + "\n".join(option_lines) + "\n\nAnswer:"
    )


def _label_to_semantic(n) -> list[int]:
    return list(range(int(n)))


def build_variant_prompts(target: pd.DataFrame, variant: str, *, k: int, seed: int,
                          calibration_by_domain: Mapping[str, Sequence[str]],
                          answer_lookup: Mapping[tuple[str, str], int],
                          question_text: Mapping[str, str],
                          options_by_question: Mapping[str, Sequence[str]],
                          behavior_index: Mapping[tuple[str, str], str],
                          shuffled_history=None) -> tuple[list[str], list[int], list[list[int]]]:
    """Build (prompts, n_options, permutations) for one variant over ``target``.

    ``shuffled_history`` is keyed by ``(panel_id, domain)`` so the falsification
    control aligns per row (see ``build_shuffled_history``).
    """
    prompts, n_options, permutations = [], [], []
    for row in target.itertuples(index=False):
        permutation = _label_to_semantic(row.n_options)
        history_block = "None"
        if variant == "pe_history":
            history = _calibration_history(row.panel_id, row.domain, k, seed,
                                            calibration_by_domain, answer_lookup)
            history_block = _history_lines(history, question_text, options_by_question)
        elif variant == "pe_shuffled_behavior":
            history = list((shuffled_history or {}).get(
                (str(row.panel_id), str(row.domain)), []))
            history_block = _history_lines(history, question_text, options_by_question)
        persona_text = behavior_index.get((str(row.domain), str(row.panel_id)), "") \
            if variant == "pe_demographic_behavior" else ""
        prompts.append(build_prompt(row, _options_of(row), variant, permutation,
                                    persona_text=persona_text, history_block=history_block))
        n_options.append(int(row.n_options))
        permutations.append(permutation)
    return prompts, n_options, permutations


def score_variant(target: pd.DataFrame, variant: str, reader: OptionProbFn,
                  **kwargs) -> list[np.ndarray]:
    """Return one semantic-order probability vector per row of ``target``;
    ``kwargs`` are forwarded to ``build_variant_prompts``."""
    prompts, n_options, permutations = build_variant_prompts(target, variant, **kwargs)
    probabilities = reader(prompts, n_options, permutations)
    return [np.asarray(probabilities[i], np.float32) for i in range(len(target))]


def make_causal_reader(model_name: str, seed: int = 1701, max_length: int = 512,
                       batch_size: int = 16, quantization: str | None = None) -> OptionProbFn:
    """Frozen Qwen3 reader (thinking disabled). Reuses the shared CausalExtractor
    exactly as the Persona-DB baseline; torch/transformers import lazily here."""
    from responsevec.personadb import make_causal_reader as _make
    return _make(model_name, seed, max_length, batch_size, quantization)
