"""Persona-DB (Sun et al., COLING 2025), ADAPTED to the frozen doubly-cold
persona protocol.

This is an ADAPTED (not EXACT) reimplementation. Differences from the paper:
  * persona records are turned into hierarchical "persona keys" by deterministic
    rules/templates, NOT by an LLM extractor (reproducibility + no target leak);
  * the downstream reader is the frozen Qwen3-8B CausalExtractor (option-token
    probabilities), not gpt-3.5;
  * evaluation is cold-item and cold-respondent, not the paper's warm setting.

Leakage guards baked in here:
  * self persona DB uses ONLY the respondent's own calibration answers;
  * the collaborative DB is built from TRAIN respondents only, so held-out
    respondents never contribute evidence to each other;
  * query-focused retrieval ranks evidence by target OPTION vectors only and
    never reads the target answer label;
  * JOIN backfill fires only when self evidence is sparse.

torch/transformers are imported lazily inside the real-reader factory so the CPU
smoke path and unit tests never require a GPU or the model weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from responsevec.persona_router import DEMO_COLS, stable_history

# reader signature: (prompts, n_options, label_to_semantic) -> (n, max_opt) probs
OptionProbFn = Callable[[Sequence[str], Sequence[int], Sequence[Sequence[int]]], np.ndarray]

VARIANTS = (
    "personadb_history_full",
    "personadb_history_retrieval",
    "personadb_intsum",
    "personadb_nojoin",
    "personadb_full",
)


@dataclass
class PersonaRecord:
    """One structured (question, chosen option) fact from a calibration answer."""
    panel_id: str
    domain: str
    question_key: str
    question: str
    answer_index: int
    answer_text: str
    option_vector: np.ndarray  # centroid of the chosen option's vector


@dataclass
class SelfPersonaDB:
    """Per-respondent self database plus deterministic hierarchical keys."""
    panel_id: str
    records: list[PersonaRecord] = field(default_factory=list)
    persona_keys: dict[str, str] = field(default_factory=dict)  # domain -> summary
    profile: np.ndarray | None = None  # mean chosen-option vector (retrieval key)


def _options_of(row) -> list[str]:
    import json
    value = row.options_json
    return json.loads(value) if isinstance(value, str) else list(value)


def build_self_databases(calibration_responses: pd.DataFrame,
                         option_vectors: Mapping[str, np.ndarray],
                         calibration_by_domain: Mapping[str, Sequence[str]]
                         ) -> dict[str, SelfPersonaDB]:
    """Structured self persona DB for every respondent, calibration answers only.

    `calibration_responses` MUST already be restricted to the same respondent
    split; this function additionally keeps only rows whose question is a frozen
    calibration item so no target answer can ever enter a persona record.
    """
    allowed = {str(q) for questions in calibration_by_domain.values() for q in questions}
    databases: dict[str, SelfPersonaDB] = {}
    for row in calibration_responses.itertuples(index=False):
        question_key = str(row.question_key)
        if question_key not in allowed or question_key not in option_vectors:
            continue
        answer = int(row.answer_index)
        options = _options_of(row)
        vectors = np.asarray(option_vectors[question_key], np.float32)
        if answer < 0 or answer >= len(options) or answer >= len(vectors):
            raise ValueError(f"invalid calibration answer for {question_key}")
        panel_id = str(row.panel_id)
        db = databases.setdefault(panel_id, SelfPersonaDB(panel_id))
        db.records.append(PersonaRecord(panel_id, str(row.domain), question_key,
                                        str(row.question), answer, str(options[answer]),
                                        vectors[answer].astype(np.float32)))
    for db in databases.values():
        db.persona_keys = _hierarchical_persona_keys(db.records)
        stack = [record.option_vector for record in db.records]
        db.profile = np.mean(stack, axis=0).astype(np.float32) if stack else None
    return databases


def _hierarchical_persona_keys(records: Sequence[PersonaRecord]) -> dict[str, str]:
    """Deterministic template summary of value tendencies, grouped by domain.

    The hierarchy is domain -> (most consistent question-level tendencies). No
    LLM is used; the text is a reproducible rule-based rendering of the answers.
    """
    keys: dict[str, str] = {}
    by_domain: dict[str, list[PersonaRecord]] = {}
    for record in records:
        by_domain.setdefault(record.domain, []).append(record)
    for domain, domain_records in sorted(by_domain.items()):
        ordered = sorted(domain_records, key=lambda r: r.question_key)[:5]
        tendencies = "; ".join(f"{_short(r.question, 70)} -> {_short(r.answer_text, 40)}"
                               for r in ordered)
        keys[domain] = f"[{domain}] tendencies (calibration only): {tendencies}"
    return keys


def _short(text: str, characters: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= characters else text[: characters - 1].rstrip() + "~"


def build_collaborative_index(train_self_dbs: Mapping[str, SelfPersonaDB]
                              ) -> dict[str, Any]:
    """Collaborative DB built from TRAIN respondents only.

    Stores each train respondent's profile vector (for neighbor search) and their
    self persona records (the collaborative evidence pool). Held-out respondents
    are never inserted, so they cannot become each other's neighbors.
    """
    panels = sorted(p for p, db in train_self_dbs.items() if db.profile is not None)
    if not panels:
        return {"panels": [], "profiles": np.zeros((0, 0), np.float32), "records": {}}
    profiles = np.vstack([train_self_dbs[p].profile for p in panels]).astype(np.float32)
    records = {p: list(train_self_dbs[p].records) for p in panels}
    return {"panels": panels, "profiles": profiles, "records": records}


def _cosine(target: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), np.float32)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    return (matrix @ target) / norms


def collaborative_neighbors(query_profile: np.ndarray, index: Mapping[str, Any],
                            top_neighbors: int, exclude: str = "") -> list[str]:
    """Top-N train neighbors by profile cosine. `exclude` drops a self-match so a
    train respondent never retrieves itself as its own neighbor."""
    if query_profile is None or not index["panels"]:
        return []
    position = {p: i for i, p in enumerate(index["panels"])}
    panels = [p for p in index["panels"] if p != str(exclude)]
    if not panels:
        return []
    profiles = np.vstack([index["profiles"][position[p]] for p in panels]).astype(np.float32)
    sims = _cosine(np.asarray(query_profile, np.float32), profiles)
    order = np.argsort(-sims, kind="stable")
    return [panels[int(i)] for i in order[: max(0, int(top_neighbors))]]


def query_focused_retrieval(target_vectors: np.ndarray, records: Sequence[PersonaRecord],
                            top_evidence: int) -> list[PersonaRecord]:
    """Rank persona records by cosine of their chosen-option vector to the target
    OPTION centroid, take top-k. Uses only option text vectors; the target answer
    label is never read."""
    if not records or top_evidence <= 0:
        return []
    pool = sorted(records, key=lambda r: (r.panel_id, r.question_key))
    target_centroid = np.asarray(target_vectors, np.float32).mean(axis=0)
    matrix = np.vstack([r.option_vector for r in pool]).astype(np.float32)
    sims = _cosine(target_centroid, matrix)
    order = np.argsort(-sims, kind="stable")
    return [pool[int(i)] for i in order[: int(top_evidence)]]


def assemble_evidence(self_db: SelfPersonaDB, index: Mapping[str, Any],
                      target_vectors: np.ndarray, variant: str, top_neighbors: int,
                      top_evidence: int, join_threshold: int
                      ) -> tuple[list[PersonaRecord], list[str], bool]:
    """Return (evidence records, persona-key summaries, join_fired) for a variant.

    History-Full : all self records (no retrieval, no collaborative).
    History-Retr.: query-focused self records only.
    IntSum       : self hierarchical summaries only (no per-record evidence).
    w/o JOIN     : self retrieval + persona keys, no collaborative backfill.
    Full         : self retrieval + persona keys + collaborative JOIN when sparse.
    """
    self_records = list(self_db.records) if self_db is not None else []
    keys = dict(self_db.persona_keys) if self_db is not None else {}
    if variant == "personadb_history_full":
        return self_records, {}, False
    if variant == "personadb_intsum":
        return [], keys, False
    retrieved = query_focused_retrieval(target_vectors, self_records, top_evidence)
    if variant == "personadb_history_retrieval":
        return retrieved, {}, False
    if variant == "personadb_nojoin":
        return retrieved, keys, False
    if variant == "personadb_full":
        join_fired = len(retrieved) < int(join_threshold)
        if join_fired and self_db is not None and self_db.profile is not None:
            neighbors = collaborative_neighbors(self_db.profile, index, top_neighbors,
                                                exclude=self_db.panel_id)
            pool: list[PersonaRecord] = []
            for neighbor in neighbors:
                pool.extend(index["records"].get(neighbor, ()))
            backfill = query_focused_retrieval(target_vectors, pool, top_evidence - len(retrieved))
            retrieved = retrieved + backfill
        return retrieved, keys, join_fired
    raise ValueError(f"unknown Persona-DB variant: {variant!r}")


def _demographic_line(row) -> str:
    parts = [f"{column}={getattr(row, column)}" for column in DEMO_COLS if hasattr(row, column)]
    return "Respondent profile: " + ", ".join(parts) if parts else "Respondent profile: unknown"


def build_prompt(row, options: Sequence[str], evidence: Sequence[PersonaRecord],
                 persona_keys: Mapping[str, str], label_to_semantic: Sequence[int]) -> str:
    """Persona-conditioned reader prompt: demographics + persona DB evidence +
    target question + presented options + 'Answer:'."""
    lines = [_demographic_line(row)]
    if persona_keys:
        lines.append("Persona database (hierarchical keys):")
        lines += [f"- {persona_keys[domain]}" for domain in sorted(persona_keys)]
    lines.append("Retrieved persona evidence:")
    if evidence:
        for position, record in enumerate(evidence, start=1):
            lines.append(f"{position}. {_short(record.question, 160)} -> "
                         f"{_short(record.answer_text, 90)}")
    else:
        lines.append("No relevant persona evidence is available.")
    lines.append("")
    lines.append(f"Question: {_short(str(row.question), 300)}")
    from responsevec.prompting_rv import OPTION_LABELS
    for label_index, semantic_index in enumerate(label_to_semantic):
        lines.append(f"{OPTION_LABELS[label_index]}. {_short(str(options[int(semantic_index)]), 120)}")
    lines.append("Answer with the single letter of your choice.")
    lines.append("Answer:")
    return "\n".join(lines)


def make_causal_reader(model_name: str, seed: int = 1701, max_length: int = 512,
                       batch_size: int = 16, quantization: str | None = None) -> OptionProbFn:
    """Real causal reader: option-token probabilities with thinking disabled.

    torch/transformers are imported here so the CPU path never needs them.
    ``quantization`` is passed through to ``load_causal_backbone`` (e.g. "nf4"
    for large backbones that do not fit in bf16 on a single card).
    """
    from responsevec.llm_rv import CausalExtractor, choose_device, load_causal_backbone

    model, tokenizer = load_causal_backbone(model_name, dtype="bfloat16",
                                            quantization=quantization)
    extractor = CausalExtractor(model, tokenizer, choose_device(), max_length=max_length,
                                batch_size=batch_size, enable_thinking=False)

    def option_prob_fn(prompts, n_options, label_to_semantic):
        result = extractor.extract(list(prompts), list(n_options), list(label_to_semantic))
        return np.asarray(result["probabilities"], np.float32)

    return option_prob_fn
