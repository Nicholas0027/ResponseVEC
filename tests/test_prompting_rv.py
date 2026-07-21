from __future__ import annotations

import json

import numpy as np
import pytest

from responsevec.prompting_rv import (
    build_canonical_prompt,
    deterministic_permutation,
    select_history,
    semantic_probabilities,
)


def _row(answer_index=2, n_options=5):
    return {
        "row_id": "D::p1::q_target",
        "question": "How concerned are you about climate change?",
        "options_json": json.dumps([f"Option {i}" for i in range(n_options)]),
        "n_options": n_options,
        "answer_index": answer_index,
        "demographic_text": "Country: A; Age group: 30-44; Gender: Female",
    }


def _sources():
    return [
        {"question_key": "D::q_env", "question": "climate change concern level", "answer_text": "Very concerned", "answer_index": 4},
        {"question_key": "D::q_tax", "question": "income tax fairness opinion", "answer_text": "Unfair", "answer_index": 1},
        {"question_key": "D::q_job", "question": "job satisfaction rating", "answer_text": "Satisfied", "answer_index": 3},
    ]


def _keyword_retriever(texts):
    """Deterministic fake retriever: vector = counts of a few marker tokens.
    'climate'/'change' load dim 0 so the env question is nearest the target."""
    markers = ["climate", "change", "tax", "job"]
    vecs = []
    for text in texts:
        lowered = text.lower()
        vecs.append([float(lowered.count(m)) for m in markers])
    return np.asarray(vecs, dtype=np.float64)


# --- canonical prompt -------------------------------------------------------


def test_prompt_contains_all_sections_and_history():
    history = _sources()[:2]
    prompt, correct_label, permutation = build_canonical_prompt(_row(), history)
    assert "Respondent background:" in prompt
    assert "Relevant previous responses:" in prompt
    assert "Target question:" in prompt
    assert "Options:" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "climate change concern level" in prompt  # a history question rendered
    assert permutation == [0, 1, 2, 3, 4]
    assert correct_label == 2  # identity permutation -> label == semantic index


def test_empty_history_renders_placeholder():
    prompt, _, _ = build_canonical_prompt(_row(), [])
    assert "No relevant previous responses are available." in prompt


def test_permutation_moves_correct_label_but_semantic_recovers():
    permutation = [4, 3, 2, 1, 0]  # reverse
    prompt, correct_label, perm = build_canonical_prompt(_row(answer_index=0), [], permutation)
    # semantic answer 0 is presented last -> label position 4
    assert correct_label == 4
    # a distribution over presented labels maps back to semantic order
    label_probs = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    semantic = semantic_probabilities(label_probs, perm)
    # presented label 4 (prob 0.5) corresponds to semantic option 0
    assert semantic[0] == pytest.approx(0.5)
    assert semantic.sum() == pytest.approx(1.0)


def test_permutation_must_be_complete():
    with pytest.raises(ValueError, match="every semantic option"):
        build_canonical_prompt(_row(), [], permutation=[0, 1, 2])  # missing 3,4


def test_deterministic_permutation_is_stable():
    a = deterministic_permutation(5, seed=7, row_id="r1")
    b = deterministic_permutation(5, seed=7, row_id="r1")
    np.testing.assert_array_equal(a, b)
    assert sorted(a.tolist()) == [0, 1, 2, 3, 4]


# --- history selection ------------------------------------------------------


def test_semantic_selection_ranks_by_similarity_to_target():
    target = "How concerned are you about climate change?"
    picked = select_history(target, _sources(), k=1, retriever=_keyword_retriever, selection="semantic")
    assert len(picked) == 1
    assert picked[0]["question_key"] == "D::q_env"  # climate/change nearest


def test_semantic_selection_orders_most_relevant_first():
    target = "How concerned are you about climate change?"
    picked = select_history(target, _sources(), k=3, retriever=_keyword_retriever, selection="semantic")
    assert picked[0]["question_key"] == "D::q_env"
    assert len(picked) == 3


def test_random_selection_is_deterministic_and_capped():
    a = select_history("t", _sources(), k=2, retriever=None, selection="random", random_seed=1, panel_id="p1")
    b = select_history("t", _sources(), k=2, retriever=None, selection="random", random_seed=1, panel_id="p1")
    assert [x["question_key"] for x in a] == [x["question_key"] for x in b]
    assert len(a) == 2


def test_all_selection_returns_pool_capped_at_k():
    picked = select_history("t", _sources(), k=2, retriever=None, selection="all")
    assert len(picked) == 2
    full = select_history("t", _sources(), k=10, retriever=None, selection="all")
    assert len(full) == 3  # only three eligible items exist


def test_k_zero_returns_empty():
    assert select_history("t", _sources(), k=0, retriever=_keyword_retriever) == []


def test_semantic_requires_retriever():
    with pytest.raises(ValueError, match="retriever"):
        select_history("t", _sources(), k=1, retriever=None, selection="semantic")
