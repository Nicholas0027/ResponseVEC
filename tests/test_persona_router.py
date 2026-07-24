import json

import numpy as np
import pandas as pd

from responsevec.persona_router import (build_item_split, fit_persona_bank,
    fold_roles, question_aligned_shuffle, update_posterior)


def item_fixture():
    rows, table = [], {}
    for d in ["A", "B", "C", "D"]:
        for i in range(20):
            q = f"{d}::q{i}"
            rows.append({"domain": d, "question_key": q})
            center = np.zeros(12); center[i % 12] = 1
            table[q] = np.stack([center, center + 0.01])
    return pd.DataFrame(rows), table


def response_fixture():
    rows = []
    for split, panels in [("train", range(12)), ("validation", range(12, 16)), ("test", range(16, 22))]:
        for p in panels:
            for i in range(12):
                answer = (p % 4 + i * (1 + p % 2)) % 3
                rows.append({"row_id": f"A::{p}::q{i}", "panel_id": f"A::{p}", "domain": "A",
                    "question_key": f"A::q{i}", "question": f"CAL QUESTION {i}" if i < 8 else f"HELD OUT {i}",
                    "options_json": json.dumps(["No", "Maybe", "Yes"]), "n_options": 3,
                    "answer_index": answer, "split": split, "country": "X", "sex": "F",
                    "age_bin": "30-44", "education": str(p % 3), "income_quintile": "Q3"})
    return pd.DataFrame(rows)


def test_item_split_is_cluster_disjoint_and_domain_balanced():
    items, table = item_fixture()
    split = build_item_split(items, table, seed=11)
    roles = fold_roles(split, 2)
    for domain, data in split["domains"].items():
        cal = set(data["calibration"]); target = set().union(*map(set, data["target_folds"]))
        assert not cal & target
        for cluster in data["semantic_clusters"].values():
            assert not (set(cluster) & cal and set(cluster) & target)
        for role in ("train", "validation", "test"):
            assert any(q.startswith(domain + "::") for q in roles[role])


def test_posterior_normalizes_and_k0_equals_prior():
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    probs = {str(z): {"q": [0.1 + z * 0.1, 0.9 - z * 0.1]} for z in range(4)}
    assert np.allclose(update_posterior(prior, [], probs), prior)
    post = update_posterior(prior, [("q", 0)], probs)
    assert np.isclose(post.sum(), 1.0)
    assert np.all(post >= 0)


def test_question_aligned_shuffle_changes_answers_without_changing_questions():
    frame = response_fixture(); test = frame[frame.split.eq("test")]
    histories = {p: ["A::q0", "A::q1", "A::q2"] for p in test.panel_id.unique()}
    shuffled = question_aligned_shuffle(test, histories, seed=3)
    real = {(r.panel_id, r.question_key): r.answer_index for r in test.itertuples()}
    assert all([q for q, _ in shuffled[p]] == histories[p] for p in histories)
    assert any(a != real[(p, q)] for p, hist in shuffled.items() for q, a in hist)


def test_persona_text_uses_calibration_items_only():
    frame = response_fixture(); train = frame[frame.split.eq("train")]
    bank = fit_persona_bank(train, [f"A::q{i}" for i in range(8)], seed=7)
    text = " ".join(bank["domains"]["A"]["persona_text"].values())
    assert "CAL QUESTION" in text
    assert "HELD OUT" not in text
    assert all(f"q{i}" not in bank["domains"]["A"]["response_prob"]["0"] for i in range(8, 12))
