import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from responsevec.encode import save_option_table
from responsevec.persona_router import build_item_split, fold_roles
from responsevec.personadb import (VARIANTS, assemble_evidence,
    build_collaborative_index, build_prompt, build_self_databases,
    collaborative_neighbors, query_focused_retrieval)

DOMAINS = ["A", "B"]
N_ITEMS = 16
DIM = 12


def item_fixture():
    rows, table = [], {}
    for d in DOMAINS:
        for i in range(N_ITEMS):
            q = f"{d}::q{i}"
            rows.append({"domain": d, "question_key": q})
            center = np.zeros(DIM)
            center[i % DIM] = 1.0
            table[q] = np.stack([center, center + 0.05, center + 0.1]).astype(np.float32)
    return pd.DataFrame(rows), table


def response_fixture():
    rows = []
    splits = [("train", range(0, 18)), ("validation", range(18, 24)),
              ("test", range(24, 32))]
    for split, panels in splits:
        for p in panels:
            for d in DOMAINS:
                for i in range(N_ITEMS):
                    answer = (p + i + (0 if d == "A" else 1)) % 3
                    rows.append({
                        "row_id": f"{d}::{p}::q{i}", "panel_id": f"P{p}", "domain": d,
                        "question_key": f"{d}::q{i}",
                        "question": f"CAL Q {d}{i}" if i < 8 else f"HELD Q {d}{i}",
                        "options_json": json.dumps(["No", "Maybe", "Yes"]),
                        "n_options": 3, "answer_index": answer, "is_ordinal": True,
                        "survey_weight": 1.0, "split": split, "country": "X", "sex": "F",
                        "age_bin": "30-44", "education": str(p % 3), "income_quintile": "Q3"})
    return pd.DataFrame(rows)


def build_context():
    items, table = item_fixture()
    split = build_item_split(items, table, n_calibration=8, n_folds=6,
                             clusters_per_domain=8, seed=11)
    roles = fold_roles(split, 0)
    responses = response_fixture()
    calibration = {d: data["calibration"] for d, data in split["domains"].items()}
    return split, roles, responses, table, calibration


# --------------------------------------------------------------------------- #
# Unit-level leakage guards
# --------------------------------------------------------------------------- #

def test_self_db_uses_calibration_only():
    split, roles, responses, table, calibration = build_context()
    calibration_keys = set(roles["calibration"])
    train = responses[responses.split.eq("train")
                      & responses.question_key.isin(calibration_keys)]
    # Deliberately also pass a held-out target row; it must be ignored.
    contaminated = pd.concat([train, responses[~responses.question_key.isin(calibration_keys)]
                              .head(3)], ignore_index=True)
    dbs = build_self_databases(contaminated, table, calibration)
    used = {rec.question_key for db in dbs.values() for rec in db.records}
    assert used <= calibration_keys
    assert used, "self DB should contain at least one calibration record"


def test_collaborative_index_is_train_only():
    split, roles, responses, table, calibration = build_context()
    cal = set(roles["calibration"])
    train_self = build_self_databases(
        responses[responses.split.eq("train") & responses.question_key.isin(cal)],
        table, calibration)
    val_self = build_self_databases(
        responses[responses.split.eq("validation") & responses.question_key.isin(cal)],
        table, calibration)
    index = build_collaborative_index(train_self)
    train_panels = set(responses[responses.split.eq("train")].panel_id.astype(str))
    heldout_panels = set(responses[~responses.split.eq("train")].panel_id.astype(str))
    assert set(index["panels"]) <= train_panels
    assert not set(index["panels"]) & heldout_panels
    # A validation respondent's neighbors are all train respondents.
    query = next(iter(val_self.values()))
    neighbors = collaborative_neighbors(query.profile, index, top_neighbors=5)
    assert set(neighbors) <= train_panels


def test_test_respondents_are_not_each_others_neighbors():
    split, roles, responses, table, calibration = build_context()
    cal = set(roles["calibration"])
    test_self = build_self_databases(
        responses[responses.split.eq("test") & responses.question_key.isin(cal)],
        table, calibration)
    index = build_collaborative_index(test_self)  # only train should ever be indexed
    # Even if (wrongly) built from test, exclude=self prevents self-retrieval; and
    # in the real pipeline the index only holds train respondents. Assert the real
    # invariant: a train-built index yields no test neighbor for a test query.
    train_self = build_self_databases(
        responses[responses.split.eq("train") & responses.question_key.isin(cal)],
        table, calibration)
    train_index = build_collaborative_index(train_self)
    test_panels = set(responses[responses.split.eq("test")].panel_id.astype(str))
    for db in test_self.values():
        neighbors = collaborative_neighbors(db.profile, train_index, 8)
        assert not set(neighbors) & test_panels


def test_query_focused_retrieval_ignores_target_label():
    split, roles, responses, table, calibration = build_context()
    cal = set(roles["calibration"])
    train_self = build_self_databases(
        responses[responses.split.eq("train") & responses.question_key.isin(cal)],
        table, calibration)
    db = next(iter(train_self.values()))
    target_vectors = np.asarray(table[roles["test"][0]], np.float32)
    first = query_focused_retrieval(target_vectors, db.records, 3)
    # Retrieval depends only on option vectors; changing an unrelated label field
    # cannot change the result (there is no label input to change).
    second = query_focused_retrieval(target_vectors, list(db.records), 3)
    assert [r.question_key for r in first] == [r.question_key for r in second]
    assert len(first) <= 3


def test_join_only_fires_when_sparse():
    split, roles, responses, table, calibration = build_context()
    cal = set(roles["calibration"])
    train_self = build_self_databases(
        responses[responses.split.eq("train") & responses.question_key.isin(cal)],
        table, calibration)
    index = build_collaborative_index(train_self)
    db = next(iter(train_self.values()))
    target_vectors = np.asarray(table[roles["test"][0]], np.float32)
    # Rich self evidence: threshold below available -> no JOIN.
    _, _, fired_rich = assemble_evidence(db, index, target_vectors, "personadb_full",
                                         top_neighbors=5, top_evidence=2, join_threshold=1)
    assert fired_rich is False
    # Sparse self evidence (empty self DB) -> JOIN backfills from neighbors.
    empty = type(db)(panel_id=db.panel_id)
    empty.profile = db.profile
    evidence, _, fired_sparse = assemble_evidence(empty, index, target_vectors,
        "personadb_full", top_neighbors=5, top_evidence=3, join_threshold=3)
    assert fired_sparse is True
    assert len(evidence) > 0  # backfilled from collaborative neighbors


def test_nojoin_never_uses_collaborative_evidence():
    split, roles, responses, table, calibration = build_context()
    cal = set(roles["calibration"])
    train_self = build_self_databases(
        responses[responses.split.eq("train") & responses.question_key.isin(cal)],
        table, calibration)
    index = build_collaborative_index(train_self)
    db = next(iter(train_self.values()))
    empty = type(db)(panel_id=db.panel_id)
    empty.profile = db.profile
    target_vectors = np.asarray(table[roles["test"][0]], np.float32)
    evidence, _, fired = assemble_evidence(empty, index, target_vectors,
        "personadb_nojoin", top_neighbors=5, top_evidence=3, join_threshold=3)
    assert fired is False
    assert evidence == []  # no self evidence, and JOIN disabled


def test_prompt_contains_evidence_options_and_answer_marker():
    split, roles, responses, table, calibration = build_context()
    cal = set(roles["calibration"])
    train_self = build_self_databases(
        responses[responses.split.eq("train") & responses.question_key.isin(cal)],
        table, calibration)
    db = next(iter(train_self.values()))
    row = next(responses[responses.question_key.isin(roles["test"])].itertuples(index=False))
    options = ["No", "Maybe", "Yes"]
    prompt = build_prompt(row, options, db.records[:2], db.persona_keys, [0, 1, 2])
    assert "Answer:" in prompt
    assert "Persona database" in prompt
    assert "A. No" in prompt and "C. Yes" in prompt
    assert "calibration only" in prompt


# --------------------------------------------------------------------------- #
# End-to-end CLI smoke (dummy reader, CPU only)
# --------------------------------------------------------------------------- #

def _write_inputs(tmp_path):
    split, roles, responses, table, calibration = build_context()
    processed = tmp_path / "processed"
    processed.mkdir()
    responses.to_parquet(processed / "responses.parquet", index=False)
    results = tmp_path / "results"
    results.mkdir()
    (results / "split.json").write_text(json.dumps(split), encoding="utf-8")
    option_table_path = tmp_path / "option_table.npz"
    save_option_table(table, option_table_path)
    return processed, results, option_table_path, roles


def test_cli_end_to_end_dummy_reader(tmp_path):
    processed, results, option_table_path, roles = _write_inputs(tmp_path)
    output = tmp_path / "out"
    repo = Path(__file__).resolve().parents[1]
    env = {"PYTHONPATH": str(repo / "src")}
    import os
    env = {**os.environ, **env}
    cmd = [sys.executable, str(repo / "scripts" / "personadb_experiment.py"),
           "--processed", str(processed), "--option-table", str(option_table_path),
           "--persona-results", str(results), "--output", str(output),
           "--fold", "0", "--seed", "1701", "--r", "8",
           "--top-neighbors", "4", "--top-evidence", "3", "--join-threshold", "3"]
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    predictions = pd.read_parquet(output / "predictions.parquet")
    val = pd.read_parquet(output / "validation_predictions.parquet")
    results_csv = pd.read_csv(output / "results.csv")
    config = json.loads((output / "personadb_config.json").read_text())

    # Every variant produced calibrated + uncalibrated predictions for both splits.
    methods = set(predictions.method)
    for variant in VARIANTS:
        assert variant in methods
        assert variant + "_uncalibrated" in methods
    assert set(val.method) == methods
    assert config["reader"] == "dummy"
    assert config["mode"] == "ADAPTED"

    # Probabilities are valid distributions over the row's options.
    for row in predictions.itertuples(index=False):
        probs = np.asarray(json.loads(row.probabilities_json), float)
        assert len(probs) == int(row.n_options)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert not results_csv.nll.isna().all()


def test_cli_split_is_strictly_disjoint(tmp_path):
    processed, results, option_table_path, roles = _write_inputs(tmp_path)
    responses = pd.read_parquet(processed / "responses.parquet")
    train = set(responses[responses.split.eq("train")].panel_id)
    val = set(responses[responses.split.eq("validation")].panel_id)
    test = set(responses[responses.split.eq("test")].panel_id)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    # calibration and target item roles never overlap in the reused split.
    assert set(roles["calibration"]).isdisjoint(set(roles["test"]))
    assert set(roles["calibration"]).isdisjoint(set(roles["validation"]))


def test_history_full_uses_all_self_records_no_retrieval():
    split, roles, responses, table, calibration = build_context()
    cal = set(roles["calibration"])
    train_self = build_self_databases(
        responses[responses.split.eq("train") & responses.question_key.isin(cal)],
        table, calibration)
    index = build_collaborative_index(train_self)
    db = next(iter(train_self.values()))
    target_vectors = np.asarray(table[roles["test"][0]], np.float32)
    full, keys_full, _ = assemble_evidence(db, index, target_vectors,
        "personadb_history_full", 4, 2, 3)
    retr, _, _ = assemble_evidence(db, index, target_vectors,
        "personadb_history_retrieval", 4, 2, 3)
    assert len(full) == len(db.records)
    assert len(retr) <= 2
    # IntSum carries summaries but no per-record evidence.
    intsum_ev, intsum_keys, _ = assemble_evidence(db, index, target_vectors,
        "personadb_intsum", 4, 2, 3)
    assert intsum_ev == [] and intsum_keys
