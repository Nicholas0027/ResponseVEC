import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from responsevec.encode import save_option_table
from responsevec.persona_prompting import (VARIANTS, build_behavior_persona_index,
    build_prompt, build_shuffled_history, demographic_text, score_variant)
from responsevec.persona_router import (build_item_split, fit_persona_bank,
    fold_roles)

DOMAINS = ["A", "B"]
N_ITEMS = 16
DIM = 12
M = 4


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
    train_cal = responses[responses.split.eq("train")
                          & responses.question_key.isin(set(roles["calibration"]))]
    bank = fit_persona_bank(train_cal, roles["calibration"], m=M, seed=11)
    return split, roles, responses, table, calibration, bank


def _lookup(responses, calibration_keys):
    subset = responses[responses.question_key.isin(calibration_keys)]
    return {(str(r.panel_id), str(r.question_key)): int(r.answer_index)
            for r in subset.itertuples(index=False)}


def _text_maps(responses):
    catalogue = responses.drop_duplicates("question_key")
    qtext = {str(r.question_key): str(r.question) for r in catalogue.itertuples(index=False)}
    opts = {str(r.question_key): json.loads(r.options_json)
            for r in catalogue.itertuples(index=False)}
    return qtext, opts


# --------------------------------------------------------------------------- #
# Prompt-content leakage guards
# --------------------------------------------------------------------------- #

def test_no_persona_prompt_omits_demographic_and_history():
    _, roles, responses, _, _, _ = build_context()
    row = next(responses[responses.question_key.isin(roles["test"])].itertuples(index=False))
    prompt = build_prompt(row, ["No", "Maybe", "Yes"], "pe_no_persona", [0, 1, 2])
    assert "Withheld." in prompt          # demographic block blanked
    assert "Relevant previous responses:\nNone" in prompt
    # The demographic key/value never appears.
    assert "education=" not in prompt and "country=" not in prompt
    assert "Answer:" in prompt and "A. No" in prompt and "C. Yes" in prompt


def test_demographic_prompt_contains_demographic_not_history():
    _, roles, responses, _, _, _ = build_context()
    row = next(responses[responses.question_key.isin(roles["test"])].itertuples(index=False))
    prompt = build_prompt(row, ["No", "Maybe", "Yes"], "pe_demographic", [0, 1, 2])
    assert demographic_text(row) in prompt
    assert "Relevant previous responses:\nNone" in prompt   # no history in this variant
    assert "Persona tendencies" not in prompt


def test_history_uses_only_own_calibration_answers():
    _, roles, responses, _, calibration, _ = build_context()
    cal_keys = set(roles["calibration"])
    test = responses[responses.split.eq("test")
                     & responses.question_key.isin(roles["test"])].copy()
    test_cal = responses[responses.split.eq("test") & responses.question_key.isin(cal_keys)]
    lookup = _lookup(test_cal, cal_keys)
    qtext, opts = _text_maps(responses)
    reader = _record_reader()
    score_variant(test, "pe_history", reader.fn, k=5, seed=1701,
                  calibration_by_domain=calibration, answer_lookup=lookup,
                  question_text=qtext, options_by_question=opts, behavior_index={})
    # Every question named in any history line is a calibration item, and no
    # target (test-fold) question text ever appears as a history line.
    calibration_texts = {qtext[q] for q in roles["calibration"]}
    target_texts = {qtext[q] for q in roles["test"]}
    seen = 0
    for prompt in reader.prompts:
        for line in _history_block(prompt).splitlines():
            if "->" not in line:
                continue
            question = line.split(". ", 1)[1].split(" -> ")[0]
            assert question in calibration_texts
            assert question not in target_texts
            seen += 1
    assert seen > 0, "pe_history should render calibration history"


def test_test_target_label_never_enters_prompt():
    _, roles, responses, _, calibration, bank = build_context()
    cal_keys = set(roles["calibration"])
    test = responses[responses.split.eq("test")
                     & responses.question_key.isin(roles["test"])].copy()
    test_cal = responses[responses.split.eq("test") & responses.question_key.isin(cal_keys)]
    lookup = _lookup(test_cal, cal_keys)
    qtext, opts = _text_maps(responses)
    behavior = build_behavior_persona_index(test_cal, bank)
    shuffle = build_shuffled_history(test_cal, 5, 1701, calibration)
    for variant in VARIANTS:
        reader = _record_reader()
        score_variant(test, variant, reader.fn, k=5, seed=1701,
                      calibration_by_domain=calibration, answer_lookup=lookup,
                      question_text=qtext, options_by_question=opts,
                      behavior_index=behavior, shuffled_history=shuffle)
        for prompt, row in zip(reader.prompts, test.itertuples(index=False)):
            # The gold answer text may legitimately appear as an OPTION; assert it
            # never appears inside the history/persona blocks (before "Target").
            before_target = prompt.split("Target question:")[0]
            answer_text = json.loads(row.options_json)[int(row.answer_index)]
            # The history block for this variant must not encode this row's gold
            # answer for the *target question key* (it is never a calibration item).
            assert f"{row.question} -> {answer_text}" not in before_target


def test_shuffled_shares_questions_but_differs_from_real_history():
    _, roles, responses, _, calibration, _ = build_context()
    cal_keys = set(roles["calibration"])
    test = responses[responses.split.eq("test")
                     & responses.question_key.isin(roles["test"])].copy()
    test_cal = responses[responses.split.eq("test") & responses.question_key.isin(cal_keys)]
    lookup = _lookup(test_cal, cal_keys)
    qtext, opts = _text_maps(responses)
    shuffle = build_shuffled_history(test_cal, 5, 1701, calibration)

    real_reader, shuf_reader = _record_reader(), _record_reader()
    score_variant(test, "pe_history", real_reader.fn, k=5, seed=1701,
                  calibration_by_domain=calibration, answer_lookup=lookup,
                  question_text=qtext, options_by_question=opts, behavior_index={})
    score_variant(test, "pe_shuffled_behavior", shuf_reader.fn, k=5, seed=1701,
                  calibration_by_domain=calibration, answer_lookup=lookup,
                  question_text=qtext, options_by_question=opts, behavior_index={},
                  shuffled_history=shuffle)
    differ = 0
    for real, shuf in zip(real_reader.prompts, shuf_reader.prompts):
        real_qs = _history_questions(real)
        shuf_qs = _history_questions(shuf)
        if real_qs and shuf_qs:
            assert real_qs == shuf_qs                 # same question keys preserved
        if _history_block(real) != _history_block(shuf):
            differ += 1
    assert differ > 0, "shuffle should change at least some answers"


def test_behavior_persona_text_comes_from_frozen_bank():
    _, roles, responses, _, _, bank = build_context()
    cal_keys = set(roles["calibration"])
    test_cal = responses[responses.split.eq("test") & responses.question_key.isin(cal_keys)]
    index = build_behavior_persona_index(test_cal, bank)
    assert index, "every test respondent should route to a behaviour persona"
    frozen_texts = {t for d in bank["domains"].values() for t in d["persona_text"].values()}
    for text in index.values():
        assert text in frozen_texts


# --------------------------------------------------------------------------- #
# Split disjointness (reused frozen split)
# --------------------------------------------------------------------------- #

def test_split_is_strictly_disjoint():
    _, roles, responses, _, _, _ = build_context()
    train = set(responses[responses.split.eq("train")].panel_id)
    val = set(responses[responses.split.eq("validation")].panel_id)
    test = set(responses[responses.split.eq("test")].panel_id)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert set(roles["calibration"]).isdisjoint(set(roles["test"]))
    assert set(roles["calibration"]).isdisjoint(set(roles["validation"]))
    assert set(roles["test"]).isdisjoint(set(roles["validation"]))


def test_five_variants_covered():
    assert set(VARIANTS) == {"pe_no_persona", "pe_demographic",
                             "pe_demographic_behavior", "pe_history",
                             "pe_shuffled_behavior"}


# --------------------------------------------------------------------------- #
# End-to-end CLI smoke (dummy reader, CPU only)
# --------------------------------------------------------------------------- #

def _write_inputs(tmp_path):
    split, roles, responses, table, calibration, bank = build_context()
    processed = tmp_path / "processed"
    processed.mkdir()
    responses.to_parquet(processed / "responses.parquet", index=False)
    results = tmp_path / "results"
    results.mkdir()
    (results / "split.json").write_text(json.dumps(split), encoding="utf-8")
    (results / "persona_bank.json").write_text(json.dumps(bank), encoding="utf-8")
    option_table_path = tmp_path / "option_table.npz"
    save_option_table(table, option_table_path)
    return processed, results, option_table_path, roles


def test_cli_end_to_end_dummy_reader(tmp_path):
    processed, results, option_table_path, roles = _write_inputs(tmp_path)
    output = tmp_path / "out"
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    cmd = [sys.executable, str(repo / "scripts" / "persona_prompting_experiment.py"),
           "--processed", str(processed), "--option-table", str(option_table_path),
           "--persona-results", str(results), "--output", str(output),
           "--fold", "0", "--seed", "1701", "--k", "5", "--r", "8"]
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    predictions = pd.read_parquet(output / "predictions.parquet")
    val = pd.read_parquet(output / "validation_predictions.parquet")
    results_csv = pd.read_csv(output / "results.csv")
    config = json.loads((output / "persona_prompting_config.json").read_text())

    methods = set(predictions.method)
    for variant in VARIANTS:
        assert variant in methods
        assert variant + "_uncalibrated" in methods
    assert set(val.method) == methods
    assert config["reader"] == "dummy" and config["mode"] == "ADAPTED"

    for row in predictions.itertuples(index=False):
        probs = np.asarray(json.loads(row.probabilities_json), float)
        assert len(probs) == int(row.n_options)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)
    assert not results_csv.nll.isna().all()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class _record_reader:
    """Capture prompts while returning valid dummy distributions."""
    def __init__(self):
        self.prompts = []

    def fn(self, prompts, n_options, label_to_semantic):
        self.prompts.extend(prompts)
        out = np.zeros((len(prompts), max(int(n) for n in n_options)), np.float32)
        for i, n in enumerate(n_options):
            out[i, : int(n)] = 1.0 / int(n)
        return out


def _history_block(prompt):
    section = prompt.split("Relevant previous responses:\n", 1)[1]
    return section.split("\n\nTarget question:", 1)[0]


def _history_questions(prompt):
    block = _history_block(prompt)
    if block.strip() == "None":
        return []
    return [line.split(". ", 1)[1].split(" -> ")[0] for line in block.splitlines()
            if "->" in line]
