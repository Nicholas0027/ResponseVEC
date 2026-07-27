from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load_script("build_cross_survey_split.py")
ceiling = load_script("run_k_ceiling.py")
coverage = load_script("run_coverage_ablation.py")
inference = load_script("summarize_selector_inference.py")
cold_catalog = load_script("build_cold_item_catalog.py")
cold_loading = load_script("run_cold_item_loading.py")
structured = load_script("build_structured_item_prompts.py")
generator = load_script("generate_structured_items.py")
personadelta = load_script("run_personadelta_pretest.py")


def test_family_collapses_grid_subitems():
    assert split.family("CC20_430a_8") == "CC20_430"
    assert split.family("CC20_442e") == "CC20_442"
    assert split.family("CC20_401") == "CC20_401"


def test_respondent_bucket_is_deterministic():
    assert ceiling.bucket(123, 1701) == ceiling.bucket(123, 1701)
    assert 0 <= ceiling.bucket(123, 1701) <= 9


def test_source_selection_respects_k_and_seed():
    items = [f"q{i}" for i in range(20)]
    a = ceiling.choose_source(items, 5, 7)
    b = ceiling.choose_source(items, 5, 7)
    assert a == b and len(a) == 5 and len(set(a)) == 5
    assert ceiling.choose_source(items, 0, 7) == []
    assert ceiling.choose_source(items, "full", 7) == items


def test_matched_shuffle_preserves_demographics_and_row_count():
    frame = pd.DataFrame({
        "gender": [1, 1, 1, 1],
        "birthyr": [1980, 1981, 1982, 1983],
        "educ": [3, 3, 3, 3],
        "q": [10, 20, 30, 40],
    })
    out = ceiling.matched_shuffle(frame, ["q"], 1)
    assert len(out) == len(frame)
    assert sorted(out.q.tolist()) == sorted(frame.q.tolist())


def test_metrics_computes_exact_nll_and_accuracy():
    probabilities = [np.array([[0.8, 0.2], [0.1, 0.9]])]
    classes = [np.array([1, 2])]
    y = np.array([[1], [2]])
    result, loss = ceiling.metrics(probabilities, classes, y)
    assert result["accuracy"] == 1.0
    assert np.isclose(result["nll"], (-np.log(0.8) - np.log(0.9)) / 2)
    assert loss.shape == (2, 1)


def test_identity_bootstrap_sign_is_shuffled_minus_true():
    true = np.full((100, 2), 0.5)
    shuffled = np.full((100, 2), 0.7)
    result = ceiling.bootstrap_gap(true, shuffled, 3, draws=200)
    assert result["gap_nats"] > 0
    assert result["true_history_better"]


def test_greedy_cover_prefers_complementary_items():
    # a covers target 1, b target 2, c redundantly covers target 1.
    items = ["a", "b", "c"]
    scores = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.0]])
    selected, _ = coverage.greedy_cover(items, scores, 2)
    assert set(selected) == {"a", "b"}


def test_family_balanced_uses_distinct_families_first():
    catalog = pd.DataFrame({
        "item": ["a1", "a2", "b1", "b2", "c1"],
        "family": ["a", "a", "b", "b", "c"],
    })
    selected = coverage.family_balanced_select(catalog, 3, 9)
    families = set(catalog.set_index("item").loc[selected].family)
    assert families == {"a", "b", "c"}


def test_paired_bootstrap_difference_has_expected_sign():
    a = np.zeros(200)
    b = np.ones(200)
    result = inference.bootstrap_difference(a, b, seed=2, draws=200)
    assert result["difference"] < 0
    assert result["ci"][1] < 0


def test_option_parser_recovers_single_choice_labels():
    snippet = "CC20_1 SINGLE CHOICE Test? 1 ○ Yes 2 ○ No 8 Skipped 9 Not Asked"
    assert cold_catalog.option_map("CC20_1", snippet, [1, 2]) == {1: "Yes", 2: "No"}


def test_option_parser_recovers_truncated_agreement_scale():
    snippet = ("CC20_2 GRID Do you agree or disagree? CC20_2a Statement "
               "1 ○ Strongly agree 2 ○ Somewhat agree")
    mapping = cold_catalog.option_map("CC20_2a", snippet, [1, 2, 3, 4, 5])
    assert mapping[5] == "Strongly disagree"


def test_position_features_are_finite_and_option_specific():
    frame = pd.DataFrame({"n_options": [5, 5], "option_position": [0, 4]})
    features = cold_loading.position_features(frame)
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])


def test_predictions_from_coefficients_return_normalized_probabilities():
    z = np.array([[0.0, 0.0], [1.0, -1.0]])
    options = pd.DataFrame({"option_code": [1, 2], "option_position": [0, 1]})
    coefficients = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    probabilities, codes = cold_loading.predictions_from_coefficients(
        z, options, coefficients
    )
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert codes.tolist() == [1, 2]


def test_structured_prompt_cache_key_is_deterministic_and_template_scoped():
    options = [{"code": 1, "label": "Support"}, {"code": 2, "label": "Oppose"}]
    a = structured.cache_key("q", "Question?", options, "v1")
    b = structured.cache_key("q", "Question?", options, "v1")
    c = structured.cache_key("q", "Question?", options, "v2")
    assert a == b and a != c


def test_structured_prompts_explicitly_forbid_answer_prevalence_and_person_data():
    for text in structured.TEMPLATES.values():
        lower = text.lower()
        assert "do not" in lower
        assert "respondent" in lower or "simulate a person" in lower
        assert "frequenc" in lower or "common" in lower


def test_json_extractor_ignores_prose_and_finds_object():
    assert generator.extract_json('Here is JSON: {"x": 1} done') == {"x": 1}


def test_json_extractor_rejects_missing_object():
    import pytest
    with pytest.raises(ValueError):
        generator.extract_json("no object here")


def test_personadelta_grid_stem_extracts_item_specific_wording():
    text = ("[CC20_327grid] {grid} Health proposals [CC20_327a] Expand Medicare "
            "to cover all Americans. ◯ ◯ [CC20_327b] Negotiate drug prices.")
    assert personadelta.clean_item_stem(text, "CC20_327a") == (
        "Expand Medicare to cover all Americans."
    )


def test_personadelta_prompt_changes_only_history_block():
    full = personadelta.render_prompt("Target?", ["A", "B"], ["Yes", "No"],
                                      ["- Prior? Answer: Yes"])
    empty = personadelta.render_prompt("Target?", ["A", "B"], ["Yes", "No"], None)
    assert "Prior? Answer: Yes" in full and "Prior? Answer: Yes" not in empty
    assert full.split("Target question:", 1)[1] == empty.split("Target question:", 1)[1]


def test_personadelta_option_permutation_is_deterministic():
    a = personadelta.deterministic_permutation("r1", "q1", 5, 1701)
    b = personadelta.deterministic_permutation("r1", "q1", 5, 1701)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(5))


def test_personadelta_parses_openai_top_logprobs():
    payload = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": " A", "logprob": -0.2},
        {"token": "B", "logprob": -1.2},
    ]}]}}]}
    probabilities = personadelta.parse_openai_logprobs(payload, ["A", "B"])
    assert np.isclose(sum(probabilities), 1.0)
    assert probabilities[0] > probabilities[1]
