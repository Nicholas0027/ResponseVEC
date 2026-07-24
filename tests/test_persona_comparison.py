import numpy as np
import pandas as pd
import pytest

from responsevec.persona_comparison import (build_choice_arrays, fit_lambda,
    fit_position_prior, inductive_pca_option_vectors, log_opinion_pool,
    predict_position_prior, respondent_split, strict_align)


def test_pca_fits_only_calibration_and_train_keys():
    table = {"cal": np.array([[0.0, 0.0], [1.0, 0.0]]),
             "train": np.array([[0.0, 1.0], [1.0, 1.0]]),
             "held": np.array([[100.0, 100.0], [200.0, 200.0]])}
    roles = {"calibration": ["cal"], "train": ["train"],
             "validation": ["held"], "test": []}
    _, first, keys = inductive_pca_option_vectors(table, roles, 2, seed=4)
    changed = dict(table); changed["held"] = table["held"] * 1000
    _, second, _ = inductive_pca_option_vectors(changed, roles, 2, seed=4)
    assert keys == ["cal", "train"]
    assert np.allclose(first.mean_, second.mean_)
    assert np.allclose(np.abs(first.components_), np.abs(second.components_))


def test_history_is_calibration_only_and_same_respondent_split():
    vectors = {"cal": np.array([[1.0, 0.0], [2.0, 0.0]], np.float32),
               "target": np.array([[0.0, 1.0], [0.0, 2.0]], np.float32)}
    rows = pd.DataFrame([{"row_id": "r", "panel_id": "p", "domain": "d",
                          "question_key": "target", "n_options": 2, "answer_index": 1}])
    responses = pd.DataFrame([{"panel_id": "p", "question_key": "cal", "answer_index": 1},
                              {"panel_id": "other", "question_key": "cal", "answer_index": 0}])
    arrays = build_choice_arrays(rows, responses, vectors, {"d": ["cal"]}, 1, 7)
    assert np.allclose(arrays.history[0, 0], vectors["cal"][1])
    wrong_split = responses[responses.panel_id.eq("other")]
    with pytest.raises(ValueError, match="different respondent splits"):
        build_choice_arrays(rows, wrong_split, vectors, {"d": ["cal"]}, 1, 7)


def test_calibration_target_cannot_include_its_own_answer_in_history():
    vectors = {"cal_target": np.array([[1.0, 0.0], [2.0, 0.0]], np.float32),
               "cal_other": np.array([[0.0, 3.0], [0.0, 4.0]], np.float32)}
    rows = pd.DataFrame([{"row_id": "r", "panel_id": "p", "domain": "d",
                          "question_key": "cal_target", "n_options": 2, "answer_index": 1}])
    responses = pd.DataFrame([{"panel_id": "p", "question_key": "cal_target", "answer_index": 1},
                              {"panel_id": "p", "question_key": "cal_other", "answer_index": 0}])
    arrays = build_choice_arrays(
        rows, responses, vectors, {"d": ["cal_target", "cal_other"]}, 2, 3)
    assert arrays.history_mask[0].sum() == 1
    assert np.allclose(arrays.history[0, 0], vectors["cal_other"][0])
    assert not np.any(np.all(arrays.history[0] == vectors["cal_target"][1], axis=1))


def test_lambda_depends_only_on_supplied_validation_labels():
    stat = np.array([[0.9, 0.1], [0.9, 0.1]])
    llm = np.array([[0.1, 0.9], [0.1, 0.9]])
    assert fit_lambda(stat, llm, [0, 0]) == 0.0
    assert fit_lambda(stat, llm, [1, 1]) == 1.0


def test_log_pool_endpoints_equal_components():
    stat = np.array([[0.8, 0.2], [0.3, 0.7]])
    llm = np.array([[0.4, 0.6], [0.9, 0.1]])
    assert np.allclose(log_opinion_pool(stat, llm, 0.0), stat)
    assert np.allclose(log_opinion_pool(stat, llm, 1.0), llm)


def test_validation_respondent_split_is_disjoint_and_complete():
    panels = [f"p{i}" for i in range(40)]
    group_a, group_b = respondent_split(panels, 11)
    assert group_a.isdisjoint(group_b)
    assert group_a | group_b == set(panels)
    assert group_a and group_b


def test_position_prior_does_not_read_held_out_question_labels():
    train = pd.DataFrame({"question_key": ["q1", "q2"], "answer_index": [0, 1],
                          "n_options": [3, 3]})
    held = pd.DataFrame({"question_key": ["held"], "answer_index": [2], "n_options": [3]})
    first = predict_position_prior(fit_position_prior(train), held)[0]
    held.loc[0, "answer_index"] = 0
    second = predict_position_prior(fit_position_prior(train), held)[0]
    assert np.allclose(first, second)


def test_demographic_position_prior_falls_back_for_small_groups():
    train = pd.DataFrame({
        "country": ["small"] * 2 + ["large"] * 30,
        "answer_index": [0] * 2 + [1] * 30,
        "n_options": [2] * 32,
    })
    model = fit_position_prior(train, demographic=True, minimum_group_n=25)
    target = pd.DataFrame({"country": ["small", "large"], "n_options": [2, 2]})
    small, large = predict_position_prior(model, target)
    fallback = np.asarray(model["fallback"]); fallback /= fallback.sum()
    assert model["group_n"]["small"] == 2
    assert model["group_n"]["large"] == 30
    assert np.allclose(small, fallback)
    assert large[1] > fallback[1]


def test_strict_alignment_rejects_duplicates_and_metadata_mismatch():
    left = pd.DataFrame([{"row_id": "1", "panel_id": "p", "question_key": "q",
                          "answer_index": 0, "n_options": 2}])
    duplicate = pd.concat([left, left], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate row_id"):
        strict_align(left, duplicate)
    mismatch = left.copy(); mismatch["answer_index"] = 1
    with pytest.raises(ValueError, match="answer_index"):
        strict_align(left, mismatch)
