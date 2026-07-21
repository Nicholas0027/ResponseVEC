from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_primary.py"


@pytest.fixture(scope="module")
def train_primary_module():
    """Load scripts/train_primary.py as a module so its resumability helpers
    (method_is_complete, load_saved_validation_nll, load_existing_predictions)
    can be unit-tested directly, without needing scripts/ to be a package."""
    spec = importlib.util.spec_from_file_location("train_primary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predictions(rows):
    return pd.DataFrame(rows)


# --- method_is_complete ------------------------------------------------------


def test_none_existing_is_never_complete(train_primary_module):
    assert not train_primary_module.method_is_complete(None, ["direct_raw"], [0])


def test_missing_method_is_not_complete(train_primary_module):
    existing = _predictions([{"method": "other_method", "option_seed": 0}])
    assert not train_primary_module.method_is_complete(existing, ["direct_raw"], [0])


def test_complete_when_all_methods_and_option_seeds_present(train_primary_module):
    existing = _predictions([
        {"method": "raw_mean_seed1701", "option_seed": 0},
        {"method": "raw_mean_seed1701", "option_seed": 7},
        {"method": "raw_mean_seed7", "option_seed": 0},
        {"method": "raw_mean_seed7", "option_seed": 7},
    ])
    assert train_primary_module.method_is_complete(
        existing, ["raw_mean_seed1701", "raw_mean_seed7"], [0, 7]
    )


def test_incomplete_when_an_option_seed_is_missing(train_primary_module):
    """A prior run that only covered option_seed 0 must NOT be treated as
    complete when this run additionally requests option_seed 7 -- silently
    reusing a narrower prior result would understate real coverage."""
    existing = _predictions([
        {"method": "raw_mean_seed1701", "option_seed": 0},
    ])
    assert not train_primary_module.method_is_complete(existing, ["raw_mean_seed1701"], [0, 7])


def test_incomplete_when_a_seed_variant_is_entirely_missing(train_primary_module):
    existing = _predictions([
        {"method": "raw_mean_seed1701", "option_seed": 0},
    ])
    assert not train_primary_module.method_is_complete(
        existing, ["raw_mean_seed1701", "raw_mean_seed7"], [0]
    )


def test_seedless_methods_supported_directly(train_primary_module):
    """direct_raw/direct_scalar have no seed suffix at all -- method_is_complete
    must accept literal method name strings, not assume a "_seed{n}" pattern."""
    existing = _predictions([
        {"method": "direct_raw", "option_seed": 0},
        {"method": "direct_scalar", "option_seed": 0},
    ])
    assert train_primary_module.method_is_complete(existing, ["direct_raw", "direct_scalar"], [0])


# --- load_saved_validation_nll ------------------------------------------------


def test_load_saved_validation_nll_reads_fit_json(tmp_path, train_primary_module):
    fit_dir = tmp_path / "fold_00" / "k_5" / "raw_mean_seed1701"
    fit_dir.mkdir(parents=True)
    (fit_dir / "fit.json").write_text(json.dumps({"best_validation_nll": 1.2345, "best_epoch": 3}))
    value = train_primary_module.load_saved_validation_nll(tmp_path, 0, 5, "raw_mean", 1701)
    assert value == pytest.approx(1.2345)


def test_load_saved_validation_nll_raises_when_fit_json_missing(tmp_path, train_primary_module):
    with pytest.raises(FileNotFoundError, match="force-retrain"):
        train_primary_module.load_saved_validation_nll(tmp_path, 0, 5, "raw_mean", 1701)


# --- load_existing_predictions ------------------------------------------------


def test_load_existing_predictions_returns_none_when_absent(tmp_path, train_primary_module):
    assert train_primary_module.load_existing_predictions(tmp_path) is None


def test_load_existing_predictions_reads_parquet_when_present(tmp_path, train_primary_module):
    frame = pd.DataFrame({"method": ["a"], "option_seed": [0]})
    frame.to_parquet(tmp_path / "predictions_all_replicates.parquet", index=False)
    loaded = train_primary_module.load_existing_predictions(tmp_path)
    assert loaded is not None
    assert list(loaded["method"]) == ["a"]
