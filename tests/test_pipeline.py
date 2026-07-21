from __future__ import annotations

from responsevec.cache import cache_fingerprint
from responsevec.pipeline import extraction_settings


def _config():
    return {
        "representation": {"max_length": 512, "dtype": "bfloat16", "revisions": {}},
        "history": {"retriever": "BAAI/bge-base-en-v1.5"},
    }


def test_resolved_quantization_enters_settings():
    bf16 = extraction_settings(_config(), "semantic", resolved_quantization=None)
    nf4 = extraction_settings(_config(), "semantic", resolved_quantization="nf4")
    assert bf16["resolved_quantization"] is None
    assert nf4["resolved_quantization"] == "nf4"
    # dtype alone is identical ('bfloat16') for both, so ONLY the resolved
    # quantization distinguishes an A100 bf16 run from an L4 4-bit run.
    assert bf16["dtype"] == nf4["dtype"]


def test_nf4_and_bf16_produce_different_fingerprints():
    """The core of the confirmed review bug: same family/checkpoint/split/K but
    different resolved precision must NOT collide on one fingerprint."""
    bf16 = extraction_settings(_config(), "semantic", resolved_quantization=None)
    nf4 = extraction_settings(_config(), "semantic", resolved_quantization="nf4")
    fp_bf16 = cache_fingerprint(
        "causal_final", "Qwen/Qwen3-8B", "B/shared/split=test", 5, 0, settings=bf16
    )
    fp_nf4 = cache_fingerprint(
        "causal_final", "Qwen/Qwen3-8B", "B/shared/split=test", 5, 0, settings=nf4
    )
    assert fp_bf16 != fp_nf4


def test_synthetic_runs_keep_quantization_free_fingerprint():
    """Smoke (synthetic) runs never touch a GPU; their fingerprint must stay
    stable across machines, i.e. carry no resolved_quantization key."""
    settings = extraction_settings(_config(), "semantic")  # default sentinel
    assert "resolved_quantization" not in settings
