from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from responsevec.cache import RepresentationCache, cache_fingerprint


def _rows(row_ids):
    return pd.DataFrame({"row_id": list(row_ids), "panel_id": ["p"] * len(row_ids), "k": [5] * len(row_ids)})


def test_fingerprint_is_deterministic_and_sensitive():
    a = cache_fingerprint("response_centric", "ckpt", "foldA", 5, 0)
    b = cache_fingerprint("response_centric", "ckpt", "foldA", 5, 0)
    assert a == b
    assert a != cache_fingerprint("response_centric", "ckpt", "foldA", 8, 0)  # K differs
    assert a != cache_fingerprint("direct", "ckpt", "foldA", 5, 0)            # family differs
    assert a != cache_fingerprint("response_centric", "ckpt2", "foldA", 5, 0) # checkpoint differs


def test_append_and_read_roundtrip(tmp_path):
    cache = RepresentationCache.create(
        tmp_path / "c", family="response_centric", checkpoint="ckpt",
        item_split="foldA", k=5, option_seed=0, has_logits=False,
    )
    vecs = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    cache.append(_rows(["r0", "r1", "r2"]), vecs)
    assert cache.n_rows == 3
    read_back = cache.read_vectors()
    assert read_back.dtype == np.float16          # stored compact
    assert read_back.shape == (3, 8)
    np.testing.assert_allclose(read_back.astype(np.float32), vecs, atol=1e-2)
    assert list(cache.read_rows()["row_id"]) == ["r0", "r1", "r2"]


def test_resumable_append_across_shards(tmp_path):
    cache = RepresentationCache.create(
        tmp_path / "c", family="direct", checkpoint="ckpt",
        item_split="foldA", k=5, option_seed=0, has_logits=True,
    )
    cache.append(_rows(["r0", "r1"]), np.zeros((2, 4), np.float32), logits=np.ones((2, 5), np.float32))
    assert cache.already_done_row_ids() == {"r0", "r1"}
    # A resumed run skips completed rows and appends the remainder.
    cache.append(_rows(["r2"]), np.zeros((1, 4), np.float32), logits=np.ones((1, 5), np.float32))
    assert cache.n_rows == 3
    assert cache.read_logits().shape == (3, 5)
    assert len(list((tmp_path / "c" / "shards").glob("*.vectors.npy"))) == 2


def test_duplicate_append_is_rejected(tmp_path):
    cache = RepresentationCache.create(
        tmp_path / "c", family="x", checkpoint="ckpt", item_split="foldA",
        k=5, option_seed=0, has_logits=False,
    )
    cache.append(_rows(["r0"]), np.zeros((1, 4), np.float32))
    with pytest.raises(ValueError, match="duplicate completed"):
        cache.append(_rows(["r0"]), np.zeros((1, 4), np.float32))


def test_settings_participate_in_fingerprint(tmp_path):
    cache = RepresentationCache.create(
        tmp_path / "c", family="x", checkpoint="ckpt", item_split="foldA",
        k=5, option_seed=0, has_logits=False,
        settings={"history_selection": "semantic", "max_length": 512},
    )
    with pytest.raises(ValueError, match="settings"):
        RepresentationCache.load(
            tmp_path / "c", expect_settings={"history_selection": "random", "max_length": 512}
        )
    assert cache.validate()["n_rows"] == 0


def test_load_rejects_fingerprint_mismatch(tmp_path):
    RepresentationCache.create(
        tmp_path / "c", family="response_centric", checkpoint="ckpt",
        item_split="foldA", k=5, option_seed=0, has_logits=False,
    )
    # Loading with a different K must refuse rather than serve wrong embeddings.
    with pytest.raises(ValueError, match="expects"):
        RepresentationCache.load(tmp_path / "c", expect_family="response_centric", expect_k=8)
    # Correct expectations load fine.
    ok = RepresentationCache.load(tmp_path / "c", expect_family="response_centric", expect_k=5)
    assert ok.n_rows == 0


def test_logits_contract_enforced(tmp_path):
    cache = RepresentationCache.create(
        tmp_path / "c", family="response_centric", checkpoint="ckpt",
        item_split="foldA", k=5, option_seed=0, has_logits=False,
    )
    with pytest.raises(ValueError, match="has_logits=False but logits provided"):
        cache.append(_rows(["r0"]), np.zeros((1, 4), np.float32), logits=np.ones((1, 5), np.float32))
