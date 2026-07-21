from __future__ import annotations

import numpy as np
import pytest

from responsevec.llm_rv import LLM2VecEncoder, SentenceEncoder


def test_llm2vec_wrapper_validates_shape():
    enc = LLM2VecEncoder(lambda prompts: np.ones((len(prompts), 16)), name="fake")
    out = enc.encode(["a", "b", "c"])
    assert out.shape == (3, 16)
    assert out.dtype == np.float32


def test_llm2vec_wrapper_rejects_bad_shape():
    enc = LLM2VecEncoder(lambda prompts: np.ones((len(prompts) + 1, 16)), name="fake")
    with pytest.raises(ValueError, match="n_prompts"):
        enc.encode(["a", "b"])


def test_sentence_encoder_wrapper():
    enc = SentenceEncoder(lambda texts: np.arange(len(texts) * 4).reshape(len(texts), 4), name="fake")
    out = enc.encode(["x", "y"])
    assert out.shape == (2, 4)
    assert out.dtype == np.float32


def test_causal_extractor_direct_and_hidden_states():
    """Tiny random causal LM exercises the real forward path: one pass yields
    masked semantic-order option probabilities plus final/mean hidden states."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from responsevec.llm_rv import CausalExtractor

    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    extractor = CausalExtractor(
        model, tokenizer, torch.device("cpu"), max_length=64, batch_size=2, use_chat_template=False
    )
    prompts = ["Target question:\nHow concerned?\nOptions:\nA. x\nB. y\nC. z\nAnswer:",
               "Target question:\nAgree?\nOptions:\nA. p\nB. q\nAnswer:"]
    out = extractor.extract(prompts, n_options=[3, 2])

    probs = out["probabilities"]
    assert probs.shape == (2, 3)                         # padded to max_options=3
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert probs[1, 2] == 0.0                            # second item has only 2 options -> slot masked
    assert out["final"].shape[0] == 2
    assert out["mean"].shape == out["final"].shape
    assert out["final"].shape[1] > 0                     # hidden dim recovered
    # final and mean pooling genuinely differ
    assert not np.allclose(out["final"], out["mean"])
