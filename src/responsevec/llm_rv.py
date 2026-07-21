"""Frozen-model representation extractors (design §2.2, §2.3).

Three extraction paths over ONE canonical prompt:
  1. CausalExtractor      — one forward pass yields BOTH direct option-token
                            probabilities (semantic order, masked) AND raw
                            causal hidden states (final-token + mean pooling).
                            Backs baselines "direct logits", "raw causal final",
                            "raw causal mean".
  2. LLM2VecEncoder       — input-centric / response-centric sentence vectors.
                            Wraps the official checkpoint remote code and
                            llm2vec-gen package behind one encode interface.
  3. SentenceEncoder      — a conventional text embedder (BGE) for the
                            sentence-encoder baseline and history retrieval.

Nothing here trains. GPU is used for inference only; outputs are cached
(cache.py) and every downstream head reads from the cache.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .prompting_rv import OPTION_LABELS, option_token_ids


def resolve_quantization(quantization: str | None) -> str | None:
    """'nf4' for 24GB-class cards; bf16 (no quant) on >=30GB GPUs and CPU."""
    import torch

    if quantization not in ("nf4", "auto"):
        return quantization
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 30e9:
        print("[rv] large GPU detected — running bf16 (no quantization)")
        return None
    if not torch.cuda.is_available():
        return None
    return "nf4"


def load_causal_backbone(
    model_name: str,
    dtype: str = "bfloat16",
    quantization: str | None = None,
    revision: str | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization = resolve_quantization(quantization)
    aliases = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = aliases.get(dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True, revision=revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True, "low_cpu_mem_usage": True}
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kwargs)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def choose_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_pool(hidden, attention_mask):
    """Mask-aware mean over the sequence dimension (last hidden layer)."""
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class CausalExtractor:
    """One forward pass -> (direct option probabilities, final-token vector,
    mean-pooled vector). Instruct backbones are wrapped in the chat template
    (the +7.5pt fix carried over from the RAPL scorer)."""

    def __init__(self, model, tokenizer, device, max_length: int = 512, batch_size: int = 32, use_chat_template: bool | None = None):
        import torch

        already_placed = (
            hasattr(model, "hf_device_map")
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        )
        self.model = model if already_placed else model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.use_chat_template = (
            bool(getattr(tokenizer, "chat_template", None)) if use_chat_template is None else bool(use_chat_template)
        )
        if self.use_chat_template:
            probe_prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "probe"}], tokenize=False, add_generation_prompt=True
            )
            ids = option_token_ids(
                tokenizer, len(OPTION_LABELS), continuation_prefix=probe_prefix, label_prefix=""
            )
        else:
            ids = option_token_ids(
                tokenizer, len(OPTION_LABELS), continuation_prefix="Answer:", label_prefix=" "
            )
        self.label_token_ids = torch.tensor(ids, device=device)

    def extract(
        self,
        prompts: Sequence[str],
        n_options: Sequence[int],
        label_to_semantic: Sequence[Sequence[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Returns dict with:
          'probabilities' (n, max_options) float32 semantic-order option probs,
          'final'         (n, hidden) float32 last-token hidden state,
          'mean'          (n, hidden) float32 mask-aware mean hidden state.
        Options are presented in SEMANTIC order here (permutation handled by the
        prediction/eval pipeline); this method reads the last-position logits."""
        import torch

        if len(prompts) != len(n_options):
            raise ValueError("prompts and n_options must have the same length")
        if not prompts:
            return {
                "logits": np.zeros((0, 0), np.float32),
                "probabilities": np.zeros((0, 0), np.float32),
                "final": np.zeros((0, 0), np.float32),
                "mean": np.zeros((0, 0), np.float32),
            }
        if label_to_semantic is None:
            label_to_semantic = [list(range(int(n))) for n in n_options]
        if len(label_to_semantic) != len(prompts):
            raise ValueError("label_to_semantic must have one permutation per prompt")
        max_n = max(n_options)
        logits_out: list[np.ndarray] = []
        probs_out: list[np.ndarray] = []
        final_out: list[np.ndarray] = []
        mean_out: list[np.ndarray] = []

        for start in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[start : start + self.batch_size])
            chunk_n = list(n_options[start : start + self.batch_size])
            chunk_permutations = list(label_to_semantic[start : start + self.batch_size])
            if self.use_chat_template:
                chunk = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                    )
                    for p in chunk
                ]
            tokens = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt", add_special_tokens=not self.use_chat_template,
            ).to(self.device)
            with torch.no_grad():
                output = self.model(
                    input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                    use_cache=False, output_hidden_states=True,
                    # Without this, HF's default computes lm_head over EVERY
                    # sequence position (seq_len x vocab_size per row) before we
                    # slice to the last token — with an 8B model's ~150K vocab
                    # and 512-token prompts that is a ~512x larger allocation
                    # than needed and was the actual OOM (not batch size, not
                    # hidden-state pooling). logits_to_keep=1 makes lm_head
                    # project only the final position.
                    logits_to_keep=1,
                )
            last_hidden = output.hidden_states[-1]  # (b, seq, hidden)
            logits = output.logits[:, -1, :]        # (b, 1, vocab) already sliced -> squeeze via [:, -1, :]

            # Direct option probabilities (semantic order, masked to n_options).
            label_logits = logits.index_select(-1, self.label_token_ids)
            n_tensor = torch.tensor(chunk_n, device=self.device)
            mask = torch.arange(len(self.label_token_ids), device=self.device).unsqueeze(0) < n_tensor.unsqueeze(1)
            label_logits = label_logits.masked_fill(~mask, torch.finfo(label_logits.dtype).min)
            presented_logits = label_logits.float().cpu().numpy()
            presented_probabilities = torch.softmax(label_logits.float(), dim=-1).cpu().numpy()

            final_vec = last_hidden[:, -1, :].float().cpu().numpy()
            mean_vec = _mean_pool(last_hidden, tokens["attention_mask"]).float().cpu().numpy()

            for offset, n in enumerate(chunk_n):
                permutation = [int(x) for x in chunk_permutations[offset]]
                if sorted(permutation) != list(range(int(n))):
                    raise ValueError(f"invalid label_to_semantic permutation: {permutation}")
                semantic_probs = np.zeros(max_n, dtype=np.float32)
                semantic_logits = np.full(max_n, -np.inf, dtype=np.float32)
                for label_index, semantic_index in enumerate(permutation):
                    semantic_probs[semantic_index] = presented_probabilities[offset, label_index]
                    semantic_logits[semantic_index] = presented_logits[offset, label_index]
                semantic_probs[:n] /= semantic_probs[:n].sum()
                probs_out.append(semantic_probs)
                logits_out.append(semantic_logits)
                final_out.append(final_vec[offset])
                mean_out.append(mean_vec[offset])

        return {
            "logits": np.asarray(logits_out, dtype=np.float32),
            "probabilities": np.asarray(probs_out, dtype=np.float32),
            "final": np.asarray(final_out, dtype=np.float32),
            "mean": np.asarray(mean_out, dtype=np.float32),
        }


class LLM2VecEncoder:
    """Uniform wrapper over an LLM2Vec / LLM2Vec-Gen encoder. `encode_fn` is the
    injected encoder callable (prompts -> (n, d) array); the production loader
    builds it from the official package, tests pass a fake. Keeping the wrapper
    thin means the two released checkpoints (input-centric vs response-centric)
    differ only by which one `encode_fn` came from."""

    def __init__(self, encode_fn, name: str = "llm2vec"):
        self._encode = encode_fn
        self.name = name

    def encode(self, prompts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._encode(list(prompts)), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(prompts):
            raise ValueError(f"{self.name} encoder must return (n_prompts, dim); got {vectors.shape}")
        return vectors


def load_llm2vec_encoder(
    checkpoint: str,
    dtype: str = "bfloat16",
    instruction: str | None = None,
    base_checkpoint: str | None = None,
    max_length: int = 512,
    batch_size: int = 16,
    checkpoint_revision: str | None = None,
    base_revision: str | None = None,
    foundation_revision: str | None = None,
) -> LLM2VecEncoder:
    """Load the Qwen3 MNTP encoder plus its supervised adapter.

    The PyPI ``llm2vec==0.2.3`` dispatcher predates Qwen3.  McGill's Qwen3
    MNTP checkpoints instead publish an official ``AutoModel`` remote class
    implementing bidirectional attention.  We instantiate that class with the
    underlying Qwen weights, merge the MNTP adapter, and then apply the
    supervised adapter, which is the same two-stage model represented by the
    checkpoint pair without relying on an incompatible architecture registry.
    """
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    aliases = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = aliases.get(dtype, torch.float32)
    if base_checkpoint is None:
        raise ValueError("Qwen3 LLM2Vec loading requires the MNTP base_checkpoint")
    mntp_peft = PeftConfig.from_pretrained(base_checkpoint, revision=base_revision)
    foundation = str(mntp_peft.base_model_name_or_path)
    config = AutoConfig.from_pretrained(
        base_checkpoint, trust_remote_code=True, revision=base_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_checkpoint, trust_remote_code=True, revision=base_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    load_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    base_model = AutoModel.from_pretrained(
        foundation, revision=foundation_revision, **load_kwargs
    )
    model = PeftModel.from_pretrained(
        base_model, base_checkpoint, revision=base_revision
    )
    model = model.merge_and_unload()
    if checkpoint != base_checkpoint:
        model = PeftModel.from_pretrained(
            model, checkpoint, revision=checkpoint_revision
        )
    model.eval()
    device = next(model.parameters()).device

    def encode_fn(prompts):
        payload = [f"{instruction}{prompt}" for prompt in prompts] if instruction else list(prompts)
        chunks = []
        for start in range(0, len(payload), int(batch_size)):
            tokens = tokenizer(
                payload[start : start + int(batch_size)], padding=True,
                truncation=True, max_length=int(max_length), return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
            pooled = _mean_pool(hidden, tokens["attention_mask"])
            chunks.append(pooled.cpu().float().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), np.float32)

    return LLM2VecEncoder(encode_fn, name=checkpoint)


def load_llm2vec_gen_encoder(
    checkpoint: str,
    dtype: str = "bfloat16",
    instruction: str | None = None,
    max_length: int = 512,
    batch_size: int = 16,
    revision: str | None = None,
) -> LLM2VecEncoder:
    """Load the official response-centric LLM2Vec-Gen checkpoint."""
    import torch
    from llm2vec_gen import LLM2VecGenModel  # type: ignore

    aliases = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    kwargs: dict[str, Any] = {"torch_dtype": aliases.get(dtype, torch.float32)}
    # LLM2VecGenModel.__init__ performs its own model.to(device). Passing an
    # Accelerate device map here would make that official wrapper attempt to
    # move an already-dispatched model, which fails on some runtime versions.
    model = LLM2VecGenModel.from_pretrained(checkpoint, revision=revision, **kwargs)

    def encode_fn(prompts):
        payload = [f"{instruction}{prompt}" for prompt in prompts] if instruction else list(prompts)
        chunks = []
        for start in range(0, len(payload), int(batch_size)):
            with torch.no_grad():
                vectors = model.encode(
                    payload[start : start + int(batch_size)], max_length=int(max_length)
                )
            if isinstance(vectors, tuple):
                vectors = vectors[0]
            if hasattr(vectors, "detach"):
                vectors = vectors.detach().cpu().float().numpy()
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 3:
                vectors = vectors.mean(axis=1)
            chunks.append(vectors)
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), np.float32)

    return LLM2VecEncoder(encode_fn, name=checkpoint)


class SentenceEncoder:
    """Conventional sentence embedder (BGE) for the sentence-encoder baseline
    and history retrieval. Wraps sentence-transformers; injectable for tests."""

    def __init__(self, encode_fn, name: str = "bge"):
        self._encode = encode_fn
        self.name = name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._encode(list(texts)), dtype=np.float32)


def load_sentence_encoder(checkpoint: str, revision: str | None = None) -> SentenceEncoder:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(checkpoint, revision=revision)

    def encode_fn(texts):
        return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    return SentenceEncoder(encode_fn, name=checkpoint)
