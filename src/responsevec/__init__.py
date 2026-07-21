"""ResponseVec: response-centric representations for individual-level silicon sampling.

The claim under test: a prompted LLM carries response-relevant signal in its
latent representation that its generic next-token head does not faithfully
express. A response-centric embedding (LLM2Vec-Gen) plus a small option-aware
decoder should recover that signal — beating direct option-token probabilities,
raw causal hidden states, and input-centric LLM2Vec under the SAME backbone,
prompt, decoder budget, and data splits, especially on genuinely unseen items.

All embedding backbones are frozen; only a <=2M-parameter decoder trains. GPU is
used for one-pass representation extraction, cached and hash-validated, then all
heads train on CPU/GPU from the cache.
"""

__all__: list[str] = []
