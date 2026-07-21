"""Frozen representation extractors and the trainable option-aware decoder.

Nothing here trains a backbone: direct logits, raw causal hidden states, and the
LLM2Vec / LLM2Vec-Gen encoders are all read from frozen models; only the
option-aware decoder (models/decoder.py) holds gradients."""
