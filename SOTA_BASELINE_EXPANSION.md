# SOTA Baseline Expansion Addendum

Status: planning document, written after confirmatory folds 1-5 were inspected.
Everything here is a POST-FREEZE addition. The frozen proposed method
(`hist_gbdt_llm_hybrid`) and the confirmatory gates in
`PERSONA_PROTOCOL_FREEZE.md` are NOT changed by anything in this document.

## Why this addendum exists

The current results table (Uniform, position prior, HistGBDT, statistical
persona, LLM persona history, two hybrids) is a controlled ablation plus
classical anchors. It is NOT a comparison against recent conference SOTA for
LLM-based human simulation, opinion prediction, or personalization. This
addendum defines the recent LLM-centric methods to reproduce, how to adapt each
to the frozen doubly-cold protocol, and the discipline for running them without
contaminating the confirmatory result.

## Global rules for every added baseline

- Reuse the frozen split from `split.json`; never rebuild the item split.
- Train / validation / test respondents stay disjoint; test target labels are
  never read for any hyperparameter, layer, prompt, retrieval-K, temperature, or
  fusion-weight decision.
- Any hyperparameter is chosen on validation respondents only.
- The proposed hybrid is frozen. No added baseline may trigger a change to it.
- Report every method regardless of outcome. Mark each method as EXACT (faithful
  reproduction) or ADAPTED (protocol-constrained reimplementation).
- Primary uncertainty is respondent-clustered bootstrap; also report the
  domain-stratified item bootstrap because it currently crosses zero.
- Prefer adding a fully unused external survey dataset for the SOTA comparison so
  the 40 confirmatory target items are not reused for model selection.

## Task split of the two result tables

Individual doubly-cold prediction (Table A) and population/subgroup distribution
prediction (Table B) are separate. Distribution-only methods (SubPOP, MindVote,
Modular Pluralism) belong in Table B and must not be scored on individual NLL.

## Backbone coverage requirement

Qwen3-8B alone is insufficient: ACL-2024 persona-effect scaling shows persona
gains grow with model size. The added comparison MUST include at least one
larger general model (Qwen3-32B, and a 70B-class open model if compute allows)
and at least one human-simulation-specialized checkpoint (HumanLLM-8B). All LLM
heads share the same personas, router posterior, target items, calibration
protocol, and validation-only temperature calibration.

## Method reproduction specifications

Each entry lists: venue, mode (EXACT/ADAPTED), which result table, the frozen
protocol mapping, and the leakage guard.

### 1. Persona Effect Prompting (Hu & Collier, ACL 2024) [P1]
- Mode: ADAPTED. Table A.
- Prompt-only persona conditioning; no training.
- Variants: no-persona, demographic, demographic+behavior, history, shuffled
  behavior control; add the trained linear persona oracle as an upper anchor.
- Backbones: Qwen3-8B, Qwen3-32B, one 70B-class model if compute allows.
- Guard: personas built only from calibration answers; shuffled control uses the
  frozen question-aligned shuffle.

### 2. Persona-DB (Sun et al., COLING 2025) [P2]
- Mode: ADAPTED. Table A. Direct competitor to the persona bank.
- Self DB = the respondent's K calibration answers; collaborative DB = train
  respondents only; hierarchical persona keys; query-focused retrieval; JOIN for
  cold-start backfill; frozen Qwen3-8B as the downstream reader.
- Variants: History-Full, History-Retrieval, IntSum, Persona-DB w/o JOIN,
  Persona-DB full.
- Guard: test respondents never contribute collaborative evidence to each other;
  no target-item labels enter retrieval.

### 3. SubPOP-FT (Suh et al., ACL 2025) [P3]
- Mode: EXACT recipe on aggregate task, ADAPTED extension. Table B only.
- (subpopulation, question) -> response distribution, LoRA fine-tuning on
  train-respondent-derived subgroup distributions over calibration+train items.
- Variants: zero-shot PORTRAY, few-shot, Modular Pluralism, SubPOP-FT, and a
  clearly labelled SubPOP-FT + history extension.
- Guard: fine-tune only on train items/respondents; evaluate on cold target
  items; never fit on target-item distributions.

### 4. MindVote inference recipe (Mao et al., AAAI 2026) [P4]
- Mode: ADAPTED. Table B (distribution) with individual-projection noted.
- Recipes: zero-shot distribution, demographic-context priming, history-context
  priming, few-shot, reasoning-before-choice, self-consistency sampled
  population, temperature-controlled sampling, one reasoning-model backbone.
- Purpose: test whether the hybrid gain survives beyond first-token decoding by
  comparing token-prob, repeated sampling, explicit JSON distribution, and
  reasoning elicitation.
- Guard: prompts fixed on validation; no target labels in prompt selection.

### 5. Query-Focused Progressive Persona Completion (Su et al., ACL Findings 2026) [P5]
- Mode: ADAPTED. Table A.
- Persona acquisition restricted to the calibration pool (no new info from test
  respondents). Variants: query-agnostic fixed-K, semantic-RAG top-K,
  query-focused one-shot, query-focused progressive with confidence stopping
  (<=5), gold-relevance upper bound defined on TRAIN mutual information.
- Guard: relevance ranking never uses test answers; confidence stopping is the
  model's own signal.

### 6. OPINIONS residual-stream probing (2026) [P6]
- Mode: ADAPTED. Table A. Highest-risk competitor to the ResponseVec line.
- next-token prob, per-layer linear probe, per-layer nonlinear probe,
  validation-selected best layer, unembedding-only tuning, LoRA upper bound.
- Guard: probes trained only on target-train items; layer chosen on validation
  items; test items excluded from selection; fixed prompt.

### 7. HumanLLM head (ACL 2026) [P7]
- Mode: ADAPTED. Table A. Backbone swap of the LLM head.
- Same persona bank, router posterior, target items, calibration, temperature
  scaling; only the generation model changes to HumanLLM-8B.
- Compares Qwen3-8B head vs HumanLLM-8B head, and HistGBDT+Qwen vs
  HistGBDT+HumanLLM hybrids.

### 8. ThinkPersona graph head (ACL 2026) [P8]
- Mode: ADAPTED. Table A.
- Build a persona graph from calibration history (values, attitudes, behavior
  evidence, demographic nodes, QA-evidence edges, train-derived cluster nodes);
  grounded reasoning before the choice.
- Variants: flat persona, persona graph, persona graph + grounded reasoning,
  persona graph + HistGBDT hybrid.

### 9. Learned Persona Loading (persona-embedding travel-choice paper) [P9]
- Mode: ADAPTED. Table A. Closest architectural competitor.
- Variants: random persona, same-group persona, learned demographic loading,
  stochastic loading, learned loading + history posterior.
- Kept as a direct architecture competitor, not as "conference SOTA".

### 10. Dual-agent textual persona calibration (TR-C 2026) [P10]
- Mode: ADAPTED, third priority (expensive). Table A.
- Separate item pools for persona induction, error feedback, candidate selection,
  target test; textual pseudo-gradient candidate personas with held-out
  selection; group-prior smoothing.
- Guard: candidate generation/selection never touches target-test labels.

## Priority tiers

- Tier 1 (required to claim SOTA comparison): P1, P2, P3, P6, P4, plus a 32B/70B
  general backbone.
- Tier 2 (strong additions): P5, P7, P8, Modular Pluralism.
- Tier 3 (expensive, optional): P10, proprietary reasoning models, recursive
  persona-graph calibration.

## Freeze-safe execution order

1. Finalize this addendum; label every method EXACT/ADAPTED; fix backbones,
   prompts, retrieval-K, LoRA rank, layer-selection, temperature, metrics.
2. Prefer an unused external survey dataset for the SOTA comparison; otherwise
   reuse the frozen split with explicit "reused-items" caveat.
3. Implement Tier 1, one module + unit tests per method, CPU synthetic smoke
   test, then Lab GPU runs.
4. Score into Table A / Table B with respondent- and item-clustered bootstrap.
5. Never modify the frozen `hist_gbdt_llm_hybrid` or its gates; all SOTA results
   are reported as a post-freeze comparison.

## Key references (verify with ref_verify before any write into refs.bib)

- Hu & Collier, Quantifying the Persona Effect in LLM Simulations, ACL 2024.
- Sun et al., Persona-DB, COLING 2025.
- Suh et al., Language Model Fine-Tuning on Scaled Survey Data (SubPOP), ACL 2025.
- Mao et al., MindVote, AAAI 2026.
- Su et al., Query-Focused Individual Simulation with Progressive Persona
  Completion, ACL Findings 2026.
- What Do Large Language Models Know About Opinions? (residual-stream probing), 2026.
- HumanLLM: Benchmarking and Improving LLM Anthropomorphism, ACL 2026.
- ThinkPersona: Thinking with Persona Graphs, ACL 2026.
- Santurkar et al., OpinionQA (Whose Opinions Do LLMs Reflect?), 2023 (Table B anchor).
- Durmus et al., GlobalOpinionQA, 2023/2024 (Table B anchor).
