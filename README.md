# ResponseVec

ResponseVec is the executable MVP for testing whether a response-centric LLM representation predicts a held-out respondent answer better than the same backbone's direct output, raw hidden states, and input-centric LLM2Vec on unseen survey items.

The primary experiment is deliberately narrow. It uses one frozen 8B backbone family, one option-aware decoder contract, Protocol B unseen-item evaluation (R1), and exactly three preregistered NLL comparisons. The two transfer regimes (R2 cross-domain via Protocol C, R3 OOD-demographic-intersection via Protocol D) and the cost-accounting table are produced by `scripts/evaluate_transfer.py` and do not gate G1. Expensive K curves and extensions run only after the signal and primary gates pass.

## Scientific contract

- Respondents are split 70/10/20; respondent-test labels never fit a head, prior, retriever, or calibration parameter.
- R1 (Protocol B): per domain, a fixed calibration-source pool plus six candidate-item bins. For outer fold `f`, bin `f` is test, `(f + 1) % 6` is validation, and the other four bins train the decoder. Every candidate item is tested exactly once.
- R2 (Protocol C): leave-one-domain-out; decoder trained on the other three domains' seen items.
- R3 (Protocol D): entire demographic-intersection cells held out into `ood_intersection` split; decoder trained on ID-respondent train split (as R1), scored on held-out respondents' seen items.
- Protocol B representations are cached once per respondent split, K, option order, and representation family. Fold roles filter the shared cache only when the small head is trained.
- Item-specific priors and baselines are fit only on the current outer fold's training items.
- All representation families use the same semantic prompt content, frozen option vectors, decoder form, training labels, and fixed prior weight.
- Final-test probabilities average three deterministic option-label permutations and three decoder seeds.
- The unseen-item interval is a paired, domain-macro crossed respondent-by-item bootstrap. The three primary tests use Holm correction and require at least `0.02` nats improvement.
- Per-method GPU-hours, latency, and peak memory are recorded by `responsevec.eval.cost` for the accuracy--cost Pareto frontier (Figure 2).
- Smoke outputs are deterministic integration fixtures and are never scientific evidence.

## Fast local verification

Python 3.9+ is supported. The CPU path does not download language models.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/run_smoke.py --config configs/responsevec.yaml --workdir artifacts/smoke
```

A successful smoke run writes `artifacts/smoke/smoke_summary.json` with `"status": "PASS"`. It exercises the same data, cache, head, metric, and primary-bootstrap contracts as the GPU path, using fake deterministic encoders.

## Colab A100 run

Use [responsevec_aaai_pipeline.ipynb](output/jupyter-notebook/responsevec_aaai_pipeline.ipynb). The default `RUN_MODE="smoke"` must pass first. Then connect an A100 runtime, change the mode to `"a100"`, and run top-to-bottom.

For a clean Colab session:

1. Upload `responsevec_colab.zip` to `/content/drive/MyDrive/`.
2. Open the notebook in Colab and run it in smoke mode.
3. Connect an A100 40 GB or larger runtime, switch to A100 mode, and rerun.
4. Keep `RUN_K_CURVE=False` until G1 has been inspected.

The 8B experiment requires local model weights and hidden-state access. A chat-completion API cannot replace this path. Hugging Face access, enough Drive space for immutable caches, and acceptance of the upstream dataset/model terms are required. The notebook installs `llm2vec-gen==0.1.3` with `--no-deps`: its published package metadata includes training-only dependencies and an exact Torch build, while this project pins the smaller inference-compatible stack explicitly.

## Staged commands

The notebook calls these scripts in order:

```bash
python scripts/prepare_data.py --config <runtime-config>
python scripts/signal_audit.py --config <runtime-config> --fold 0
python scripts/encode_options.py --config <runtime-config>

# Run for train/validation/test, representation families, and test option seeds.
python scripts/extract_representations.py \
  --config <runtime-config> --family response_centric \
  --split test --k 5 --option-seed 0 --selection semantic

# Run folds 0..5.
python scripts/train_primary.py \
  --config <runtime-config> --fold 0 --k 5 \
  --option-seeds 0,7,42 --seeds 1701,7,42

python scripts/evaluate_primary.py --config <runtime-config> --k 5
```

The `causal` extraction produces both direct option logits and raw final/mean hidden states in one pass. `input_centric` and `response_centric` are separate frozen encoder passes. Each cache has a manifest fingerprint containing its model/settings, item partition, and candidate-row hashes; writes are immutable, numbered, and resumable.

## Gate interpretation

- **G0 seen signal:** an item-conditional history model must beat its K=0 counterpart by the configured NLL margin.
- **G0 unseen headroom:** an oracle unseen-item marginal must improve over the deployable scale-position fallback. This is diagnostic only and never enters prediction.
- **G1 primary:** ResponseVec must beat `direct_selected`, `input_centric`, and `raw_selected` at unseen-item K=5 after Holm correction and the practical-effect threshold.

If G0 fails, do not launch the 8B primary extraction. If G1 fails, report the negative result and do not expand the claim by searching K values or optional baselines.

## Main outputs

- `processed/item_folds.json`: frozen six-fold item partition.
- `cache/**/manifest.json`: cache identity, shapes, and completed shards.
- `heads/fold_*/k_*/`: fitted head state and selection metadata.
- `predictions/fold_*/k_*/predictions_seed_averaged.parquet`: auditable row probabilities.
- `metrics/signal_audit.json`: G0 decision.
- `metrics/k_5/primary_family_nll.csv`: three preregistered comparisons.
- `metrics/k_5/gate_g1.json`: primary decision.

The authoritative settings are in [configs/responsevec.yaml](configs/responsevec.yaml). Do not edit cached manifests or prediction files by hand; change the config and rerun into a new artifact directory.
