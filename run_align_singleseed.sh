#!/usr/bin/env bash
# Single-seed (1701) x 6-fold ResponseVec-Align run, then G1 re-evaluation.
# Reuses ALL existing extraction + baseline caches; only the *_aligned methods
# are new. Purpose: check cross-fold G1 support with ONE seed before spending
# compute on the full 3-seed grid.
set -eo pipefail
cd /data/lab/responsevec
export PYTHONPATH=/data/lab/responsevec/src
CONFIG=configs/responsevec.yaml
K=5
ts() { date -u +%H:%M:%S; }

for fold in 0 1 2 3 4 5; do
  echo "[$(ts)] ALIGN fold=$fold seed=1701"
  python3 scripts/train_primary.py --config "$CONFIG" \
    --fold "$fold" --k "$K" --option-seeds 0,7,42 --seeds 1701 \
    --include-respondent-vec \
    --align-families response_centric,raw_mean,sentence,input_centric
done

echo "[$(ts)] G1 primary evaluation (single-seed)"
python3 scripts/evaluate_primary.py --config "$CONFIG" --k "$K"
echo "[$(ts)] ALIGN SINGLE-SEED DONE"
