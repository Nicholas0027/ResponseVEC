#!/usr/bin/env bash
# Supplement seeds 7 and 42 for aligned methods (resume skips seed 1701 + all baselines).
set -eo pipefail
cd /data/lab/responsevec
export PYTHONPATH=/data/lab/responsevec/src
CONFIG=configs/responsevec.yaml
ts() { date -u +%H:%M:%S; }
for seed in 7 42; do
  for fold in 0 1 2 3 4 5; do
    echo "[$(ts)] ALIGN fold=$fold seed=$seed"
    python3 scripts/train_primary.py --config "$CONFIG" \
      --fold "$fold" --k 5 --option-seeds 0,7,42 --seeds $seed \
      --include-respondent-vec \
      --align-families response_centric,raw_mean,sentence,input_centric
  done
done
echo "[$(ts)] G1 primary evaluation (3-seed)"
python3 scripts/evaluate_primary.py --config "$CONFIG" --k 5
echo "[$(ts)] SEEDS 7+42 DONE"
