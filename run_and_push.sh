#!/usr/bin/env bash
# run_and_push.sh — end-to-end: run the ResponseVec experiment pipeline on the
# A100, then automatically commit + push code, paper, and result artifacts to
# GitHub when (and only when) the run completes successfully.
#
# SECURITY: pass the GitHub token ONLY via the GITHUB_TOKEN environment
# variable. It is never written to disk and is scrubbed from the git remote
# after the push (see push_results.sh).
#
# Usage:
#   GITHUB_TOKEN=<token> RUN_MODE=smoke bash run_and_push.sh
#   GITHUB_TOKEN=<token> RUN_MODE=a100  bash run_and_push.sh
#
# RUN_MODE=smoke  -> CPU synthetic smoke (fast, non-scientific; verifies wiring)
# RUN_MODE=a100   -> real Qwen3-8B run: G0 signal audit -> (if pass) R1 primary
#                    -> R2/R3 transfer + cost -> figures. Auto-push at the end.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/lab/responsevec}"
RUN_MODE="${RUN_MODE:-smoke}"
CONFIG="${CONFIG:-configs/responsevec.yaml}"
K="${K:-5}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

cd "$REPO_DIR"
mkdir -p artifacts

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [[ "$RUN_MODE" == "smoke" ]]; then
  log "SMOKE: CPU synthetic end-to-end (non-scientific)"
  python3 scripts/run_smoke.py --config "$CONFIG" --workdir artifacts/smoke
  RESULT_MSG="chore: smoke run $(date -u +%Y-%m-%dT%H:%M:%SZ) (non-scientific wiring check)"
else
  log "A100: real Qwen3-8B pipeline"

  log "Step 1/7: prepare real SocioBench data + item folds"
  python3 scripts/prepare_data.py --config "$CONFIG"

  log "Step 2/7: G0 signal audit (CPU gate — do not spend 8B extraction if this fails)"
  python3 scripts/signal_audit.py --config "$CONFIG" --fold 0
  # Inspect the gate; abort the expensive path if G0 fails.
  if ! python3 - "$CONFIG" <<'PY'
import json, sys, yaml, pathlib
cfg = yaml.safe_load(open(sys.argv[1]))
p = pathlib.Path(cfg["paths"]["metrics"]) / "signal_audit.json"
if not p.exists():
    print("G0 report missing"); sys.exit(1)
g0 = json.loads(p.read_text())
ok = bool(g0.get("seen_gate_pass", False))
print("G0:", g0)
sys.exit(0 if ok else 1)
PY
  then
    log "G0 FAILED — not launching 8B extraction. Pushing the negative-result report only."
    RESULT_MSG="results: G0 signal audit failed — negative result ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    GITHUB_TOKEN="${GITHUB_TOKEN}" bash "${REPO_DIR}/push_results.sh" "$RESULT_MSG"
    exit 0
  fi

  log "Step 3/7: encode options (once)"
  python3 scripts/encode_options.py --config "$CONFIG" --overwrite

  # HF 429 mitigation: every model weight is already cached locally. Standard
  # LLM2Vec adapters (causal/input_centric/sentence) load cleanly with
  # HF_HUB_OFFLINE=1 (zero network HEAD calls). Only the LLM2Vec-Gen wrapper
  # (response_centric) needs online-cached loading; we give it retries and it
  # is now the ONLY family issuing any HTTP request, so 429 pressure is minimal.
  extract_family () {
    local family=$1 split=$2 oseed=$3
    log "Step 4/7: extract R1 family=$family split=$split option-seed=$oseed k=$K"
    if [[ "$family" == "response_centric" ]]; then
      local ok=0
      for try in 1 2 3 4 5; do
        if python3 scripts/extract_representations.py --config "$CONFIG" \
            --protocol B --family "$family" --split "$split" --k "$K" \
            --option-seed "$oseed" --selection semantic --shard-size 512; then ok=1; break; fi
        log "  response_centric attempt $try hit an error (likely 429) -> backoff $((try*30))s"; sleep $((try*30))
      done
      [[ $ok -eq 1 ]] || { log "FATAL: response_centric extraction failed after retries"; exit 1; }
    else
      HF_HUB_OFFLINE=1 python3 scripts/extract_representations.py --config "$CONFIG" \
        --protocol B --family "$family" --split "$split" --k "$K" \
        --option-seed "$oseed" --selection semantic --shard-size 512
    fi
  }
  # train/validation heads use option-seed 0 only; the TEST split is extracted
  # for all three option seeds (0,7,42) for option-permutation averaging.
  for split in train validation test; do
    if [[ "$split" == "test" ]]; then OPTS="0 7 42"; else OPTS="0"; fi
    for oseed in $OPTS; do
      for family in causal input_centric response_centric sentence; do
        extract_family "$family" "$split" "$oseed"
      done
    done
    HF_HUB_OFFLINE=1 python3 scripts/extract_respondentvec.py --config "$CONFIG" \
      --split "$split" --k "$K" --shard-size 512
  done

  log "Step 5/7: train R1 heads across 6 folds (3 decoder seeds x 3 option seeds)"
  for fold in 0 1 2 3 4 5; do
    python3 scripts/train_primary.py --config "$CONFIG" \
      --fold "$fold" --k "$K" --option-seeds 0,7,42 --seeds 1701,7,42 \
      --include-respondent-vec \
      --align-families response_centric,raw_mean,sentence,input_centric
  done

  log "Step 6/7: G1 primary evaluation (the gated claim)"
  python3 scripts/evaluate_primary.py --config "$CONFIG" --k "$K"

  log "Step 7/7: R2/R3 transfer + cost + figures"
  # Protocol A train cache + D transfer caches for each family. Same 429
  # mitigation as Step 4: standard adapters load OFFLINE (zero HTTP), only the
  # LLM2Vec-Gen response_centric family loads online-cached with retries.
  extract_transfer () {
    local family=$1 protocol=$2 split=$3
    if [[ "$family" == "response_centric" ]]; then
      for try in 1 2 3 4 5; do
        python3 scripts/extract_representations.py --config "$CONFIG" \
          --protocol "$protocol" --family "$family" --split "$split" --k "$K" --option-seed 0 && return 0
        log "  transfer $protocol/$family attempt $try failed -> backoff $((try*30))s"; sleep $((try*30))
      done
      log "FATAL: transfer extraction failed for $protocol/$family"; return 1
    else
      HF_HUB_OFFLINE=1 python3 scripts/extract_representations.py --config "$CONFIG" \
        --protocol "$protocol" --family "$family" --split "$split" --k "$K" --option-seed 0
    fi
  }
  for family in causal input_centric response_centric sentence; do
    extract_transfer "$family" A train
    extract_transfer "$family" D test
  done
  python3 scripts/evaluate_transfer.py --config "$CONFIG" --k "$K" || \
    log "transfer eval reported partial results (some caches may be absent)"
  python3 scripts/make_figures.py --config "$CONFIG" || log "figure step skipped"

  RESULT_MSG="results: A100 run k=$K $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

log "Auto-push to GitHub"
GITHUB_TOKEN="${GITHUB_TOKEN}" bash "${REPO_DIR}/push_results.sh" "$RESULT_MSG"
log "DONE"
