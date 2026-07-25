#!/usr/bin/env bash
# Idempotent environment restore for an ephemeral Lab instance.
# Rebuilds deps, SocioBench raw data, processed parquet (frozen seed 1701),
# and restores the frozen persona artifacts from the encrypted backup.
# Requires: BACKUP_PASSPHRASE env var for the encrypted artifact tarball.
# Does NOT contain any credential.
set -eu

ROOT="/data/lab/responsevec"
cd "$ROOT"

echo "[1/6] python deps"
pip install -q pandas numpy scikit-learn scipy pyarrow pytest pyyaml \
  'transformers==4.56.2' 'accelerate>=1.0' >/dev/null 2>&1 || true

echo "[2/6] SocioBench raw data"
if [ ! -d external/SocioBench/Dataset_all ]; then
  mkdir -p external
  git clone --depth 1 https://github.com/JiaWANG-TJ/SocioBench.git external/SocioBench >/dev/null 2>&1
fi

echo "[3/6] restore frozen artifacts from encrypted backup"
if [ -f artifacts/backups/persona-confirmatory-20260724.tar.gz.enc ]; then
  if [ -z "${BACKUP_PASSPHRASE:-}" ]; then
    echo "  BACKUP_PASSPHRASE not set; skipping artifact restore"
  else
    openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 \
      -in artifacts/backups/persona-confirmatory-20260724.tar.gz.enc \
      -out /tmp/pc.tar.gz -pass pass:"$BACKUP_PASSPHRASE"
    tar -xzf /tmp/pc.tar.gz
    rm -f /tmp/pc.tar.gz
  fi
fi

echo "[4/6] rebuild processed parquet (frozen seed)"
if [ ! -f artifacts/processed/responses.parquet ]; then
  PYTHONPATH=src python3 scripts/prepare_data.py --config configs/responsevec.yaml
fi

echo "[5/6] consistency check against frozen split"
PYTHONPATH=src python3 - <<'PY'
import json, glob, numpy as np, pandas as pd
resp = pd.read_parquet('artifacts/processed/responses.parquet')
cand = sorted(glob.glob('artifacts/persona_router/*/split.json'))
if not cand:
    print('  no frozen split found; skipping strict check'); raise SystemExit(0)
split = json.load(open(cand[0]))
bank_files = sorted(glob.glob('artifacts/persona_router/*/persona_bank.json'))
bank = json.load(open(bank_files[0])) if bank_files else None
items = set(resp.question_key.astype(str)); split_items=set()
for d in split['domains'].values():
    split_items |= set(d['calibration'])
    for f in d['target_folds']: split_items |= set(f)
assert not (split_items - items), 'MISSING split items in rebuilt data'
if bank is not None:
    bp=set()
    for db in bank['domains'].values(): bp |= set(map(str, db['panel_cluster'].keys()))
    tr=set(resp[resp.split.eq('train')].panel_id.astype(str))
    assert bp <= tr, 'persona_bank panels not all train in rebuilt data'
    dom=next(iter(bank['domains'])); db=bank['domains'][dom]
    cal=db['calibration']; labels={str(k):int(v) for k,v in db['panel_cluster'].items()}
    rows=resp[resp.split.eq('train') & resp.domain.eq(dom) & resp.question_key.isin(cal)]
    q=cal[0]; a=float(bank['alpha']); qr=rows[rows.question_key.eq(q)]; n=int(qr.n_options.iloc[0])
    zr=qr[qr.panel_id.map(labels).eq(0)]
    cnt=np.bincount(zr.answer_index.astype(int),minlength=n)+a; rec=(cnt/cnt.sum())
    fr=np.asarray(db['response_prob']['0'][q]); d=float(np.max(np.abs(rec-fr)))
    assert d < 1e-9, f'persona response prob mismatch {d}'
    print('  consistency OK: split items, train panels, persona prob reproduce exactly')
PY

echo "[6/6] tests"
PYTHONPATH=src python3 -m pytest -q tests/test_personadb.py tests/test_persona_comparison.py tests/test_persona_router.py 2>&1 | tail -3
echo "restore complete"
