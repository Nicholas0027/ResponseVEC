#!/usr/bin/env python
"""B3 skeleton: Fisher-score semantic adaptation vs raw history vs base.

Answers ONE question:
  Does centering the historical choice by demographic-semantic expectation
  extract more transferable individual signal than raw chosen-option mean?

Usage:
  PYTHONPATH=src python scripts/fisher_score_experiment.py \
      --processed artifacts/processed \
      --option-table artifacts/cache/option_table.npz \
      --fold 0 --k 5 --k-curve --r 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# 1. New fold builder
# ---------------------------------------------------------------------------

def build_b3_folds(
    responses: pd.DataFrame,
    items: pd.DataFrame,
    option_table: dict[str, np.ndarray],
    n_calibration: int = 8,
    n_folds: int = 6,
    seed: int = 1701,
) -> dict:
    rng = np.random.default_rng(seed)
    folds: dict = {"calibration_pool": [], "target_folds": [[] for _ in range(n_folds)]}
    for domain in sorted(responses["domain"].unique()):
        dom_items = items[items["domain"].eq(domain)].copy()
        keys = dom_items["question_key"].astype(str).tolist()
        if len(keys) < n_calibration + n_folds:
            raise ValueError(f"{domain}: only {len(keys)} items")
        sem = np.stack([option_table[k].mean(axis=0) for k in keys])
        sem = sem / (np.linalg.norm(sem, axis=1, keepdims=True) + 1e-12)
        n_clusters = min(len(keys), n_calibration + 2)
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        ).fit_predict(sem)
        cal_idx: list[int] = []
        for c in range(n_clusters):
            candidates = np.where(labels == c)[0].tolist()
            if candidates:
                cal_idx.append(candidates[0])
                if len(cal_idx) >= n_calibration:
                    break
        remaining = [i for i in range(len(keys)) if i not in set(cal_idx)]
        while len(cal_idx) < n_calibration and remaining:
            cal_idx.append(remaining.pop(rng.integers(len(remaining))))
        cal_keys = {keys[i] for i in cal_idx}
        target_keys = [k for k in keys if k not in cal_keys]
        order = rng.permutation(len(target_keys))
        for j, idx in enumerate(order):
            folds["target_folds"][j % n_folds].append(target_keys[idx])
        folds["calibration_pool"].extend(sorted(cal_keys))
    return folds


# ---------------------------------------------------------------------------
# 2. Demographic vocabulary
# ---------------------------------------------------------------------------

DEMO_COLS = ["country", "sex", "age_bin", "education", "income_quintile"]


class DemoVocab:
    def __init__(self):
        self.maps: dict[str, dict[str, int]] = {}

    @classmethod
    def fit(cls, rows: pd.DataFrame) -> "DemoVocab":
        v = cls()
        for col in DEMO_COLS:
            vals = ["<UNK>"] + sorted(rows[col].dropna().astype(str).unique().tolist())
            v.maps[col] = {val: i for i, val in enumerate(vals)}
        return v

    def encode(self, rows: pd.DataFrame) -> np.ndarray:
        n = len(rows)
        out = np.zeros((n, len(DEMO_COLS)), dtype=np.int64)
        for j, col in enumerate(DEMO_COLS):
            for i, val in enumerate(rows[col].astype(str)):
                out[i, j] = self.maps[col].get(val, 0)
        return out

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.maps[c]) for c in DEMO_COLS]

    def group_key(self, row: pd.Series) -> tuple:
        return tuple(str(row.get(c, "")) for c in DEMO_COLS)

    def coarse_group_key(self, row: pd.Series) -> tuple:
        return tuple(str(row.get(c, "")) for c in ["country", "sex", "age_bin"])


# ---------------------------------------------------------------------------
# 3. Semantic demographic base model (M0)
# ---------------------------------------------------------------------------

class SemanticDemoBase(nn.Module):
    def __init__(self, card: list[int], dim: int, max_options: int = 11):
        super().__init__()
        self.dim = dim
        self.embeddings = nn.ModuleList([nn.Embedding(c, dim, padding_idx=0) for c in card])
        self.global_mean = nn.Parameter(torch.zeros(dim))
        self.position_bias = nn.Embedding(max_options, 1)
        nn.init.zeros_(self.position_bias.weight)

    def respondent_state(self, demo_idx: torch.Tensor) -> torch.Tensor:
        state = self.global_mean.unsqueeze(0).expand(demo_idx.shape[0], -1).clone()
        for j, emb in enumerate(self.embeddings):
            state = state + emb(demo_idx[:, j])
        return state

    def logits(self, option_matrix, option_mask, demo_idx):
        mu = self.respondent_state(demo_idx)
        scores = torch.einsum("ncd,nd->nc", option_matrix, mu)
        positions = torch.arange(option_matrix.shape[1], device=option_matrix.device)
        scores = scores + self.position_bias(positions).unsqueeze(0).squeeze(-1)
        scores = scores.masked_fill(option_mask == 0, float("-inf"))
        return scores

    def loss(self, option_matrix, option_mask, demo_idx, targets, l2=1e-3):
        logits = self.logits(option_matrix, option_mask, demo_idx)
        nll = nn.functional.cross_entropy(logits, targets, reduction="mean")
        reg = sum((emb.weight ** 2).sum() for emb in self.embeddings)
        reg += (self.global_mean ** 2).sum()
        return nll + l2 * reg


def train_base_model(train_data, val_data, card, dim, max_options, device="cpu",
                     epochs=100, lr=1e-3, l2=1e-3, patience=10):
    model = SemanticDemoBase(card, dim, max_options).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, best_ep = float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = model.loss(
            train_data["options"].to(device), train_data["mask"].to(device),
            train_data["demo"].to(device), train_data["targets"].to(device), l2=l2,
        )
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = model.logits(
                val_data["options"].to(device), val_data["mask"].to(device),
                val_data["demo"].to(device),
            )
            vn = nn.functional.cross_entropy(vl, val_data["targets"].to(device)).item()
        history.append({"epoch": epoch, "train": loss.item(), "val_nll": vn})
        if vn < best_val:
            best_val, best_ep = vn, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_ep >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_val_nll": best_val, "history": history}


# ---------------------------------------------------------------------------
# 4. Adaptation core
# ---------------------------------------------------------------------------

def base_probabilities(model, data, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model.logits(
            data["options"].to(device), data["mask"].to(device), data["demo"].to(device),
        )
        return torch.softmax(logits, dim=-1).cpu().numpy()


def compute_choice_scores(h_opts, h_targets, h_probs):
    chosen = h_opts[np.arange(len(h_targets)), h_targets]
    expected = np.einsum("nc,ncd->nd", h_probs, h_opts)
    return chosen - expected


def compute_fisher_matrices(h_opts, h_probs):
    expected = np.einsum("nc,ncd->nd", h_probs, h_opts)
    diff = h_opts - expected[:, None, :]
    weighted = diff * h_probs[:, :, None]
    return np.einsum("ncd,nce->nde", weighted, diff)


def predict_with_delta(base_logits, option_matrix, deltas, alpha=1.0):
    delta_scores = np.einsum("ncd,nd->nc", option_matrix, deltas) * alpha
    logits = base_logits + delta_scores
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def adapt_one(scores, fisher, raw, shuffled_raw, shuffled_scores, method, dim, alpha, tau):
    K = len(scores)
    if K == 0:
        return np.zeros(dim)
    if method == "raw":
        return (alpha / (tau + K)) * raw.sum(axis=0)
    if method == "shuffled":
        return (alpha / (tau + K)) * shuffled_scores.sum(axis=0)
    if method == "centered":
        return (alpha / (tau + K)) * scores.sum(axis=0)
    if method == "fisher":
        fsum = fisher.sum(axis=0)
        avg_trace = np.trace(fsum) / (dim * K + 1e-12)
        lam = tau * (avg_trace + 1e-3)
        reg = lam * np.eye(dim) + fsum
        return alpha * np.linalg.solve(reg, scores.sum(axis=0))
    raise ValueError(method)


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(preds, targets, mask, is_ordinal, resp_ids, name):
    eps = 1e-12
    p = np.clip(preds, eps, 1.0)
    onehot = np.zeros_like(preds)
    onehot[np.arange(len(targets)), targets] = 1.0
    nll = float(-np.mean(np.log(p[np.arange(len(targets)), targets])))
    brier = float(np.mean(((p - onehot) ** 2).sum(axis=1)))
    acc = float(np.mean(preds.argmax(axis=1) == targets))
    om = is_ordinal.astype(bool)
    rps = float("nan")
    if om.any():
        cp = np.cumsum(preds[om], axis=1)
        ct = np.cumsum(onehot[om], axis=1)
        rps = float(np.mean(((cp - ct) ** 2).sum(axis=1) / 2.0))
    per = defaultdict(list)
    for i, rid in enumerate(resp_ids):
        per[rid].append(-np.log(p[i, targets[i]]))
    macro = float(np.mean([np.mean(v) for v in per.values()]))
    return {"method": name, "nll": nll, "brier": brier, "accuracy": acc, "rps": rps,
            "macro_nll": macro, "n": len(targets)}


def pop_wd(preds, targets, is_ordinal, n_options, weights, question_keys):
    wds = []
    for qk in np.unique(question_keys):
        sel = question_keys == qk
        if not sel.any():
            continue
        if not bool(is_ordinal[sel][0]):
            continue
        nc = int(n_options[sel][0])
        w = weights[sel]
        w = w / (w.sum() + 1e-12)
        pd_ = np.average(preds[sel, :nc], axis=0, weights=w)
        od = np.zeros(nc)
        for a, wt in zip(targets[sel], w):
            od[a] += wt
        od /= od.sum()
        pd_ /= pd_.sum()
        pos = np.linspace(0, 1, nc)
        wds.append(wasserstein_distance(pos, pos, u_weights=od, v_weights=pd_))
    return float(np.mean(wds)) if wds else float("nan")


# ---------------------------------------------------------------------------
# 6. Data preparation
# ---------------------------------------------------------------------------

def prepare_arrays(rows, option_pca, vocab, dim, max_options):
    n = len(rows)
    demo_idx = vocab.encode(rows)
    options = np.zeros((n, max_options, dim), dtype=np.float32)
    mask = np.zeros((n, max_options), dtype=np.float32)
    for i, qk in enumerate(rows["question_key"].astype(str)):
        opts = option_pca[qk]
        c = opts.shape[0]
        options[i, :c] = opts
        mask[i, :c] = 1.0
    targets = rows["answer_index"].astype(int).to_numpy()
    return {
        "options": torch.from_numpy(options), "mask": torch.from_numpy(mask),
        "demo": torch.from_numpy(demo_idx), "targets": torch.from_numpy(targets),
        "rows": rows.reset_index(drop=True),
    }


def build_history_for_respondent(panel_id, responses, cal_keys, k, seed, option_pca, dim, max_options):
    panel = responses[(responses["panel_id"] == panel_id)
                      & (responses["question_key"].astype(str).isin(cal_keys))]
    if len(panel) == 0 or k == 0:
        return {"option_vectors": np.zeros((0, max_options, dim), np.float32),
                "targets": np.array([], np.int64), "question_keys": [],
                "n_options": np.array([], np.int64)}
    available = panel.sample(min(k, len(panel)), random_state=seed)
    h_opts, h_targets, h_qkeys, h_nopts = [], [], [], []
    for _, hr in available.iterrows():
        qk = str(hr["question_key"])
        opts = option_pca.get(qk)
        if opts is None:
            continue
        c = opts.shape[0]
        padded = np.zeros((max_options, dim), np.float32)
        padded[:c] = opts
        h_opts.append(padded)
        h_targets.append(int(hr["answer_index"]))
        h_qkeys.append(qk)
        h_nopts.append(c)
    return {
        "option_vectors": np.stack(h_opts) if h_opts else np.zeros((0, max_options, dim), np.float32),
        "targets": np.array(h_targets, np.int64),
        "question_keys": h_qkeys,
        "n_options": np.array(h_nopts, np.int64),
    }


def compute_history_base_probs(model, history, respondent_row, vocab, max_options, device="cpu"):
    if len(history["targets"]) == 0:
        return np.zeros((0, max_options), np.float32)
    demo = vocab.encode(respondent_row.to_frame().T)
    demo_t = torch.from_numpy(demo).expand(len(history["targets"]), -1).to(device)
    opts_t = torch.from_numpy(history["option_vectors"]).to(device)
    mask = np.zeros((len(history["targets"]), max_options), np.float32)
    for i, c in enumerate(history["n_options"]):
        mask[i, :c] = 1.0
    mask_t = torch.from_numpy(mask).to(device)
    model.eval()
    with torch.no_grad():
        logits = model.logits(opts_t, mask_t, demo_t)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def build_shuffled_history(all_respondent_data, responses, cal_keys, seed):
    rng = np.random.default_rng(seed + 999)
    groups = defaultdict(list)
    for row_id, rd in all_respondent_data.items():
        if len(rd["raw"]) > 0:
            groups[rd["group_key"]].append(row_id)
    shuffled = {}
    for row_id, rd in all_respondent_data.items():
        if len(rd["raw"]) == 0:
            shuffled[row_id] = rd["raw"].copy()
            continue
        grp = rd["group_key"]
        pool = groups[grp]
        if len(pool) <= 1:
            shuffled[row_id] = rd["raw"].copy()
            continue
        donor = pool[rng.integers(len(pool))]
        donor_raw = all_respondent_data[donor]["raw"]
        n = min(len(rd["raw"]), len(donor_raw))
        shuffled[row_id] = donor_raw[:n].copy() if n > 0 else rd["raw"].copy()
    return shuffled


# ---------------------------------------------------------------------------
# 7. Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    processed = Path(args.processed)
    responses = pd.read_parquet(processed / "responses.parquet")
    items = pd.read_parquet(processed / "items.parquet")
    raw_table = np.load(args.option_table, allow_pickle=False)
    option_table = {k: raw_table[k] for k in raw_table.files}

    all_vecs = np.vstack([option_table[k] for k in items["question_key"].astype(str)])
    pca = PCA(n_components=args.r, random_state=args.seed)
    pca.fit(all_vecs)
    option_pca = {k: pca.transform(option_table[k]).astype(np.float32)
                  for k in items["question_key"].astype(str)}
    dim = args.r
    max_options = int(responses.groupby("question_key")["n_options"].max().max()) if "n_options" in responses.columns else 11

    folds = build_b3_folds(responses, items, option_table, seed=args.seed)
    cal_keys = set(folds["calibration_pool"])
    fold_targets = folds["target_folds"][args.fold]
    n_total = len(fold_targets)
    n_train = max(1, int(n_total * 0.67))
    n_val = max(1, (n_total - n_train) // 2)
    train_items = set(fold_targets[:n_train])
    val_items = set(fold_targets[n_train:n_train + n_val])
    test_items = set(fold_targets[n_train + n_val:])

    train_resp = responses[responses["split"].eq("train")]
    val_resp = responses[responses["split"].eq("validation")]
    test_resp = responses[responses["split"].eq("test")]

    base_train = train_resp[train_resp["question_key"].astype(str).isin(cal_keys | train_items)].copy()
    base_val = val_resp[val_resp["question_key"].astype(str).isin(val_items)].copy()
    test_rows = test_resp[test_resp["question_key"].astype(str).isin(test_items)].copy()

    vocab = DemoVocab.fit(base_train)
    train_arr = prepare_arrays(base_train, option_pca, vocab, dim, max_options)
    val_arr = prepare_arrays(base_val, option_pca, vocab, dim, max_options)
    test_arr = prepare_arrays(test_rows, option_pca, vocab, dim, max_options)

    print(f"Fold {args.fold}: train={len(base_train)}, val={len(base_val)}, test={len(test_rows)} rows")
    print(f"  cal={len(cal_keys)}, train_items={len(train_items)}, val_items={len(val_items)}, test_items={len(test_items)}")

    print("Training base model...")
    model, info = train_base_model(train_arr, val_arr, vocab.cardinalities, dim, max_options,
                                   device=args.device, epochs=args.epochs, l2=args.l2)
    print(f"  Best val NLL: {info['best_val_nll']:.4f}")

    test_base_probs = base_probabilities(model, test_arr, args.device)
    test_base_logits = np.log(test_base_probs + 1e-12)

    print("Building respondent histories...")
    test_panel_ids = test_rows["panel_id"].unique()
    panel_history = {}
    panel_demo = {}
    for pid in test_panel_ids:
        panel_demo[pid] = test_rows[test_rows["panel_id"].eq(pid)].iloc[0]
        panel_history[pid] = build_history_for_respondent(
            pid, responses, cal_keys, 8, args.seed, option_pca, dim, max_options)

    respondent_data = {}
    for pid in test_panel_ids:
        hist = panel_history[pid]
        if len(hist["targets"]) == 0:
            respondent_data[pid] = {
                "scores": np.zeros((0, dim)), "fisher": np.zeros((0, dim, dim)),
                "raw": np.zeros((0, dim)), "shuffled_raw": np.zeros((0, dim)),
                "shuffled_scores": np.zeros((0, dim)),
                "group_key": vocab.coarse_group_key(panel_demo[pid]),
            }
            continue
        h_probs = compute_history_base_probs(model, hist, panel_demo[pid], vocab, max_options, args.device)
        scores = compute_choice_scores(hist["option_vectors"], hist["targets"], h_probs)
        fisher = compute_fisher_matrices(hist["option_vectors"], h_probs)
        raw = hist["option_vectors"][np.arange(len(hist["targets"])), hist["targets"]]
        respondent_data[pid] = {
            "scores": scores, "fisher": fisher, "raw": raw,
            "shuffled_raw": raw.copy(), "shuffled_scores": scores.copy(),
            "group_key": vocab.coarse_group_key(panel_demo[pid]),
            "hist": hist, "h_probs": h_probs,
        }

    # Build shuffled: swap answer targets within coarse demographic groups
    rng_shuffle = np.random.default_rng(args.seed + 999)
    groups = defaultdict(list)
    for pid in test_panel_ids:
        if len(respondent_data[pid]["raw"]) > 0:
            groups[respondent_data[pid]["group_key"]].append(pid)
    n_shuffled = 0
    for pid in test_panel_ids:
        rd = respondent_data[pid]
        if len(rd["raw"]) == 0:
            continue
        grp = rd["group_key"]
        pool = [p for p in groups[grp] if p != pid]
        if not pool:
            # Fallback: use all respondents as pool
            pool = [p for p in test_panel_ids if p != pid and len(respondent_data[p]["raw"]) > 0]
        if not pool:
            continue
        donor_pid = pool[rng_shuffle.integers(len(pool))]
        donor = respondent_data[donor_pid]
        hist = rd["hist"]
        h_probs = rd["h_probs"]
        K = len(hist["targets"])
        n_donor = min(K, len(donor["hist"]["targets"]))
        if n_donor == 0:
            continue
        # Use donor's answers on the same history questions (question-aligned shuffle)
        shuffled_targets = donor["hist"]["targets"][:n_donor].copy()
        shuffled_raw = hist["option_vectors"][:n_donor][np.arange(n_donor), shuffled_targets]
        shuffled_scores = compute_choice_scores(
            hist["option_vectors"][:n_donor], shuffled_targets, h_probs[:n_donor])
        rd["shuffled_raw"] = shuffled_raw
        rd["shuffled_scores"] = shuffled_scores

    k_values = [0, 1, 3, 5, 8] if args.k_curve else [args.k]
    all_results = []

    for k_val in k_values:
        print(f"\n--- K={k_val} ---")
        results = []

        for method in ["base", "raw", "centered", "fisher", "shuffled"]:
            if method == "base":
                preds = test_base_probs
            else:
                deltas = np.zeros((len(test_rows), dim), np.float32)
                for i, (_, row) in enumerate(test_rows.iterrows()):
                    pid = row["panel_id"]
                    rd = respondent_data[pid]
                    sc = rd["scores"][:k_val] if k_val > 0 else rd["scores"][:0]
                    fi = rd["fisher"][:k_val] if k_val > 0 else rd["fisher"][:0]
                    rw = rd["raw"][:k_val] if k_val > 0 else rd["raw"][:0]
                    sw = rd["shuffled_scores"][:k_val] if k_val > 0 else rd["shuffled_scores"][:0]
                    srw = rd["shuffled_raw"][:k_val] if k_val > 0 else rd["shuffled_raw"][:0]
                    deltas[i] = adapt_one(sc, fi, rw, srw, sw, method, dim, args.alpha, args.tau)
                preds = predict_with_delta(test_base_logits, test_arr["options"].numpy(), deltas, alpha=args.alpha)

            is_ord = test_rows["is_ordinal"].to_numpy() if "is_ordinal" in test_rows else np.ones(len(test_rows))
            sw_col = test_rows["survey_weight"].to_numpy() if "survey_weight" in test_rows else np.ones(len(test_rows))
            m = evaluate_predictions(
                preds, test_arr["targets"].numpy(), test_arr["mask"].numpy(),
                is_ord, test_rows["panel_id"].to_numpy(), f"M_{method}_K{k_val}",
            )
            m["k"] = k_val
            m["population_wd"] = pop_wd(
                preds, test_arr["targets"].numpy(), is_ord,
                test_rows["n_options"].to_numpy(), sw_col,
                test_rows["question_key"].astype(str).to_numpy(),
            )
            results.append(m)
            print(f"  {method:10s}  NLL={m['nll']:.4f}  Brier={m['brier']:.4f}  "
                  f"Acc={m['accuracy']:.4f}  WD={m['population_wd']:.4f}")
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 90)
    print(f"RESULTS (fold={args.fold}, r={args.r}, alpha={args.alpha}, tau={args.tau}, l2={args.l2})")
    print("=" * 90)
    print(df[["method", "k", "nll", "brier", "accuracy", "rps", "macro_nll", "population_wd"]].to_string(index=False))

    primary_k = args.k
    pri = df[df["k"].eq(primary_k)]
    base_nll = pri.loc[pri["method"].eq(f"M_base_K{primary_k}"), "nll"].values
    gates = {}
    if len(base_nll) > 0:
        base_nll = base_nll[0]
        for m in ["raw", "centered", "fisher", "shuffled"]:
            row = pri.loc[pri["method"].eq(f"M_{m}_K{primary_k}")]
            if len(row) > 0:
                gain = base_nll - row["nll"].values[0]
                gates[f"M_{m}_vs_base"] = {"gain": round(float(gain), 6), "passed": bool(gain >= 0.005)}
        fn = pri.loc[pri["method"].eq(f"M_fisher_K{primary_k}"), "nll"]
        sn = pri.loc[pri["method"].eq(f"M_shuffled_K{primary_k}"), "nll"]
        if len(fn) > 0 and len(sn) > 0:
            gap = float(sn.values[0] - fn.values[0])
            gates["real_vs_shuffled"] = {"gain": round(gap, 6), "passed": bool(gap >= 0.01)}
    fisher_curve = df[df["method"].str.startswith("M_fisher")].sort_values("k")["nll"].tolist()
    if len(fisher_curve) >= 2:
        mono = all(fisher_curve[i] >= fisher_curve[i + 1] - 0.002 for i in range(len(fisher_curve) - 1))
        gates["k_curve_monotonic"] = {"passed": bool(mono), "curve": [round(x, 4) for x in fisher_curve]}
    base_wd = pri.loc[pri["method"].eq(f"M_base_K{primary_k}"), "population_wd"]
    fisher_wd = pri.loc[pri["method"].eq(f"M_fisher_K{primary_k}"), "population_wd"]
    if len(base_wd) > 0 and len(fisher_wd) > 0:
        wd_diff = float(fisher_wd.values[0] - base_wd.values[0])
        gates["wd_not_degraded"] = {"diff": round(wd_diff, 6), "passed": bool(wd_diff <= 0.005)}

    print("\nGATES:")
    print(json.dumps(gates, indent=2))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    with open(out / "gates.json", "w") as f:
        json.dump(gates, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fisher-score semantic adaptation experiment")
    p.add_argument("--processed", default="artifacts/processed")
    p.add_argument("--option-table", default="artifacts/cache/option_table.npz")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--k-curve", action="store_true", help="run K=0,1,3,5,8")
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="artifacts/fisher_score_results")
    run_experiment(p.parse_args())
