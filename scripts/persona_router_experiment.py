#!/usr/bin/env python
"""Persona Bank Lite doubly-cold SocioBench experiment."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from responsevec.encode import load_option_table
from responsevec.persona_router import (DEMO_COLS, assign_personas, build_item_split,
    fit_demographic_router, fit_persona_bank, fold_roles, question_aligned_shuffle,
    router_prior, stable_history, update_posterior)


def seeds(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError: pass


def fit_head(train, val, vectors, labels, val_prior, m, r, maxopt, args):
    import torch
    device = torch.device(args.device)
    theta = torch.nn.Parameter(torch.zeros(m, r, device=device))
    bias = torch.nn.Parameter(torch.zeros(maxopt, device=device))
    opt = torch.optim.Adam([theta, bias], lr=args.lr, weight_decay=args.weight_decay)
    def batch(frame):
        x = np.zeros((len(frame), maxopt, r), np.float32)
        valid = np.zeros((len(frame), maxopt), bool)
        for i, q in enumerate(frame.question_key):
            n = len(vectors[q]); x[i, :n] = vectors[q]; valid[i, :n] = True
        return (torch.tensor(x, device=device), torch.tensor(valid, device=device),
                torch.tensor(frame.answer_index.to_numpy(), device=device, dtype=torch.long))
    tx, tm, ty = batch(train); vx, vm, vy = batch(val)
    tz = torch.tensor([labels[p] for p in train.panel_id], device=device)
    vp = torch.tensor(np.stack([val_prior[p] for p in val.panel_id]), device=device, dtype=torch.float32)
    best, state, stale = np.inf, None, 0
    for _ in range(args.epochs):
        opt.zero_grad()
        logits = torch.einsum("ncr,nr->nc", tx, theta[tz]) + bias
        logits = logits.masked_fill(~tm, -1e9)
        loss = torch.nn.functional.cross_entropy(logits, ty)
        loss.backward(); opt.step()
        with torch.no_grad():
            logits = torch.einsum("ncr,mr->nmc", vx, theta) + bias[None, None, :]
            logits = logits.masked_fill(~vm[:, None, :], -1e9)
            cp = torch.softmax(logits, dim=2)
            mixed = torch.einsum("nm,nmc->nc", vp, cp)
            score = torch.nn.functional.nll_loss(torch.log(mixed.clamp_min(1e-9)), vy).item()
        if score < best - 1e-6:
            best, stale = score, 0; state = (theta.detach().cpu().clone(), bias.detach().cpu().clone())
        else:
            stale += 1
            if stale >= args.patience: break
    return state[0].numpy(), state[1].numpy(), best


def conditional(head, option_vectors):
    theta, bias = head; logits = option_vectors @ theta.T + bias[:len(option_vectors), None]
    logits -= logits.max(0, keepdims=True); p = np.exp(logits); return (p / p.sum(0, keepdims=True)).T


def metrics(frame):
    maxn = int(frame.n_options.max()); p = np.zeros((len(frame), maxn))
    for i, x in enumerate(frame.probability): p[i, :len(x)] = x
    y = frame.answer_index.to_numpy(int)
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()
    brier = ((p-oh)**2).sum(1).mean(); acc = (p.argmax(1) == y).mean()
    ordinal = frame.is_ordinal.to_numpy(bool)
    rps_values = [((np.cumsum(a[:n-1])-np.cumsum(b[:n-1]))**2).sum()/(n-1)
                  for a,b,n,keep in zip(p,oh,frame.n_options,ordinal) if keep and n > 1]
    rps = float(np.mean(rps_values)) if rps_values else np.nan
    wd = []
    for _, g in frame.groupby("question_key"):
        if not bool(g.is_ordinal.iloc[0]): continue
        n = int(g.n_options.iloc[0]); obs = np.bincount(g.answer_index, minlength=n).astype(float); obs /= obs.sum()
        weights = g.survey_weight.to_numpy(float); weights /= weights.sum()
        pred = np.average(np.stack(g.probability)[:, :n], axis=0, weights=weights)
        observed = np.zeros(n)
        for answer, weight in zip(g.answer_index.astype(int), weights): observed[answer] += weight
        positions = np.linspace(0.0, 1.0, n)
        wd.append(wasserstein_distance(positions, positions, observed, pred))
    return {"nll": nll, "brier": brier, "accuracy": acc, "rps": rps, "population_wd": np.mean(wd), "rows": len(frame)}


def paired_nll_gap(pred, k, left="stat_history", right="stat_shuffled", seed=1701, draws=2000):
    """Return respondent-bootstrap NLL(right)-NLL(left) and its 95% CI."""
    subset = pred[pred.k.eq(k) & pred.method.isin([left, right])]
    wide = subset.pivot(index=["panel_id", "row_id", "answer_index"], columns="method", values="probability")
    if left not in wide or right not in wide: return np.nan, np.nan, np.nan
    losses = wide.reset_index()
    losses["gap"] = [np.log(max(a[int(y)], 1e-12))-np.log(max(b[int(y)], 1e-12))
                     for y,a,b in zip(losses.answer_index, losses[left], losses[right])]
    person = losses.groupby("panel_id").gap.mean().to_numpy()
    rng = np.random.default_rng(seed + int(k)); boot = rng.choice(person, (draws, len(person)), replace=True).mean(1)
    return float(person.mean()), *map(float, np.quantile(boot, [0.025, 0.975]))


def fit_temperature(rows):
    from scipy.optimize import minimize_scalar
    def objective(log_temperature):
        temperature = np.exp(log_temperature); losses = []
        for row in rows:
            p = np.power(np.maximum(row["probability"], 1e-12), 1.0 / temperature); p /= p.sum()
            losses.append(-np.log(max(p[row["answer_index"]], 1e-12)))
        return float(np.mean(losses))
    result = minimize_scalar(objective, bounds=(-3.0, 5.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(row, temperature, suffix=""):
    updated = dict(row); p = np.power(np.maximum(row["probability"], 1e-12), 1.0 / temperature)
    updated["probability"] = p / p.sum(); updated["method"] = row["method"] + suffix
    return updated


def llm_predictions(args, test, catalogue, banks, priors, real_hist, posteriors):
    from responsevec.llm_rv import CausalExtractor, choose_device, load_causal_backbone
    model, tok = load_causal_backbone(args.llm_model, args.llm_dtype, args.llm_quantization)
    ext = CausalExtractor(model, tok, choose_device(), args.llm_max_length, args.llm_batch_size)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; prompts, layout = [], []
    for row in test.itertuples():
        db = banks[row.domain]; opts = json.loads(row.options_json); hist = real_hist[row.panel_id][:args.k]
        rendered = []
        for q, answer in hist:
            source = catalogue.loc[q]
            source_options = json.loads(source.options_json)
            rendered.append(f"{source.question} -> {source_options[answer]}")
        htext = "; ".join(rendered) or "None"
        for kind in ("llm_demographic", "llm_history", "llm_persona"):
            persona_ids = range(banks["m"]) if kind == "llm_persona" else [0]
            for z in persona_ids:
                persona = db["persona_text"][str(z)] if kind == "llm_persona" else ""
                history = htext if kind == "llm_history" else "None"
                options = "\n".join(f"{labels[i]}. {o}" for i,o in enumerate(opts))
                prompts.append(
                    "Task: Predict how this real survey respondent is likely to answer the target question. "
                    "Return exactly one option letter.\n\n"
                    f"Respondent background:\n{row.demographic_text}\n\n"
                    f"Behaviorally grounded persona:\n{persona or 'Not supplied.'}\n\n"
                    f"Relevant previous responses:\n{history}\n\n"
                    f"Target question:\n{row.question}\n\n"
                    f"Options:\n{options}\n\nAnswer:"
                )
                layout.append((kind, row.row_id, row.panel_id, row.domain, z))
    raw = ext.extract(prompts, [int(test.set_index('row_id').loc[x[1], 'n_options']) for x in layout])["probabilities"]
    grouped = {}
    for meta, p in zip(layout, raw): grouped.setdefault(meta[:4], []).append(p)
    out = []
    index = test.set_index("row_id")
    for (kind, rid, pid, domain), cp in grouped.items():
        row = index.loc[rid]; conditionals = np.stack(cp)
        if kind == "llm_persona":
            out.append(record(row, rid, "llm_persona_prior", args.k,
                              np.asarray(priors[domain][pid]) @ conditionals))
            out.append(record(row, rid, "llm_persona_history", args.k,
                              np.asarray(posteriors[pid]) @ conditionals))
        else:
            out.append(record(row, rid, kind, args.k, conditionals[0]))
    return out


def record(row, rid, method, k, probability):
    return {"row_id": rid, "panel_id": row.panel_id, "domain": row.domain, "question_key": row.question_key,
            "answer_index": int(row.answer_index), "n_options": int(row.n_options),
            "is_ordinal": bool(row.is_ordinal), "survey_weight": float(row.survey_weight),
            "method": method, "k": k,
            "probability": np.asarray(probability)[:int(row.n_options)]}


def run(args):
    seeds(args.seed); processed = Path(args.processed); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    responses = pd.read_parquet(processed/"responses.parquet"); items = pd.read_parquet(processed/"items.parquet")
    table = load_option_table(args.option_table); split = build_item_split(items, table, seed=args.seed)
    roles = fold_roles(split, args.fold); (out/"split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    train_all = responses[responses.split.eq("train")]; val_all = responses[responses.split.eq("validation")]
    test_all = responses[responses.split.eq("test")]; banks = fit_persona_bank(train_all, roles["calibration"], args.personas, seed=args.seed)
    (out/"persona_bank.json").write_text(json.dumps(banks, indent=2), encoding="utf-8")
    allvec = np.vstack([table[q] for q in sorted(table)]); r = min(args.r, allvec.shape[0], allvec.shape[1])
    pca = PCA(r, random_state=args.seed).fit(allvec); vectors = {q:pca.transform(v).astype(np.float32) for q,v in table.items()}
    heads, priors, validation_posteriors = {}, {}, {}
    for domain, db in banks["domains"].items():
        tr = train_all[(train_all.domain.eq(domain)) & train_all.question_key.isin(roles["calibration"]+roles["train"])]
        va = val_all[(val_all.domain.eq(domain)) & val_all.question_key.isin(roles["validation"])]
        labels = db["panel_cluster"]; router = fit_demographic_router(train_all[train_all.domain.eq(domain)], labels)
        priors[domain] = router_prior(router, pd.concat([va, test_all[test_all.domain.eq(domain)]]), args.personas)
        val_mixture = {}
        for pid, panel in val_all[val_all.domain.eq(domain)].groupby("panel_id"):
            order = stable_history(pid, db["calibration"], args.k, args.seed)
            observed = panel[panel.question_key.isin(order)]
            answers = dict(zip(observed.question_key, observed.answer_index.astype(int)))
            history = [(q, answers[q]) for q in order if q in answers]
            val_mixture[pid] = update_posterior(priors[domain][pid], history, db["response_prob"])
            validation_posteriors[pid] = val_mixture[pid]
        heads[domain] = fit_head(tr, va, vectors, labels, val_mixture, args.personas, r, int(responses.n_options.max()), args)[:2]
    test = test_all[test_all.question_key.isin(roles["test"])].copy(); histories, shuffled, real = {}, {}, {}
    for pid, group in test.groupby("panel_id"):
        domain = group.domain.iloc[0]; cal = banks["domains"][domain]["calibration"]
        order = stable_history(pid, cal, max(args.k_values), args.seed)
        available = responses[(responses.panel_id.eq(pid)) & responses.question_key.isin(order)]
        amap = dict(zip(available.question_key, available.answer_index.astype(int))); histories[pid] = [q for q in order if q in amap]
        real[pid] = [(q, amap[q]) for q in histories[pid]]
    shuffled = question_aligned_shuffle(test_all[test_all.panel_id.isin(test.panel_id.unique())], histories, args.seed+1)
    rows = []; posterior_at_k = {}
    for k in args.k_values:
        for row in test.itertuples():
            db = banks["domains"][row.domain]; prior = priors[row.domain][row.panel_id]
            rp = update_posterior(prior, real[row.panel_id][:k], db["response_prob"])
            sp = update_posterior(prior, shuffled[row.panel_id][:k], db["response_prob"])
            if k == args.k: posterior_at_k[row.panel_id] = rp
            cp = conditional(heads[row.domain], vectors[row.question_key])
            import hashlib
            random_z = int.from_bytes(hashlib.sha256(f"{args.seed}|{row.panel_id}".encode()).digest()[:4], "little") % args.personas
            mixes = {"stat_demographic":prior, "stat_history":rp, "stat_shuffled":sp,
                     "stat_uniform":np.ones(args.personas)/args.personas,
                     "stat_random":np.eye(args.personas)[random_z],
                     "stat_hard_demographic":np.eye(args.personas)[int(np.argmax(prior))]}
            if k == max(args.k_values): mixes["stat_oracle_full"] = rp
            for method, weights in mixes.items(): rows.append(record(row, row.row_id, method, k, weights @ cp))
    if args.run_llm:
        if not posterior_at_k:
            for pid, group in test.groupby("panel_id"):
                domain=group.domain.iloc[0]; db=banks["domains"][domain]
                posterior_at_k[pid]=update_posterior(priors[domain][pid], real[pid][:args.k], db["response_prob"])
        validation = val_all[val_all.question_key.isin(roles["validation"])].copy()
        val_real = {}
        for pid, group in validation.groupby("panel_id"):
            domain = group.domain.iloc[0]; cal = banks["domains"][domain]["calibration"]
            order = stable_history(pid, cal, args.k, args.seed)
            observed = val_all[(val_all.panel_id.eq(pid)) & val_all.question_key.isin(order)]
            answers = dict(zip(observed.question_key, observed.answer_index.astype(int)))
            val_real[pid] = [(q, answers[q]) for q in order if q in answers]
        catalogue = responses.drop_duplicates("question_key").set_index("question_key")
        combined = pd.concat([validation, test], ignore_index=True)
        llm_rows = llm_predictions(args, combined, catalogue, banks["domains"]|{"m":args.personas},
                                   priors, val_real | real, validation_posteriors | posterior_at_k)
        validation_ids = set(validation.row_id); val_rows = [x for x in llm_rows if x["row_id"] in validation_ids]
        test_rows = [x for x in llm_rows if x["row_id"] not in validation_ids]
        temperatures = {}
        for method in sorted({x["method"] for x in val_rows}):
            temperatures[method] = fit_temperature([x for x in val_rows if x["method"] == method])
        (out/"llm_temperatures.json").write_text(json.dumps(temperatures, indent=2), encoding="utf-8")
        rows += [apply_temperature(x, temperatures[x["method"]]) for x in test_rows]
        rows += [apply_temperature(x, 1.0, suffix="_uncalibrated") for x in test_rows]
        val_frame = pd.DataFrame(val_rows)
        val_frame["probabilities_json"] = val_frame.probability.map(lambda x: json.dumps(x.tolist()))
        val_frame.drop(columns="probability").to_parquet(out/"llm_validation_predictions.parquet", index=False)
    pred = pd.DataFrame(rows); pred["probabilities_json"] = pred.probability.map(lambda x: json.dumps(x.tolist())); pred.drop(columns="probability").to_parquet(out/"predictions.parquet", index=False)
    result = []
    for (method,k), g in pred.groupby(["method","k"]): result.append({"method":method,"k":k,**metrics(g)})
    rdf = pd.DataFrame(result); rdf["real_shuffle_nll_gap"] = np.nan
    rdf["gap_ci_low"] = np.nan; rdf["gap_ci_high"] = np.nan
    for k in args.k_values:
        a=rdf[(rdf.method.eq("stat_history"))&(rdf.k.eq(k))]
        gap, low, high = paired_nll_gap(pred, k, seed=args.seed)
        if len(a):
            rdf.loc[a.index,"real_shuffle_nll_gap"] = gap
            rdf.loc[a.index,"gap_ci_low"] = low; rdf.loc[a.index,"gap_ci_high"] = high
    rdf.to_csv(out/"results.csv", index=False)
    max_k = max(args.k_values); gap, low, high = paired_nll_gap(pred, max_k, seed=args.seed)
    gates = {"fold": args.fold, "development_only": args.fold == 0, "k": max_k,
             "real_minus_shuffle_nll_gain": gap, "bootstrap_95_ci": [low, high],
             "pass_history_signal": bool(np.isfinite(low) and low > 0.0),
             "run_llm_recommended": bool(np.isfinite(low) and low > 0.0)}
    if "llm_persona_history" in set(pred.method):
        head_gap, head_low, head_high = paired_nll_gap(
            pred, args.k, left="stat_history", right="llm_persona_history", seed=args.seed)
        router_gain, router_low, router_high = paired_nll_gap(
            pred, args.k, left="llm_persona_history", right="llm_persona_prior", seed=args.seed)
        gates.update({
            "llm_head_minus_statistical_nll": head_gap,
            "llm_head_difference_95_ci": [head_low, head_high],
            "llm_router_history_gain": router_gain,
            "llm_router_history_gain_95_ci": [router_low, router_high],
            "llm_head_beats_statistical": bool(np.isfinite(head_high) and head_high < 0.0),
        })
    (out/"gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(rdf.to_string(index=False)); print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--processed",default="artifacts/processed"); p.add_argument("--option-table",default="artifacts/cache/option_table.npz"); p.add_argument("--output",default="artifacts/persona_router"); p.add_argument("--fold",type=int,default=0); p.add_argument("--seed",type=int,default=1701); p.add_argument("--personas",type=int,default=8); p.add_argument("--r",type=int,default=32); p.add_argument("--k-values",type=int,nargs="+",default=[0,1,3,5,8]); p.add_argument("--k",type=int,default=5); p.add_argument("--epochs",type=int,default=100); p.add_argument("--patience",type=int,default=10); p.add_argument("--lr",type=float,default=1e-2); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--device",default="cpu"); p.add_argument("--run-llm",action="store_true"); p.add_argument("--llm-model",default="Qwen/Qwen3-8B"); p.add_argument("--llm-dtype",default="bfloat16"); p.add_argument("--llm-quantization",default="nf4"); p.add_argument("--llm-max-length",type=int,default=512); p.add_argument("--llm-batch-size",type=int,default=8); run(p.parse_args())
