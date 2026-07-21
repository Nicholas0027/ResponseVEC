"""Training and prediction from frozen representation caches.

Only the small option-aware heads receive gradients. The functions here are
used identically for raw causal, input-centric, and response-centric vectors;
this is the implementation-level fairness constraint behind H2/H3.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .decoder import DecoderConfig, ObjectiveWeights, OptionAwareDecoder, total_loss
from .direct import DirectLogitCalibrator
from .encode import stack_option_matrix
from .utils import seed_everything, write_json


@dataclass
class HeadArrays:
    rows: pd.DataFrame
    z: np.ndarray
    option_matrix: np.ndarray
    option_mask: np.ndarray
    log_prior: np.ndarray
    targets: np.ndarray
    ordinal_mask: np.ndarray
    direct_probabilities: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)


def subset_arrays(arrays: HeadArrays, question_keys: set[str] | frozenset[str]) -> HeadArrays:
    """Zero-copy-in-spirit item-role slice of a shared representation cache."""
    allowed = {str(key) for key in question_keys}
    keep = arrays.rows["question_key"].astype(str).isin(allowed).to_numpy()
    if not keep.any():
        raise ValueError("item-role slice contains no rows")
    return HeadArrays(
        rows=arrays.rows.loc[keep].reset_index(drop=True),
        z=arrays.z[keep], option_matrix=arrays.option_matrix[keep],
        option_mask=arrays.option_mask[keep], log_prior=arrays.log_prior[keep],
        targets=arrays.targets[keep], ordinal_mask=arrays.ordinal_mask[keep],
        direct_probabilities=(arrays.direct_probabilities[keep] if arrays.direct_probabilities is not None else None),
    )


@dataclass
class HeadFit:
    model: Any
    best_epoch: int
    best_validation_nll: float
    history: list[dict[str, float]]


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.where(mask > 0, logits, -np.inf)
    values = values - np.max(values, axis=-1, keepdims=True)
    probability = np.exp(values) * mask
    return probability / np.clip(probability.sum(axis=-1, keepdims=True), 1e-12, None)


def arrays_from_cache(
    cache,
    option_table: Mapping[str, np.ndarray],
    prior,
    train_item_keys: frozenset[str] | set[str],
    *,
    max_options: int,
) -> HeadArrays:
    rows = cache.read_rows().reset_index(drop=True)
    vectors = cache.read_vectors().astype(np.float32)
    if len(rows) != len(vectors):
        raise ValueError("cache row/vector alignment failed")
    option_matrix, option_mask = stack_option_matrix(
        option_table, rows["question_key"].astype(str).tolist(), max_options=max_options
    )
    log_prior = np.full((len(rows), max_options), -np.inf, dtype=np.float32)
    train_item_keys = {str(key) for key in train_item_keys}
    for index, row in enumerate(rows.itertuples(index=False)):
        n = int(row.n_options)
        probability = prior.predict(
            str(row.question_key), str(row.country), n,
            item_is_seen=str(row.question_key) in train_item_keys,
        )
        log_prior[index, :n] = np.log(np.clip(probability, 1e-12, None))
    direct = None
    if cache.has_logits:
        raw_logits = cache.read_logits().astype(np.float32)
        if raw_logits.shape[1] > max_options:
            raise ValueError("cached logits exceed configured max_options")
        padded_logits = np.full((len(rows), max_options), -np.inf, dtype=np.float32)
        padded_logits[:, : raw_logits.shape[1]] = raw_logits
        direct = _masked_softmax(padded_logits, option_mask)
    return HeadArrays(
        rows=rows,
        z=vectors,
        option_matrix=option_matrix,
        option_mask=option_mask,
        log_prior=log_prior,
        targets=rows["answer_index"].to_numpy(dtype=np.int64),
        ordinal_mask=rows["is_ordinal"].fillna(False).to_numpy(dtype=bool),
        direct_probabilities=direct,
    )


def arrays_from_respondent_cache(target_arrays: HeadArrays, respondent_cache) -> HeadArrays:
    """Broadcast a query-independent RespondentVec cache (one z per
    (panel_id, domain)) onto an existing target-row HeadArrays, replacing only
    z. `target_arrays` supplies option_matrix/option_mask/log_prior/targets —
    already-correct, target-specific arrays built the normal way (e.g. from the
    input_centric cache for the identical rows) — so RespondentVec reuses 100%
    of the tested target-construction logic and differs only in where z comes
    from. This is the intended way to score a query-independent representation
    with the same option-aware decoder used for every query-conditioned
    family (design §2.3.D: "the respondent vector is combined downstream with
    item/option representations").

    Rows whose (panel_id, domain) has no cached RespondentVec are dropped
    (never silently zero-filled), so a coverage gap fails loudly downstream
    rather than corrupting training with a fabricated vector.
    """
    respondent_rows = respondent_cache.read_rows().reset_index(drop=True)
    respondent_vectors = respondent_cache.read_vectors().astype(np.float32)
    if len(respondent_rows) != len(respondent_vectors):
        raise ValueError("respondent cache row/vector alignment failed")
    lookup = {
        (str(row.panel_id), str(row.domain)): respondent_vectors[index]
        for index, row in enumerate(respondent_rows.itertuples(index=False))
    }
    keys = list(zip(target_arrays.rows["panel_id"].astype(str), target_arrays.rows["domain"].astype(str)))
    keep = np.asarray([key in lookup for key in keys], dtype=bool)
    if not keep.any():
        raise ValueError("no target rows matched any cached RespondentVec (panel_id, domain)")
    filtered_rows = target_arrays.rows.loc[keep].reset_index(drop=True)
    filtered_keys = [key for key, flag in zip(keys, keep) if flag]
    z = np.stack([lookup[key] for key in filtered_keys]).astype(np.float32)
    return HeadArrays(
        rows=filtered_rows,
        z=z,
        option_matrix=target_arrays.option_matrix[keep],
        option_mask=target_arrays.option_mask[keep],
        log_prior=target_arrays.log_prior[keep],
        targets=target_arrays.targets[keep],
        ordinal_mask=target_arrays.ordinal_mask[keep],
        direct_probabilities=(
            target_arrays.direct_probabilities[keep] if target_arrays.direct_probabilities is not None else None
        ),
    )


def _tensor_batch(arrays: HeadArrays, indices: np.ndarray, device, *, direct: bool = False):
    import torch

    batch = {
        "z": torch.as_tensor(arrays.z[indices], dtype=torch.float32, device=device),
        "option_matrix": torch.as_tensor(arrays.option_matrix[indices], dtype=torch.float32, device=device),
        "option_mask": torch.as_tensor(arrays.option_mask[indices], dtype=torch.float32, device=device),
        "log_prior": torch.as_tensor(arrays.log_prior[indices], dtype=torch.float32, device=device),
        "target": torch.as_tensor(arrays.targets[indices], dtype=torch.long, device=device),
        "ordinal_mask": torch.as_tensor(arrays.ordinal_mask[indices], dtype=torch.bool, device=device),
    }
    if direct:
        if arrays.direct_probabilities is None:
            raise ValueError("direct calibrator requires a cache with logits")
        batch["direct_probabilities"] = torch.as_tensor(
            arrays.direct_probabilities[indices], dtype=torch.float32, device=device
        )
    return batch


def _validation_nll(model, arrays: HeadArrays, batch_size: int, device, *, direct: bool = False) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays)))
            batch = _tensor_batch(arrays, indices, device, direct=direct)
            if direct:
                log_probs = model(
                    batch["direct_probabilities"], batch["option_matrix"],
                    batch["option_mask"], batch["log_prior"],
                )
            else:
                log_probs = model(
                    batch["z"], batch["option_matrix"], batch["option_mask"], batch["log_prior"]
                )
            losses.extend((-log_probs[torch.arange(len(indices), device=device), batch["target"]]).cpu().tolist())
    return float(np.mean(losses))


def train_option_head(
    train: HeadArrays,
    validation: HeadArrays,
    *,
    projection_dim: int = 256,
    dropout: float = 0.1,
    temperature_init: float = 0.1,
    prior_eta: float = 0.5,
    learnable_eta: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    rps_lambda: float = 0.1,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    gradient_clip: float = 1.0,
    seed: int = 1701,
    device: str | None = None,
) -> HeadFit:
    import torch

    if not len(train) or not len(validation):
        raise ValueError("head training requires non-empty train and validation arrays")
    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = DecoderConfig(
        z_dim=int(train.z.shape[1]), o_dim=int(train.option_matrix.shape[2]),
        projection_dim=int(projection_dim), dropout=float(dropout),
        temperature_init=float(temperature_init), prior_eta_init=float(prior_eta),
        learnable_eta=bool(learnable_eta), max_options=int(train.option_mask.shape[1]),
    )
    model = OptionAwareDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    weights = ObjectiveWeights(rps_lambda=float(rps_lambda))
    rng = np.random.default_rng(seed)
    best_state, best_nll, best_epoch, bad_epochs = None, float("inf"), -1, 0
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(train))
        epoch_losses = []
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            batch = _tensor_batch(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = total_loss(model, batch, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation_nll = _validation_nll(model, validation, int(batch_size), device)
        history.append({
            "epoch": float(epoch), "train_loss": float(np.mean(epoch_losses)),
            "validation_nll": validation_nll,
        })
        if validation_nll < best_nll - 1e-6:
            best_nll, best_epoch = validation_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("head training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return HeadFit(model, int(best_epoch), float(best_nll), history)


def train_direct_head(
    train: HeadArrays,
    validation: HeadArrays,
    *,
    projection_dim: int = 256,
    dropout: float = 0.1,
    temperature_init: float = 1.0,
    prior_eta: float = 0.5,
    learnable_eta: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    gradient_clip: float = 1.0,
    seed: int = 1701,
    device: str | None = None,
) -> HeadFit:
    import torch

    if DirectLogitCalibrator is None:
        raise ImportError("torch is required for DirectLogitCalibrator")
    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DirectLogitCalibrator(
        o_dim=int(train.option_matrix.shape[2]), projection_dim=int(projection_dim),
        max_options=int(train.option_mask.shape[1]), dropout=float(dropout),
        temperature_init=float(temperature_init), prior_eta=float(prior_eta),
        learnable_eta=bool(learnable_eta),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(seed)
    best_state, best_nll, best_epoch, bad_epochs = None, float("inf"), -1, 0
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            batch = _tensor_batch(train, indices, device, direct=True)
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(
                batch["direct_probabilities"], batch["option_matrix"],
                batch["option_mask"], batch["log_prior"],
            )
            loss = torch.nn.functional.nll_loss(log_probs, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_nll = _validation_nll(model, validation, int(batch_size), device, direct=True)
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_nll": validation_nll})
        if validation_nll < best_nll - 1e-6:
            best_nll, best_epoch = validation_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("direct calibrator produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return HeadFit(model, int(best_epoch), float(best_nll), history)


def predict_head(model, arrays: HeadArrays, *, batch_size: int = 512, device: str | None = None, direct: bool = False) -> np.ndarray:
    import torch

    device = torch.device(device or next(model.parameters()).device)
    model = model.to(device).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(arrays), int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), len(arrays)))
            batch = _tensor_batch(arrays, indices, device, direct=direct)
            if direct:
                log_probs = model(batch["direct_probabilities"], batch["option_matrix"], batch["option_mask"], batch["log_prior"])
            else:
                log_probs = model(batch["z"], batch["option_matrix"], batch["option_mask"], batch["log_prior"])
            output.append(log_probs.exp().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def prediction_frame(arrays: HeadArrays, probabilities: np.ndarray, method: str) -> pd.DataFrame:
    if len(arrays) != len(probabilities):
        raise ValueError("prediction length mismatch")
    records = []
    for index, row in enumerate(arrays.rows.itertuples(index=False)):
        n = int(row.n_options)
        p = probabilities[index, :n].astype(float)
        p = p / np.clip(p.sum(), 1e-12, None)
        target = int(row.answer_index)
        predicted = int(p.argmax())
        record = row._asdict()
        record.update({
            "method": method,
            "probabilities_json": json.dumps(p.tolist()),
            "predicted_index": predicted,
            "nll": float(-np.log(max(p[target], 1e-12))),
            "brier": float(np.square(p - np.eye(n)[target]).sum()),
            "normalized_ordinal_error": float(abs(predicted - target) / max(1, n - 1)),
            "correct": int(predicted == target),
            "calibration_seed": int(getattr(row, "history_seed", 1701)),
        })
        records.append(record)
    return pd.DataFrame(records)


def average_option_seeds(predictions: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """Average semantic probabilities after option permutations are undone."""
    frame = predictions.copy()
    if method is not None:
        frame = frame[frame["method"].eq(method)]
    key_candidates = [
        "method", "protocol", "fold", "target_role", "row_id", "panel_id",
        "domain", "question_id", "question_key", "k", "split", "item_pool",
        "history_seed", "answer_index", "n_options", "survey_weight", "is_ordinal",
        "country", "sex", "age_bin", "education", "income_quintile",
        "employment", "marital_status", "urbanicity",
    ]
    keys = [key for key in key_candidates if key in frame.columns]
    records = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        context = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        probability = np.mean(
            [np.asarray(json.loads(value), dtype=float) for value in group["probabilities_json"]], axis=0
        )
        temp_rows = group.iloc[[0]].copy()
        temp_arrays = HeadArrays(
            rows=temp_rows, z=np.zeros((1, 1), np.float32),
            option_matrix=np.zeros((1, len(probability), 1), np.float32),
            option_mask=np.ones((1, len(probability)), np.float32),
            log_prior=np.zeros((1, len(probability)), np.float32),
            targets=np.asarray([int(context["answer_index"])]),
            ordinal_mask=np.asarray([bool(context.get("is_ordinal", False))]),
        )
        result = prediction_frame(temp_arrays, probability[None, :], str(context["method"])).iloc[0].to_dict()
        result.update(context)
        result["option_seed"] = -1
        records.append(result)
    return pd.DataFrame(records)


def average_decoder_seeds(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average probability predictions across ``_seed<integer>`` heads.

    Primary inference bootstraps these seed-averaged probabilities; the
    unaveraged files remain available for the hierarchical seed sensitivity
    analysis.
    """
    frame = predictions.copy()
    frame["base_method"] = frame["method"].astype(str).map(
        lambda value: re.sub(r"_seed\d+$", "", value)
    )
    key_candidates = [
        "base_method", "protocol", "fold", "target_role", "row_id", "panel_id",
        "domain", "question_id", "question_key", "k", "split", "item_pool",
        "history_seed", "option_seed", "answer_index", "n_options", "survey_weight",
        "is_ordinal", "country", "sex", "age_bin", "education", "income_quintile",
        "employment", "marital_status", "urbanicity",
    ]
    keys = [key for key in key_candidates if key in frame.columns]
    records = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        context = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        probability = np.mean(
            [np.asarray(json.loads(value), dtype=float) for value in group["probabilities_json"]], axis=0
        )
        first = group.iloc[[0]].drop(columns=["base_method"]).copy()
        temp_arrays = HeadArrays(
            rows=first, z=np.zeros((1, 1), np.float32),
            option_matrix=np.zeros((1, len(probability), 1), np.float32),
            option_mask=np.ones((1, len(probability)), np.float32),
            log_prior=np.zeros((1, len(probability)), np.float32),
            targets=np.asarray([int(context["answer_index"])]),
            ordinal_mask=np.asarray([bool(context.get("is_ordinal", False))]),
        )
        result = prediction_frame(temp_arrays, probability[None, :], str(context["base_method"])).iloc[0].to_dict()
        result.update({key: value for key, value in context.items() if key != "base_method"})
        result["method"] = str(context["base_method"])
        result["n_decoder_seeds"] = int(group["method"].nunique())
        records.append(result)
    return pd.DataFrame(records)


def save_head_fit(fit: HeadFit, directory: str | Path, metadata: Mapping[str, Any]) -> None:
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(fit.model.state_dict(), directory / "model.pt")
    write_json(directory / "fit.json", {
        **dict(metadata),
        "best_epoch": fit.best_epoch,
        "best_validation_nll": fit.best_validation_nll,
        "history": fit.history,
    })
