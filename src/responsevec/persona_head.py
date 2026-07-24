"""Train-only statistical persona head used by comparison experiments."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def fit_persona_head(train, option_vectors: Mapping[str, np.ndarray], labels,
                      personas: int, epochs: int = 100, lr: float = 1e-2,
                      weight_decay: float = 1e-4, device: str = "cpu",
                      validation=None, validation_mixture=None, patience: int = 10):
    """Fit the persona-option scoring head without validation/test labels."""
    import torch
    rank = next(iter(option_vectors.values())).shape[1]
    max_options = max(len(value) for value in option_vectors.values())
    x = np.zeros((len(train), max_options, rank), np.float32)
    valid = np.zeros((len(train), max_options), bool)
    for i, question in enumerate(train.question_key.astype(str)):
        values = option_vectors[question]
        x[i, :len(values)] = values
        valid[i, :len(values)] = True
    dev = torch.device(device)
    tx = torch.as_tensor(x, device=dev)
    mask = torch.as_tensor(valid, device=dev)
    target = torch.as_tensor(train.answer_index.to_numpy(), dtype=torch.long, device=dev)
    persona = torch.as_tensor([labels[str(p)] for p in train.panel_id], dtype=torch.long, device=dev)
    theta = torch.nn.Parameter(torch.zeros(personas, rank, device=dev))
    bias = torch.nn.Parameter(torch.zeros(max_options, device=dev))
    optimizer = torch.optim.Adam([theta, bias], lr=lr, weight_decay=weight_decay)
    best, state, stale = np.inf, None, 0
    if validation is not None and len(validation):
        vx = np.zeros((len(validation), max_options, rank), np.float32)
        vm = np.zeros((len(validation), max_options), bool)
        for i, question in enumerate(validation.question_key.astype(str)):
            values = option_vectors[question]; vx[i, :len(values)] = values; vm[i, :len(values)] = True
        vx = torch.as_tensor(vx, device=dev); vm = torch.as_tensor(vm, device=dev)
        vy = torch.as_tensor(validation.answer_index.to_numpy(), dtype=torch.long, device=dev)
        vp = torch.as_tensor(np.stack([validation_mixture[str(p)] for p in validation.panel_id]),
                             dtype=torch.float32, device=dev)
    for _ in range(int(epochs)):
        logits = torch.einsum("ncr,nr->nc", tx, theta[persona]) + bias
        loss = torch.nn.functional.cross_entropy(logits.masked_fill(~mask, -1e9), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if validation is not None and len(validation):
            with torch.no_grad():
                logits = torch.einsum("ncr,mr->nmc", vx, theta) + bias[None, None, :]
                conditional = torch.softmax(logits.masked_fill(~vm[:, None, :], -1e9), dim=2)
                mixed = torch.einsum("nm,nmc->nc", vp, conditional)
                score = torch.nn.functional.nll_loss(torch.log(mixed.clamp_min(1e-9)), vy).item()
            if score < best - 1e-6:
                best, stale = score, 0
                state = (theta.detach().cpu().clone(), bias.detach().cpu().clone())
            else:
                stale += 1
                if stale >= int(patience):
                    break
    if state is not None:
        return state[0].numpy(), state[1].numpy()
    return theta.detach().cpu().numpy(), bias.detach().cpu().numpy()


def persona_conditionals(head, option_vectors):
    theta, bias = head
    logits = np.asarray(option_vectors) @ theta.T + bias[:len(option_vectors), None]
    logits -= logits.max(axis=0, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=0, keepdims=True)).T
