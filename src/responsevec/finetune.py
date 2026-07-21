"""QLoRA fine-tuning upper-bound control (design §6.4 baseline 20).

This is the "spend real gradient on the backbone" ceiling ResponseVec is
measured against: a rank-16 adapter trained on the SAME canonical prompts, with
the K answers in history, target = the correct option-label token. Per-example
random label permutation blocks position shortcuts. If the frozen-backbone
ResponseVec head approaches this ceiling, the paper's efficiency claim lands; if
QLoRA dominates by a wide margin, that bound is reported honestly.

The training prompt is build_canonical_prompt (prompting_rv) — identical to what
every frozen method sees — so the only difference from ResponseVec is that
gradients reach the backbone here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .data import PanelStore
from .prompting_rv import (
    OPTION_LABELS,
    build_canonical_prompt,
    deterministic_permutation,
    option_token_ids,
    select_history,
)
from .utils import seed_everything, stable_int, write_json


def build_finetune_examples(
    store: PanelStore,
    k_values: Sequence[int],
    calibration_seed: int,
    include_history: bool,
    epoch: int,
    seed: int,
    retriever=None,
    selection: str = "random",
) -> Iterator[tuple[str, int, int]]:
    """Yields (prompt, correct_label_index, n_options) over TRAIN respondents,
    seen items only, one sampled K per (row, epoch). History uses the same
    leakage-safe selector as the frozen methods; the QLoRA baseline defaults to
    'random' selection so it does not require a retriever on the training box
    (semantic can be passed when a retriever is available)."""
    train = store.responses[store.responses["split"].eq("train") & ~store.responses["is_unseen_item"]]
    rows = train.to_dict("records")
    rng = np.random.default_rng(stable_int(seed, "finetune_epoch", epoch))
    order = rng.permutation(len(rows))
    for position in order:
        row = rows[int(position)]
        k = int(rng.choice(list(k_values)))
        history_rows: list[dict[str, Any]] = []
        if include_history and k > 0:
            source = store.history_rows(row["panel_id"], row["question_id"], k, calibration_seed).to_dict("records")
            history_rows = select_history(
                row["question"], source, k, retriever, selection=selection,
                random_seed=calibration_seed, panel_id=row["panel_id"],
            )
        permutation = deterministic_permutation(int(row["n_options"]), int(rng.integers(0, 2**31 - 1)), row["row_id"])
        prompt, correct_label, _ = build_canonical_prompt(row, history_rows, permutation)
        yield prompt, correct_label, int(row["n_options"])


def finetune_qlora(
    store: PanelStore,
    backbone_name: str,
    output_dir: str | Path,
    dtype: str = "bfloat16",
    quantization: str | None = "auto",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    max_steps: int = 1200,
    micro_batch: int = 4,
    gradient_accumulation: int = 16,
    lr: float = 2e-4,
    max_length: int = 512,
    k_values: Sequence[int] = (0, 1, 3, 5, 8),
    calibration_seed: int = 1701,
    include_history: bool = True,
    seed: int = 1701,
    log_every: int = 25,
) -> list[dict[str, Any]]:
    """Compact QLoRA loop. Loss = CE over the option-label token logits at the
    last position (identical read-out to CausalExtractor's direct path)."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    from .llm_rv import load_causal_backbone

    seed_everything(seed)
    model, tokenizer = load_causal_backbone(backbone_name, dtype=dtype, quantization=quantization)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=int(lora_rank), lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout), target_modules=list(target_modules), bias="none",
    )
    model = get_peft_model(model, peft_config)
    device = next(model.parameters()).device
    if device.type == "cpu" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        device = next(model.parameters()).device
    model.train()

    use_template = bool(getattr(tokenizer, "chat_template", None))
    if use_template:
        probe_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": "probe"}], tokenize=False, add_generation_prompt=True
        )
        option_ids = option_token_ids(
            tokenizer, len(OPTION_LABELS), continuation_prefix=probe_prefix, label_prefix=""
        )
    else:
        option_ids = option_token_ids(
            tokenizer, len(OPTION_LABELS), continuation_prefix="Answer:", label_prefix=" "
        )
    label_ids = torch.tensor(option_ids, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / max(1, int(0.03 * max_steps)))
        * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, max_steps)))),
    )

    def encode(prompts: list[str]):
        if use_template:
            prompts = [
                tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]
        return tokenizer(prompts, padding=True, truncation=True, max_length=max_length,
                         return_tensors="pt", add_special_tokens=not use_template).to(device)

    history: list[dict[str, Any]] = []
    step, accumulated, running = 0, 0, 0.0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        batch_prompts, batch_targets, batch_n = [], [], []
        for prompt, target, n in build_finetune_examples(store, k_values, calibration_seed, include_history, epoch, seed):
            batch_prompts.append(prompt)
            batch_targets.append(target)
            batch_n.append(n)
            if len(batch_prompts) < micro_batch:
                continue
            tokens = encode(batch_prompts)
            logits = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], use_cache=False).logits[:, -1, :]
            label_logits = logits.index_select(-1, label_ids)
            n_tensor = torch.tensor(batch_n, device=device)
            mask = torch.arange(len(label_ids), device=device).unsqueeze(0) < n_tensor.unsqueeze(1)
            label_logits = label_logits.masked_fill(~mask, torch.finfo(label_logits.dtype).min)
            loss = torch.nn.functional.cross_entropy(
                label_logits.float(), torch.tensor(batch_targets, device=device)
            ) / gradient_accumulation
            loss.backward()
            running += float(loss.item()) * gradient_accumulation
            accumulated += 1
            batch_prompts, batch_targets, batch_n = [], [], []
            if accumulated % gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == max_steps:
                record = {"step": step, "loss": running / max(1, accumulated), "lr": scheduler.get_last_lr()[0]}
                history.append(record)
                print(f"[finetune] {record}")
                running, accumulated = 0.0, 0
            if step >= max_steps:
                break
        epoch += 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    write_json(output_dir / "history.json", history)
    return history


def load_adapter(backbone_name: str, adapter_dir: str | Path, dtype: str = "bfloat16", quantization: str | None = "auto"):
    """Load backbone + trained adapter for evaluation-time extraction."""
    from peft import PeftModel

    from .llm_rv import load_causal_backbone

    model, tokenizer = load_causal_backbone(backbone_name, dtype=dtype, quantization=quantization)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer
