#!/usr/bin/env python3
"""Generate and validate SurveyBridge structured item records on one GPU.

The cache is append-only and resumable. Every prompt is keyed by SHA-256;
malformed outputs are retained with their error instead of silently replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON object in generation")


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["cache_key"]] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="cross_survey/metadata/surveybridge_item_prompts.jsonl")
    parser.add_argument("--schema", default="cross_survey/metadata/surveybridge_item_schema.json")
    parser.add_argument("--output", default="cross_survey/results/structured_items_qwen32b.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quantization", default="auto",
                        choices=["auto", "nf4", "none"],
                        help="auto detects AWQ/GPTQ from config; nf4 forces on-the-fly bnb 4-bit")
    args = parser.parse_args()

    import torch
    from jsonschema import validate
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines()
               if line.strip()]
    if args.limit:
        prompts = prompts[:args.limit]
    schema = json.loads(Path(args.schema).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(output)
    pending = [record for record in prompts if record["cache_key"] not in cache]
    print(f"prompts={len(prompts)} cached={len(cache)} pending={len(pending)}")
    if not pending:
        return

    use_nf4 = args.quantization == "nf4"
    if args.quantization == "auto":
        try:
            cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            qcfg = getattr(cfg, "quantization_config", None)
            if qcfg and qcfg.get("quant_method", "") in ("awq", "gptq", "fp8"):
                use_nf4 = False
                print(f"detected pre-quantized model ({qcfg['quant_method']}), skipping bnb NF4")
            else:
                use_nf4 = True
        except Exception:
            use_nf4 = True

    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if use_nf4:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            rendered = []
            for record in batch:
                messages = [{"role": "user", "content": record["prompt"]}]
                rendered.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                ))
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_input_length,
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = encoded["input_ids"].shape[1]
            for index, record in enumerate(batch):
                raw = tokenizer.decode(generated[index][width:], skip_special_tokens=True)
                result = {
                    "cache_key": record["cache_key"],
                    "item": record["item"],
                    "wave": record["wave"],
                    "family": record["family"],
                    "template": record["template"],
                    "model": args.model,
                    "raw": raw,
                    "valid": False,
                    "data": None,
                    "error": None,
                }
                try:
                    data = extract_json(raw)
                    data.pop("$schema", None)
                    validate(instance=data, schema=schema)
                    if data["item_id"] != record["item"]:
                        raise ValueError("item_id does not match prompt")
                    result["data"] = data
                    result["valid"] = True
                except Exception as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            done = min(start + len(batch), len(pending))
            valid = sum(1 for value in load_cache(output).values() if value.get("valid"))
            print(f"generated {done}/{len(pending)} pending; valid cache records={valid}")


if __name__ == "__main__":
    main()
