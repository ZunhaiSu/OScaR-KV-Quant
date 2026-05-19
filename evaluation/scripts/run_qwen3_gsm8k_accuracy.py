#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from oscar import Cache, DynamicCache, StaticCache  # noqa: E402
import transformers.cache_utils  # noqa: E402

transformers.cache_utils.DynamicCache = DynamicCache
transformers.cache_utils.StaticCache = StaticCache
transformers.cache_utils.Cache = Cache

from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from qwen3 import Qwen3ForCausalLM  # noqa: E402


DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")

VARIANTS: Dict[str, Dict[str, str]] = {
    "fa2": {
        "attn_backend": "flash_attention_2",
        "kv_rotation": "none",
        "kv_norm": "0",
    },
    "plain": {
        "attn_backend": "oscar",
        "kv_rotation": "none",
        "kv_norm": "0",
    },
    "hada_norm": {
        "attn_backend": "oscar",
        "kv_rotation": "hadamard",
        "kv_norm": "1",
    },
    "hadamard_only": {
        "attn_backend": "oscar",
        "kv_rotation": "hadamard",
        "kv_norm": "0",
    },
    "norm_only": {
        "attn_backend": "oscar",
        "kv_rotation": "none",
        "kv_norm": "1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic Qwen3 GSM8K accuracy slice.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="hada_norm")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--fewshot", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--quant_mode", default="k-channel")
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--print_generation", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(config, dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str):
            return getattr(torch, config_dtype)
        if isinstance(config_dtype, torch.dtype):
            return config_dtype
        return torch.float16
    return getattr(torch, dtype_name)


def first_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match is not None:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def parse_indices(indices: str) -> List[int]:
    out: List[int] = []
    for item in indices.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(item))
    return out


def build_prompt(dataset, fewshot: int, question: str) -> str:
    prompt = ""
    for idx in range(fewshot):
        prompt += f"Question: {dataset['train'][idx]['question']}\n"
        prompt += f"Answer: {dataset['train'][idx]['answer']}\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def main() -> None:
    args = parse_args()
    random.seed(0)
    torch.manual_seed(0)

    variant = VARIANTS[args.variant]
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = resolve_torch_dtype(config, args.dtype)
    config._attn_implementation = "flash_attention_2"
    config.attn_backend = variant["attn_backend"]
    config.num_bits = args.num_bits
    config.quant_mode = args.quant_mode
    config.group_size = args.group_size
    config.kv_rotation = variant["kv_rotation"]
    config.kv_norm = variant["kv_norm"]
    config.residual_block_size = 128

    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map={"": args.device},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="left",
        pad_token="</s>",
    )
    dataset = load_dataset("gsm8k", "main")

    indices = parse_indices(args.indices)
    correct = 0
    for idx in indices:
        prompt = build_prompt(dataset, args.fewshot, dataset["test"][idx]["question"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_attention_mask=True,
        ).to(args.device)
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        generated = tokenizer.decode(output[0].tolist()[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = first_gsm8k_answer(generated)
        gold = first_gsm8k_answer(dataset["test"][idx]["answer"])
        ok = pred == gold
        correct += int(ok)
        print(f"idx={idx} pred={pred} gold={gold} ok={ok}")
        if args.print_generation:
            print(f"generation[{idx}]={generated!r}")

    print(f"accuracy={correct}/{len(indices)} variant={args.variant}")


if __name__ == "__main__":
    main()
