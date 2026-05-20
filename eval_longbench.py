#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
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
from eval_long_bench import scorer_e  # noqa: E402
from evaluation.example import apply_offline_v_hadamard  # noqa: E402


DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
LONG_BENCH_E_DATASETS = [
    "qasper_e",
    "multifieldqa_en_e",
    "hotpotqa_e",
    "2wikimqa_e",
    "gov_report_e",
    "multi_news_e",
    "trec_e",
    "triviaqa_e",
    "samsum_e",
    "passage_count_e",
    "passage_retrieval_en_e",
    "lcc_e",
    "repobench-p_e",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_longbench_samples(dataset: str, prompt_template: str, data_dir: Path) -> List[dict]:
    path = data_dir / f"{dataset}.jsonl"
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            samples.append(
                {
                    "prompt": prompt_template.format(**item),
                    "answers": item.get("answers", [""]),
                    "all_classes": item.get("all_classes", None),
                    "length": item.get("length", 0),
                }
            )
    return samples


def truncate_prompt(tokenizer, prompt: str, max_input_len: int) -> str:
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if tokenized.numel() <= max_input_len:
        return prompt
    half = max_input_len // 2
    return tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
        tokenized[-half:], skip_special_tokens=True
    )


def resolve_torch_dtype(config, dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str):
            return getattr(torch, config_dtype)
        if isinstance(config_dtype, torch.dtype):
            return config_dtype
        return torch.float16
    return getattr(torch, dtype_name)


def dataset_list(arg: str) -> List[str]:
    if arg == "all":
        return list(LONG_BENCH_E_DATASETS)
    return [item.strip() for item in arg.split(",") if item.strip()]


def json_safe_score(score):
    if isinstance(score, dict):
        return {key: json_safe_score(value) for key, value in score.items()}
    if isinstance(score, np.generic):
        score = score.item()
    if isinstance(score, float) and np.isnan(score):
        return None
    return score


def build_model(args: argparse.Namespace):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = resolve_torch_dtype(config, args.dtype)
    config._attn_implementation = "flash_attention_2"
    config.attn_backend = "oscar"
    config.num_bits = 2
    config.quant_mode = "k-channel"
    config.group_size = 32
    config.kv_rotation = "hadamard"
    config.kv_norm = "1"
    config.residual_block_size = 128
    config.residual_evict_size = args.residual_evict_size

    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map={"": args.device},
    )
    model.eval()
    if args.offline_v_hadamard:
        had_dim = apply_offline_v_hadamard(model)
        print(f"Applied offline V Hadamard to Qwen3 v_proj/o_proj weights with had_dim={had_dim}")
    return model


def run_dataset(
    args: argparse.Namespace,
    model,
    tokenizer,
    dataset: str,
    prompt_template: str,
    max_new_tokens: int,
    output_dir: Path,
) -> Dict[str, float]:
    samples = load_longbench_samples(dataset, prompt_template, args.data_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    out_file = output_dir / f"{dataset}.jsonl"
    predictions, answers, lengths, all_classes = [], [], [], None
    start_idx = 0
    if args.resume and out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                predictions.append(row["pred"])
                answers.append(row["answers"])
                lengths.append(row["length"])
                all_classes = row["all_classes"]
        start_idx = len(predictions)
        if start_idx > len(samples):
            raise ValueError(f"{out_file} has {start_idx} rows but dataset only has {len(samples)} samples")

    if start_idx >= len(samples):
        metric_key = dataset[:-2] if dataset.endswith("_e") else dataset
        score = json_safe_score(scorer_e(metric_key, predictions, answers, lengths, all_classes))
        print(f"{dataset}: already complete ({start_idx}/{len(samples)}), score={score}")
        return score

    start = time.time()
    mode = "a" if start_idx else "w"
    with out_file.open(mode, encoding="utf-8") as fout:
        for idx, sample in enumerate(samples[start_idx:], start=start_idx):
            prompt = truncate_prompt(tokenizer, sample["prompt"], args.max_input_len)
            inputs = tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(args.device)

            seed_everything(args.seed)
            with torch.no_grad():
                output_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )

            input_len = inputs.input_ids.shape[1]
            pred = tokenizer.decode(output_ids[0].tolist()[input_len:], skip_special_tokens=True)
            row = {
                "pred": pred,
                "answers": sample["answers"],
                "all_classes": sample["all_classes"],
                "length": sample["length"],
                "input_tokens": input_len,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            predictions.append(row["pred"])
            answers.append(row["answers"])
            lengths.append(row["length"])
            all_classes = row["all_classes"]

            if idx == start_idx or (idx + 1) % args.log_every == 0:
                elapsed = time.time() - start
                generated = idx + 1 - start_idx
                rate = generated / max(elapsed, 1e-6)
                print(
                    f"{dataset}: [{idx + 1}/{len(samples)}] "
                    f"input_tokens={input_len} rate={rate:.3f} samples/s",
                    flush=True,
                )

    metric_key = dataset[:-2] if dataset.endswith("_e") else dataset
    score = scorer_e(metric_key, predictions, answers, lengths, all_classes)
    score = json_safe_score(score)
    print(f"{dataset}: score={score}")
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongBench-E runner for Qwen3 OScaR.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--datasets", default="all", help="Comma-separated *_e datasets or 'all'")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "longbench_data" / "data")
    parser.add_argument("--config_dir", type=Path, default=PROJECT_ROOT / "longbench_config")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "pred_e" / "oscar-qasper")
    parser.add_argument("--max_input_len", type=int, default=32768)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--residual_evict_size", type=int, default=256)
    parser.add_argument("--offline_v_hadamard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Append to existing JSONL files and skip completed rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset2prompt = load_json(args.config_dir / "dataset2prompt.json")
    dataset2maxlen = load_json(args.config_dir / "dataset2maxlen.json")
    selected = dataset_list(args.datasets)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
        pad_token="</s>",
    )
    model = build_model(args)

    manifest = {
        "model_path": args.model_path,
        "datasets": selected,
        "max_input_len": args.max_input_len,
        "max_samples": args.max_samples,
        "attn_backend": "oscar",
        "num_bits": 2,
        "quant_mode": "k-channel",
        "group_size": 32,
        "kv_rotation": "hadamard",
        "kv_norm": "1",
        "residual_block_size": 128,
        "residual_evict_size": args.residual_evict_size,
        "offline_v_hadamard": args.offline_v_hadamard,
        "dtype": args.dtype,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    scores = {}
    for dataset in selected:
        key = dataset[:-2] if dataset.endswith("_e") else dataset
        if key not in dataset2prompt:
            raise KeyError(f"Missing prompt template for dataset={dataset} key={key}")
        if key not in dataset2maxlen:
            raise KeyError(f"Missing max_new_tokens for dataset={dataset} key={key}")
        scores[dataset] = run_dataset(
            args,
            model,
            tokenizer,
            dataset,
            dataset2prompt[key],
            int(dataset2maxlen[key]),
            args.output_dir,
        )
        gc.collect()
        torch.cuda.empty_cache()

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(scores, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
