"""
LongBench-E evaluation script for OScaR KV cache quantization.

Usage:
    # Run one experiment (one model + one mode + one dataset)
    CUDA_VISIBLE_DEVICES=0 python eval_longbench_batch.py --model qwen3_8b --mode oscar2_rsqrt --dataset qasper_e

    # Evaluate results
    python eval_long_bench.py --path pred/qwen3_8b_oscar2_rsqrt/ --e
"""

import os
import sys
import json
import argparse
import gc
import time
from types import SimpleNamespace

import torch
import numpy as np
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from kv_cache_compression.monkeypatch import replace_qwen3
from kv_cache_compression.quarot_utils import rotate_qwen3_ov_proj


DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
MODEL_PATHS = {
    "qwen3_8b": DEFAULT_MODEL_PATH,
}
MODEL_NAMES = {"qwen3_8b": "Qwen3-8B"}
MODEL_DTYPES = {"qwen3_8b": torch.bfloat16}
MODEL_MAX_LEN = {"qwen3_8b": 32768}

# Default max_new_tokens per task type
DATASET_NUM_TOKENS = {
    "lcc": 64, "repobench-p": 64,
    "gov_report": 512, "multi_news": 512, "qmsum": 512,
    "narrativeqa": 128, "qasper": 128, "samsum": 128,
    "hotpotqa": 32, "2wikimqa": 32, "triviaqa": 32,
    "musique": 32, "passage_count": 32,
    "passage_retrieval_en": 32, "passage_retrieval_zh": 32,
    "multifieldqa_en": 64, "multifieldqa_zh": 64,
    "trec": 64, "lsht": 64,
}


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(dataset_name):
    """Load a LongBench dataset and format prompts."""
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "longbench_data", "data", f"{dataset_name}.jsonl")
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "longbench_config")
    dataset2prompt = json.load(open(os.path.join(config_dir, "dataset2prompt.json")))
    prompt_key = dataset_name[:-2] if dataset_name.endswith("_e") else dataset_name
    prompt_template = dataset2prompt[prompt_key]

    samples = []
    with open(data_file, "r") as f:
        for line in f:
            sample = json.loads(line.strip())
            prompt = prompt_template.format(**sample)
            samples.append({
                "prompt": prompt,
                "answers": sample.get("answers", [""]),
                "all_classes": sample.get("all_classes", None),
                "length": sample.get("length", len(prompt)),
            })
    return samples


def truncate_prompt(tokenizer, prompt, max_len):
    """Truncate prompt to max_len tokens (keep first half + last half)."""
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > max_len:
        half = max_len // 2
        prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + \
                 tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
    return prompt


def _compress_args(args, **overrides):
    values = dict(
        k_bits=2, v_bits=2,
        k_groupsize=32, v_groupsize=32,
        k_sym=False, v_sym=False,
        k_clip_ratio=1.0, v_clip_ratio=1.0,
        residual_length=getattr(args, "residual_length", 128),
        k_token_rotation=False,
        k_norm_factoring="norm_rsqrt",
        use_hadamard=True,
        offline_v_hadamard=getattr(args, "offline_v_hadamard", False),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def load_model(model_path, dtype, mode, model_type, args=None):
    """Load model with appropriate attention implementation."""
    if mode == "baseline":
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl, trust_remote_code=True,
    )
    model.eval()

    if args is not None and args.offline_v_hadamard:
        had_dim = getattr(model.config, "head_dim", None)
        if had_dim is None:
            had_dim = model.config.hidden_size // model.config.num_attention_heads
        print(f"Applying offline V/O Hadamard rotation: had_dim={had_dim}")
        rotate_qwen3_ov_proj(model, had_dim=had_dim)

    if mode == "oscar2_rsqrt":
        compress_args = _compress_args(args)
        replace_qwen3(compress_args, model, "oscar")

    elif mode == "oscar2_base":
        compress_args = _compress_args(args, k_norm_factoring=False)
        replace_qwen3(compress_args, model, "oscar")

    elif mode == "oscar2_no_hadamard":
        compress_args = _compress_args(args, use_hadamard=False, offline_v_hadamard=False)
        replace_qwen3(compress_args, model, "oscar")

    elif mode == "oscar4_rsqrt":
        compress_args = _compress_args(args, k_bits=4, v_bits=4)
        replace_qwen3(compress_args, model, "oscar")

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3_8b"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "oscar2_rsqrt", "oscar2_base",
                                 "oscar2_no_hadamard", "oscar4_rsqrt"])
    parser.add_argument("--dataset", type=str, required=True,
                        help="LongBench dataset name (e.g. qasper_e, hotpotqa_e)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path/HuggingFace id")
    parser.add_argument("--residual_length", type=int, default=128,
                        help="OScaR full-precision residual buffer length")
    parser.add_argument("--offline_v_hadamard", action="store_true",
                        help="Absorb V Hadamard into Qwen3 v_proj/o_proj weights and skip runtime V Hadamard")
    parser.add_argument("--num_tokens", type=int, default=None,
                        help="Max new tokens to generate (auto-detected if not set)")
    parser.add_argument("--max_input_len", type=int, default=None,
                        help="Max input token length (overrides MODEL_MAX_LEN)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default="pred",
                        help="Output root directory (default: pred)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path or MODEL_PATHS[args.model]
    model_name = MODEL_NAMES[args.model]
    dtype = MODEL_DTYPES[args.model]
    max_input_len = args.max_input_len or MODEL_MAX_LEN[args.model]
    dataset_base = args.dataset[:-2] if args.dataset.endswith("_e") else args.dataset
    num_tokens = args.num_tokens or DATASET_NUM_TOKENS.get(dataset_base, 64)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           args.output_dir, f"{args.model}_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.dataset}.jsonl")

    print(f"{'='*80}")
    print(f"  LongBench Evaluation: {model_name} / {args.mode} / {args.dataset}")
    print(f"  Output: {out_file}")
    print(f"  model_path={model_path}")
    print(f"  max_input_len={max_input_len}, num_tokens={num_tokens}")
    print(f"  residual_length={args.residual_length}, offline_v_hadamard={args.offline_v_hadamard}")
    print(f"{'='*80}")

    # Load data
    samples = load_dataset(args.dataset)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Total samples: {len(samples)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    print(f"\nLoading {model_name} ({args.mode})...")
    t0 = time.time()
    model = load_model(model_path, dtype, args.mode, args.model, args=args)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Run inference
    print(f"\nRunning inference on {len(samples)} samples...")
    t_start = time.time()

    with open(out_file, "w") as fout:
        for idx, sample in enumerate(samples):
            prompt = truncate_prompt(tokenizer, sample["prompt"], max_input_len)
            input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

            seed_everything(42)
            with torch.no_grad():
                output_ids = model.generate(
                    **input_ids,
                    max_new_tokens=num_tokens,
                    num_beams=1, do_sample=False, temperature=1.0,
                )

            input_len = input_ids["input_ids"].shape[1]
            new_tokens = output_ids[0][input_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)

            result = {
                "pred": pred,
                "answers": sample["answers"],
                "all_classes": sample["all_classes"],
                "length": sample["length"],
                "input_tokens": input_len,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0 or idx == 0:
                fout.flush()
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(samples) - idx - 1) / rate
                print(f"  [{idx+1}/{len(samples)}] {rate:.1f} samples/s, "
                      f"ETA: {eta/60:.1f}min", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone! {len(samples)} samples in {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/s)")
    print(f"Results saved to: {out_file}")

    # Quick score
    try:
        from eval_long_bench import scorer, scorer_e
        predictions, answers, lengths, all_classes = [], [], [], None
        with open(out_file) as f:
            for line in f:
                d = json.loads(line)
                predictions.append(d["pred"])
                answers.append(d["answers"])
                lengths.append(d.get("length", 0))
                all_classes = d["all_classes"]
        metric_key = args.dataset[:-2] if args.dataset.endswith("_e") else args.dataset
        if args.dataset.endswith("_e"):
            score = scorer_e(metric_key, predictions, answers, lengths, all_classes)
        else:
            score = scorer(metric_key, predictions, answers, all_classes)
        print(f"\n  {args.dataset} score: {score}")
    except Exception as e:
        print(f"\n  (Could not compute score: {e})")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
