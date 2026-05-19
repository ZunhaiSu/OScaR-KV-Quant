#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = PROJECT_ROOT / "evaluation" / "bench_throughput.py"
SETUP_PY = PROJECT_ROOT / "setup.py"

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "Qwen/Qwen3-8B",
)

VARIANTS: Dict[str, Dict[str, str]] = {
    "fa2": {
        "attn_backend": "flash_attention_2",
        "num_bits": "2",
        "quant_mode": "k-channel",
        "group_size": "32",
        "kv_rotation": "none",
        "kv_norm": "0",
    },
    "oscar_hn": {
        "attn_backend": "oscar",
        "num_bits": "2",
        "quant_mode": "k-channel",
        "group_size": "32",
        "kv_rotation": "hadamard",
        "kv_norm": "1",
    },
}

SMOKE_CASES = [
    {
        "case_name": "fa2_ctx128_bs1",
        "variant": "fa2",
        "batch_size": 1,
        "context_len": 128,
        "decode_len": 1,
        "iteration": 1,
    },
    {
        "case_name": "oscar_hn_ctx128_bs1",
        "variant": "oscar_hn",
        "batch_size": 1,
        "context_len": 128,
        "decode_len": 1,
        "iteration": 1,
    },
    {
        "case_name": "oscar_hn_ctx129_bs1",
        "variant": "oscar_hn",
        "batch_size": 1,
        "context_len": 129,
        "decode_len": 1,
        "iteration": 1,
    },
    {
        "case_name": "oscar_hn_ctx1024_bs1",
        "variant": "oscar_hn",
        "batch_size": 1,
        "context_len": 1024,
        "decode_len": 4,
        "iteration": 1,
    },
]

FULL_SEQ_CONTEXTS = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    49152,
    65536,
    98304,
    131072,
    163840,
    196608,
    229376,
    262144,
    294912,
    327680,
    360448,
]

FULL_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

FIELDNAMES = [
    "case_name",
    "variant",
    "batch_size",
    "context_len",
    "decode_len",
    "iteration",
    "status",
    "runtime_s",
    "peak_mem_mb",
    "prefill_latency_s",
    "decode_latency_total_s",
    "decode_latency_per_token_ms",
    "prefill_tps",
    "decode_tps",
    "log_path",
]

PEAK_MEM_RE = re.compile(r"Peak GPU Memory:\s*([0-9.]+) MB")
PREFILL_LAT_RE = re.compile(r"Avg Prefill Latency:\s*([0-9.]+) s")
DECODE_TOTAL_RE = re.compile(r"Avg Decode Latency \(total\):\s*([0-9.]+) s")
DECODE_PER_TOKEN_RE = re.compile(r"Avg Decode Latency \(per token\):\s*([0-9.]+) s")
PREFILL_TPS_RE = re.compile(r"Prefill Throughput:\s*([0-9.]+) tokens/s")
DECODE_TPS_RE = re.compile(r"Decode Throughput:\s*([0-9.]+) tokens/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke/full Qwen3 OScaR regression suite.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default="qwen3_8b_oscar_suite")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_py_compile", action="store_true")
    return parser.parse_args()


def make_output_dir(mode: str, tag: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    else:
        timestamp = time.strftime("%Y-%m-%d_%H%M%S")
        path = PROJECT_ROOT / "evaluation" / "results" / f"{timestamp}_{tag}_{mode}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("NUMEXPR_MAX_THREADS", "64")
    return env


def parse_metric(regex: re.Pattern[str], text: str) -> float | None:
    match = regex.search(text)
    return float(match.group(1)) if match else None


def bench_command(
    python_bin: str,
    model_path: str,
    device: str,
    dtype: str,
    case: Dict[str, int | str],
) -> List[str]:
    variant_args = VARIANTS[case["variant"]]
    cmd = [
        python_bin,
        str(BENCH_SCRIPT),
        "--model_path",
        model_path,
        "--device",
        device,
        "--dtype",
        dtype,
        "--batch_size",
        str(case["batch_size"]),
        "--context_len",
        str(case["context_len"]),
        "--decode_len",
        str(case["decode_len"]),
        "--iteration",
        str(case["iteration"]),
        "--attn_backend",
        variant_args["attn_backend"],
        "--num_bits",
        variant_args["num_bits"],
        "--quant_mode",
        variant_args["quant_mode"],
        "--group_size",
        variant_args["group_size"],
        "--kv_rotation",
        variant_args["kv_rotation"],
        "--kv_norm",
        variant_args["kv_norm"],
    ]
    return cmd


def run_logged_command(cmd: List[str], env: Dict[str, str], log_path: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(output)
    return proc


def status_from_process(proc: subprocess.CompletedProcess[str], log_text: str) -> str:
    lowered = log_text.lower()
    if proc.returncode == 0:
        return "ok"
    if "out of memory" in lowered or "cuda out of memory" in lowered or "\noom" in lowered:
        return "oom"
    if "illegal memory access" in lowered:
        return "illegal_memory_access"
    return f"error_{proc.returncode}"


def row_from_case(
    case: Dict[str, int | str],
    proc: subprocess.CompletedProcess[str],
    log_path: Path,
    runtime_s: float,
) -> Dict[str, object]:
    log_text = log_path.read_text()
    status = status_from_process(proc, log_text)
    row: Dict[str, object] = {
        "case_name": case["case_name"],
        "variant": case["variant"],
        "batch_size": case["batch_size"],
        "context_len": case["context_len"],
        "decode_len": case["decode_len"],
        "iteration": case["iteration"],
        "status": status,
        "runtime_s": round(runtime_s, 2),
        "peak_mem_mb": None,
        "prefill_latency_s": None,
        "decode_latency_total_s": None,
        "decode_latency_per_token_ms": None,
        "prefill_tps": None,
        "decode_tps": None,
        "log_path": str(log_path),
    }
    if status == "ok":
        row["peak_mem_mb"] = parse_metric(PEAK_MEM_RE, log_text)
        row["prefill_latency_s"] = parse_metric(PREFILL_LAT_RE, log_text)
        row["decode_latency_total_s"] = parse_metric(DECODE_TOTAL_RE, log_text)
        decode_per_token_s = parse_metric(DECODE_PER_TOKEN_RE, log_text)
        row["decode_latency_per_token_ms"] = None if decode_per_token_s is None else decode_per_token_s * 1000.0
        row["prefill_tps"] = parse_metric(PREFILL_TPS_RE, log_text)
        row["decode_tps"] = parse_metric(DECODE_TPS_RE, log_text)
    return row


def write_csv(rows: Iterable[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_build(args: argparse.Namespace, output_dir: Path, env: Dict[str, str]) -> None:
    build_log = output_dir / "build.log"
    cmd = [args.python_bin, str(SETUP_PY), "build_ext", "--inplace"]
    proc = run_logged_command(cmd, env, build_log)
    if proc.returncode != 0:
        raise RuntimeError(f"Build failed. See {build_log}")


def run_py_compile(args: argparse.Namespace, output_dir: Path, env: Dict[str, str]) -> None:
    compile_log = output_dir / "py_compile.log"
    cmd = [
        args.python_bin,
        "-m",
        "py_compile",
        "evaluation/qwen3.py",
        "evaluation/bench_throughput.py",
        "eval_longbench.py",
        "evaluation/scripts/run_qwen3_suite.py",
        "oscar/oscar_interface.py",
    ]
    proc = run_logged_command(cmd, env, compile_log)
    if proc.returncode != 0:
        raise RuntimeError(f"py_compile failed. See {compile_log}")


def run_smoke(args: argparse.Namespace, output_dir: Path, env: Dict[str, str]) -> Path:
    logs_dir = output_dir / "logs_smoke"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for case in SMOKE_CASES:
        log_path = logs_dir / f"{case['case_name']}.log"
        cmd = bench_command(args.python_bin, args.model_path, args.device, args.dtype, case)
        start = time.time()
        proc = run_logged_command(cmd, env, log_path)
        rows.append(row_from_case(case, proc, log_path, time.time() - start))
    csv_path = output_dir / "smoke_results.csv"
    write_csv(rows, csv_path)
    return csv_path


def full_seq_cases() -> List[Dict[str, int | str]]:
    cases: List[Dict[str, int | str]] = []
    for context_len in FULL_SEQ_CONTEXTS:
        for variant in ("fa2", "oscar_hn"):
            cases.append(
                {
                    "case_name": f"{variant}_ctx{context_len}_bs1",
                    "variant": variant,
                    "batch_size": 1,
                    "context_len": context_len,
                    "decode_len": 8,
                    "iteration": 1,
                }
            )
    return cases


def full_batch_cases() -> List[Dict[str, int | str]]:
    cases: List[Dict[str, int | str]] = []
    for batch_size in FULL_BATCH_SIZES:
        for variant in ("fa2", "oscar_hn"):
            cases.append(
                {
                    "case_name": f"{variant}_ctx4096_bs{batch_size}",
                    "variant": variant,
                    "batch_size": batch_size,
                    "context_len": 4096,
                    "decode_len": 16,
                    "iteration": 1,
                }
            )
    return cases


def run_case_matrix(
    args: argparse.Namespace,
    env: Dict[str, str],
    cases: List[Dict[str, int | str]],
    logs_dir: Path,
    stop_on_first_failure_per_variant: bool,
) -> List[Dict[str, object]]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    variant_stopped = {variant: False for variant in VARIANTS}
    for case in cases:
        variant = str(case["variant"])
        if stop_on_first_failure_per_variant and variant_stopped[variant]:
            continue
        log_path = logs_dir / f"{case['case_name']}.log"
        cmd = bench_command(args.python_bin, args.model_path, args.device, args.dtype, case)
        start = time.time()
        proc = run_logged_command(cmd, env, log_path)
        row = row_from_case(case, proc, log_path, time.time() - start)
        rows.append(row)
        if stop_on_first_failure_per_variant and row["status"] != "ok":
            variant_stopped[variant] = True
        time.sleep(2)
    return rows


def write_full_summary(output_dir: Path, batch_rows: List[Dict[str, object]]) -> None:
    by_variant_batch = {
        variant: {int(row["batch_size"]): row for row in batch_rows if row["variant"] == variant}
        for variant in VARIANTS
    }
    compare_rows: List[Dict[str, object]] = []
    savings_rows: List[Dict[str, object]] = []
    for batch_size in FULL_BATCH_SIZES:
        fa2 = by_variant_batch["fa2"].get(batch_size)
        oscar = by_variant_batch["oscar_hn"].get(batch_size)
        compare_row: Dict[str, object] = {
            "batch_size": batch_size,
            "context_len": 4096,
            "fa2_status": fa2["status"] if fa2 else "missing",
            "fa2_peak_mem_mb": fa2["peak_mem_mb"] if fa2 else None,
            "oscar_hn_status": oscar["status"] if oscar else "missing",
            "oscar_hn_peak_mem_mb": oscar["peak_mem_mb"] if oscar else None,
            "saved_mem_mb": None,
            "saved_mem_ratio": None,
        }
        if (
            fa2
            and oscar
            and fa2["status"] == "ok"
            and oscar["status"] == "ok"
            and fa2["peak_mem_mb"] is not None
            and oscar["peak_mem_mb"] is not None
        ):
            saved_mem_mb = float(fa2["peak_mem_mb"]) - float(oscar["peak_mem_mb"])
            saved_mem_ratio = saved_mem_mb / float(fa2["peak_mem_mb"])
            compare_row["saved_mem_mb"] = saved_mem_mb
            compare_row["saved_mem_ratio"] = saved_mem_ratio
            savings_rows.append(
                {
                    "batch_size": batch_size,
                    "context_len": 4096,
                    "fa2_peak_mem_mb": fa2["peak_mem_mb"],
                    "oscar_hn_peak_mem_mb": oscar["peak_mem_mb"],
                    "saved_mem_mb": saved_mem_mb,
                    "saved_mem_ratio": saved_mem_ratio,
                    "fa2_decode_tps": fa2["decode_tps"],
                    "oscar_hn_decode_tps": oscar["decode_tps"],
                }
            )
        compare_rows.append(compare_row)

    compare_csv = output_dir / "batch_memory_compare.csv"
    with compare_csv.open("w", newline="") as f:
        fieldnames = [
            "batch_size",
            "context_len",
            "fa2_status",
            "fa2_peak_mem_mb",
            "oscar_hn_status",
            "oscar_hn_peak_mem_mb",
            "saved_mem_mb",
            "saved_mem_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compare_rows)

    savings_csv = output_dir / "batch_memory_savings.csv"
    with savings_csv.open("w", newline="") as f:
        fieldnames = [
            "batch_size",
            "context_len",
            "fa2_peak_mem_mb",
            "oscar_hn_peak_mem_mb",
            "saved_mem_mb",
            "saved_mem_ratio",
            "fa2_decode_tps",
            "oscar_hn_decode_tps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(savings_rows)


def run_full(args: argparse.Namespace, output_dir: Path, env: Dict[str, str]) -> None:
    seq_rows = run_case_matrix(
        args,
        env,
        full_seq_cases(),
        output_dir / "logs_seq",
        stop_on_first_failure_per_variant=True,
    )
    batch_rows = run_case_matrix(
        args,
        env,
        full_batch_cases(),
        output_dir / "logs_batch",
        stop_on_first_failure_per_variant=True,
    )
    write_csv(seq_rows, output_dir / "seq_len_decode_latency.csv")
    write_csv(batch_rows, output_dir / "batch_sweep_throughput.csv")
    write_full_summary(output_dir, batch_rows)


def print_smoke_summary(csv_path: Path) -> None:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"[smoke] results: {csv_path}")
    for row in rows:
        status = row["status"]
        decode_tps = row["decode_tps"] or "-"
        print(
            f"  {row['case_name']}: status={status}, "
            f"decode_tps={decode_tps}, peak_mem_mb={row['peak_mem_mb'] or '-'}"
        )


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args.mode, args.tag, args.output_dir)
    env = default_env()

    if not args.skip_build:
        run_build(args, output_dir, env)
    if not args.skip_py_compile:
        run_py_compile(args, output_dir, env)

    if args.mode == "smoke":
        csv_path = run_smoke(args, output_dir, env)
        print_smoke_summary(csv_path)
        return

    run_full(args, output_dir, env)
    print(f"[full] results written to {output_dir}")


if __name__ == "__main__":
    main()
