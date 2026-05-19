<h1 align="center">
  <img src="oscar.png" width="150"><br>
  OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond
</h1>

<div align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</div>

## 🔥 Latest News

- **[Upcoming]** 🔧 vLLM & SGLang backend integration — under active development, official support will be announced in future releases.

- **[2026-05-20]** 🎉 Our paper *"OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond"* is now available on arXiv! [[Link](https://arxiv.org/abs/XXXX.XXXXX)]

- **[2026-05-19]** 🚀 Codebase and evaluation suite publicly released.

## 📖 Overview

<div align="center">
  <img src="overview.png" width="90%">
</div>

The rapid advancement toward **long-context reasoning** and **multi-modal intelligence** has made KV cache memory footprint a dominant bottleneck. We revisit the inherent limitations of the established **per-channel quantization paradigm** and identify **Token Norm Imbalance (TNI)** as the primary bottleneck to quantization fidelity.

Rather than relying on intricate pipelines, we follow the principle of **Occam's Razor**. We propose **OScaR (Omni-Scaled Canalized Rotation)** , an accurate and lightweight KV cache compression framework for X-LLMs (text-only, multi-modal, and omni-modal LLMs). 

### TNI in X-LLMs

<div align="center">
  <table>
    <tr>
      <td align="center"><strong>Text-Only LLMs</strong><br><img src="LLM-TNI.png" width="95%"><br><em>Low-norm outlier tokens (Attention Sinks)</em></td>
      <td align="center"><strong>Multi-Modal LLMs</strong><br><img src="MLLM-TNI.png" width="95%"><br><em>Large-norm outliers</em></td>
    </tr>
  </table>
</div>

> TNI is pervasive across text-only, multi-modal, and omni-modal LLMs. In text-only models, it manifests as low-norm outlier tokens, commonly referred to as Attention Sinks. In multi-modal settings, TNI exhibits more diverse forms, including large-norm outliers, broader norm variations, and significant inter-modality disparities. Additional visualizations and detailed experimental configurations are provided in the paper.


## ✨ Key Features

- 🔍 **Unveils TNI as the structural bottleneck** of per-channel quantization through both empirical and theoretical analysis.

- 🪒 **Streamlined framework** guided by Occam's Razor — requiring only two essential operations, **Canalized Rotation** and **Omni-Token Scaling**, with no training or calibration overhead.

- 📈 **Redefines the Pareto front** for X-LLMs, delivering near-lossless INT2 quantization across diverse benchmarks while maintaining low computational complexity.

- ⚡ **Optimized CUDA kernels** built upon BitDecoding and HadaCore with Tensor Core acceleration, achieving 3.0× decoding speedup, 5.3× memory reduction, and 4.1× throughput increase.

## 📊 Main Results

### LongBench-E

OScaR achieves the highest average accuracy among all 2-bit methods on LongBench-E, outperforming KIVI, OTT, QuaRot, and TurboQuant+ across both Llama-3.1-8B and Qwen3-8B.

| Method | Llama-3.1-8B | Qwen3-8B |
|:-------|:------------:|:--------:|
| 16-bit Baseline | 41.70 | 49.56 |
| QuaRot (INT2) | 37.94 | 40.13 |
| RotateKV (INT2) | 37.98 | 42.95 |
| KIVI (INT2) | 39.84 | 47.95 |
| OTT (INT2) | 40.74 | 48.21 |
| TurboQuant+ (2.5-bit) | 40.03 | 47.56 |
| **OScaR (INT2)** | **41.75** | **48.74** |

### OCRBench

On OCRBench, OScaR consistently outperforms other 2-bit methods across LLaVA-v1.6-vicuna-7B, Qwen3-VL-8B, and Qwen3-VL-4B.

| Method | LLaVA-v1.6-7B | Qwen3-VL-8B | Qwen3-VL-4B |
|:-------|:-------------:|:-----------:|:-----------:|
| 16-bit Baseline | 536 | 858 | 852 |
| QuaRot (INT2) | 481 | 722 | 773 |
| RotateKV (INT2) | 473 | 754 | 638 |
| KIVI (INT2) | 488 | 851 | 813 |
| OTT (INT2) | 513 | 850 | 831 |
| TurboQuant+ (2.5-bit) | 501 | 847 | 828 |
| **OScaR (INT2)** | **519** | **856** | **838** |

### MMAU-Pro

On the challenging MMAU-Pro benchmark for omni-modal understanding, OScaR surpasses both the 16-bit baseline and all quantized methods across open-ended QA, Good Rate, and Audio Instruction Following (AIF).

| Method (Qwen3-Omni-30B-A3B) | Open-ended | Good Rate | AIF |
|:---------------------------|:----------:|:---------:|:---:|
| 16-bit Baseline | 66.2 | 27.8 | 87.4 |
| KIVI (INT2) | 65.8 | 27.0 | 78.2 |
| OTT (INT2) | 65.8 | 26.9 | 83.9 |
| TurboQuant+ (2.5-bit) | 66.6 | 27.0 | 79.3 |
| **OScaR (INT2)** | **67.4** | **29.8** | **88.5** |

> **Note:** Detailed experimental setups and TurboQuant+ implementation details are available in the original paper.



## Setup
```bash
git clone --recursive <your-oscar-repo-url> OScaR
cd OScaR

uv venv --python 3.10 .venv-local
source .venv-local/bin/activate

git submodule update --init --recursive

uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
uv pip install -r requirements.txt
uv pip install --no-build-isolation flash-attn

python setup.py build_ext --inplace
```

Validated stack on H20:

- Python `3.10.17`
- PyTorch `2.6.0+cu124`
- `flash-attn 2.8.3`
- `transformers 4.57.6`

## Run
Set `MODEL_PATH` to your local Qwen3 checkpoint path before running the suite.

In this branch, OScaR refers to the Qwen3 `2-bit + Hadamard + norm` path. The
public attention backend name is `oscar`.

### Smoke Test
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/run_qwen3_suite.py \
  --mode smoke \
  --python_bin "$(which python)" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --dtype bfloat16
```

### Full Benchmark Sweep
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/run_qwen3_suite.py \
  --mode full \
  --python_bin "$(which python)" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --dtype bfloat16
```

If the extension has already been built and Python files have already been checked, you can skip those steps:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/scripts/run_qwen3_suite.py \
  --mode full \
  --skip_build \
  --skip_py_compile \
  --python_bin "$(which python)" \
  --model_path "${MODEL_PATH}" \
  --device cuda:0 \
  --dtype bfloat16
```

### OScaR Accuracy Quick Start
The quickest end-to-end accuracy check for OScaR is Qasper-E:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export PY=.venv-local/bin/python

CUDA_VISIBLE_DEVICES=0 $PY eval_longbench.py \
  --model_path "$MODEL_PATH" \
  --datasets qasper_e \
  --max_input_len 32768 \
  --dtype bfloat16 \
  --device cuda:0 \
  --residual_evict_size 256 \
  --offline_v_hadamard \
  --output_dir pred_e/qwen3_8b_hn2bit_offline_v_r128_ev256_qasper \
  --log_every 1 \
  --resume

$PY eval_long_bench.py \
  --path pred_e/qwen3_8b_hn2bit_offline_v_r128_ev256_qasper \
  --e
```

This path is OScaR: 2-bit KV cache quantization with K Hadamard + norm and
offline V Hadamard, dispatched through the CUDA kernels in `oscar_cuda`. It assumes
`longbench_data/data/qasper_e.jsonl` and
`longbench_config/{dataset2prompt.json,dataset2maxlen.json}` are available.

### OScaR Single Example
```bash
MODEL_PATH="${MODEL_PATH}" \
DTYPE=bfloat16 \
NUM_BITS=2 \
QUANT_MODE=k-channel \
GROUP_SIZE=32 \
KV_ROTATION=hadamard \
KV_NORM=1 \
ATTN_BACKEND=oscar \
bash evaluation/scripts/example.sh
```

OScaR is a high-performance, GPU-optimized system
designed to accelerate long-context LLMs decoding with a low-bit KV
cache. Achieve **3-9x speedup** than Flash Attention v2.
![overview](imgs/overview.png)
![scheme](imgs/scheme.png)

## Benchmark
* Kernel Performance in RTX4090
![overview](imgs/4090.png)
* Kernel Performance in A100
![overview](imgs/a100.png)

## Citation
If you find OScaR useful or want to use in your projects, please kindly cite our paper:
```
@misc{du2025bitdecodingunlockingtensorcores,
      title={BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache}, 
      author={Dayou Du and Shijie Cao and Jianyi Cheng and Ting Cao and Mao Yang},
      year={2025},
      eprint={2503.18773},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2503.18773}, 
}
```

## Acknowledgement
OScaR is inspired by many open-source libraries, including (but not limited to) [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main), [flute](https://github.com/HanGuo97/flute), [Atom](https://github.com/efeslab/Atom), [omniserve](https://github.com/mit-han-lab/omniserve), [KIVI](https://github.com/jy-yuan/KIVI).
