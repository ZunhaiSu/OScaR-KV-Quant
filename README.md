# OScaR
[![arXiv](https://img.shields.io/badge/arXiv-2410.13276-b31b1b.svg)](https://arxiv.org/abs/2503.18773)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
