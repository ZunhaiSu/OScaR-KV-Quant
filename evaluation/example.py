import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import torch
import random
import argparse
import re
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oscar import DynamicCache, StaticCache, Cache
import transformers.cache_utils
transformers.cache_utils.DynamicCache = DynamicCache
transformers.cache_utils.StaticCache = StaticCache
transformers.cache_utils.Cache = Cache

from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _hadamard_transform_pytorch(x, scale=1.0):
    n = x.shape[-1]
    assert _is_pow2(n), f"Hadamard dimension must be a power of 2, got {n}"
    h = 1
    while h < n:
        x_view = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        even = x_view[..., 0, :]
        odd = x_view[..., 1, :]
        x = torch.stack((even + odd, even - odd), dim=-2).view(*x.shape[:-1], n)
        h <<= 1
    return x * scale


def _hadamard_matrix(size, device):
    eye = torch.eye(size, device=device, dtype=torch.float32)
    return _hadamard_transform_pytorch(eye, scale=size ** -0.5)


@torch.no_grad()
def apply_exact_had_to_linear(module, had_dim, output=False, R=None):
    assert isinstance(module, nn.Linear)
    assert _is_pow2(had_dim), f"Hadamard dimension must be a power of 2, got {had_dim}"

    weight = module.weight.data
    dtype_orig = weight.dtype
    weight_f = weight.float()
    had = _hadamard_matrix(had_dim, weight_f.device) if R is None else R.to(weight_f.device, torch.float32)

    if output:
        if module.out_features % had_dim != 0:
            raise ValueError(f"out_features={module.out_features} is not divisible by had_dim={had_dim}")
        rotated = torch.matmul(
            had,
            weight_f.reshape(module.out_features // had_dim, had_dim, module.in_features),
        )
        module.weight.data = rotated.reshape_as(weight_f).to(dtype_orig)
        if module.bias is not None:
            bias_dtype = module.bias.data.dtype
            bias = module.bias.data.float().reshape(module.out_features // had_dim, had_dim)
            module.bias.data = torch.matmul(bias, had).reshape_as(module.bias.data).to(bias_dtype)
    else:
        if module.in_features % had_dim != 0:
            raise ValueError(f"in_features={module.in_features} is not divisible by had_dim={had_dim}")
        rotated = torch.matmul(
            weight_f.reshape(module.out_features, module.in_features // had_dim, had_dim),
            had,
        )
        module.weight.data = rotated.reshape_as(weight_f).to(dtype_orig)


@torch.no_grad()
def apply_offline_v_hadamard(model):
    config = model.config
    had_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    if not _is_pow2(had_dim):
        raise ValueError(f"Qwen3 head_dim must be a power of two for offline V Hadamard, got {had_dim}")
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("Could not find model.model.layers for offline V Hadamard rotation")
    for layer in layers:
        apply_exact_had_to_linear(layer.self_attn.v_proj, had_dim=had_dim, output=True)
        apply_exact_had_to_linear(layer.self_attn.o_proj, had_dim=had_dim, output=False)
    return had_dim


def resolve_model_components(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3":
        from qwen3 import Qwen3ForCausalLM
        return config, Qwen3ForCausalLM
    raise ValueError(f"Unsupported model_type: {model_type}. This repo only supports Qwen3.")


def resolve_torch_dtype(config, dtype_name):
    if dtype_name == "auto":
        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str):
            return getattr(torch, config_dtype)
        if isinstance(config_dtype, torch.dtype):
            return config_dtype
        return torch.float16
    return getattr(torch, dtype_name)


def trim_gsm8k_answer(text):
    match = re.search(r"####\s*-?[\d,]+(?:\.\d+)?", text)
    if match is None:
        return text
    return text[: match.end()].rstrip()


def extract_gsm8k_number(text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match is not None:
        return match.group(1).replace(",", "")
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a Qwen3 OScaR example")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--max_length', type=int, default=131072, help='Maximum length of the input sequence')
    parser.add_argument('--dtype', type=str, default='auto', help='Torch dtype: auto, float16, bfloat16, float32')
    parser.add_argument('--num_bits', type=int, default=4, help='Number of bits for quantization')
    parser.add_argument('--quant_mode', type=str, default='k-channel', help='Quantization mode')
    parser.add_argument('--group_size', type=int, default=None, help='Group size for quantization')
    parser.add_argument('--kv_rotation', type=str, default='none', help='KV rotation mode, e.g. none or hadamard')
    parser.add_argument('--kv_norm', type=str, default='0', help='KV norm mode, e.g. 0 or 1')
    parser.add_argument('--attn_backend', type=str, default='flash_attention_2', help='Attention implementation, e.g. flash_attention_2, flash_decoding, oscar')
    parser.add_argument('--device', type=str, default='cuda:0', help='Model/device placement')
    parser.add_argument('--offline_v_hadamard', action='store_true', help='Absorb V Hadamard into Qwen3 v_proj/o_proj weights')
    parser.add_argument('--residual_evict_size', type=int, default=None, help='Override Qwen3 OScaR residual eviction size')
    parser.add_argument('--max_new_tokens', type=int, default=125, help='Maximum generated tokens')
    args = parser.parse_args()

    # For reproducibility 
    random.seed(0)
    torch.manual_seed(0)

    if args.group_size is None:
        args.group_size = 32 if args.num_bits == 2 else 128

    config, model_cls = resolve_model_components(args.model_path)
    dtype = resolve_torch_dtype(config, args.dtype)

    config._attn_implementation = "flash_attention_2"
    config.attn_backend = args.attn_backend
    config.num_bits = args.num_bits
    config.quant_mode = args.quant_mode
    config.group_size = args.group_size
    config.kv_rotation = args.kv_rotation
    config.kv_norm = args.kv_norm
    config.residual_block_size = 128
    if args.residual_evict_size is not None:
        config.residual_evict_size = args.residual_evict_size

    model = model_cls.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map={"": args.device}
    )
    if args.offline_v_hadamard:
        had_dim = apply_offline_v_hadamard(model)
        print(f"Applied offline V Hadamard to Qwen3 v_proj/o_proj weights with had_dim={had_dim}")

    enc = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='left',  # Add this line
        truncation_side='left',
        pad_token='</s>'      # Add this line
    )

    dataset = load_dataset('gsm8k', 'main')

    prompt = ''
    for i in range(15):
        prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
    prompt += "Arnel had ten boxes of pencils with the same number of pencils in each box. He kept ten pencils and shared the remaining pencils equally with his five friends. If his friends got eight pencils each, how many pencils are in each box?"

    inputs = enc(
        prompt,
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_attention_mask=True
    ).to(args.device)
    print(f"# prompt_tokens: {inputs.input_ids.shape[1]}")

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=enc.pad_token_id,
        max_new_tokens=args.max_new_tokens
    )
    config_str = f"# prompt tokens: {inputs.input_ids.shape[1]}"

    # print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nOutput:")
    # print("\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nOutput:")
    generated = enc.decode(output[0].tolist()[inputs.input_ids.shape[1]:], skip_special_tokens=True)
    trimmed = trim_gsm8k_answer(generated)
    print(trimmed)
    pred_answer = extract_gsm8k_number(trimmed)
    expected_answer = "5"
    print(f"# smoke_expected_answer: {expected_answer}")
    print(f"# smoke_pred_answer: {pred_answer}")
    print(f"# smoke_exact_match: {pred_answer == expected_answer}")

if __name__ == "__main__":
    main()
