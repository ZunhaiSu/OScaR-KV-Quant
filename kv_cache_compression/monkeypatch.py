"""
Monkeypatch module for OScaR KV cache quantization.
Supports Qwen3.
"""

import types

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
except ImportError:
    Qwen3Attention = None

from .qwen3_model import qwen3_attention_forward_oscar


def replace_qwen3(args, model, method):
    """Replace Qwen3Attention forward for OScaR quantization."""
    if Qwen3Attention is None:
        raise ImportError("Qwen3Attention not found. Need transformers >= 4.51.")

    if method == 'oscar':
        print('using oscar (Qwen3)')
        for name, module in model.named_modules():
            if isinstance(module, Qwen3Attention):
                module.forward = types.MethodType(qwen3_attention_forward_oscar, module)
                module.k_bits = getattr(args, 'k_bits', 4)
                module.v_bits = getattr(args, 'v_bits', 4)
                module.k_groupsize = getattr(args, 'k_groupsize', 32)
                module.v_groupsize = getattr(args, 'v_groupsize', 32)
                module.k_sym = getattr(args, 'k_sym', False)
                module.v_sym = getattr(args, 'v_sym', False)
                module.k_clip_ratio = getattr(args, 'k_clip_ratio', 1.0)
                module.v_clip_ratio = getattr(args, 'v_clip_ratio', 1.0)
                module.residual_length = getattr(args, 'residual_length', 0)
                module.k_token_rotation = getattr(args, 'k_token_rotation', False)
                module.k_norm_factoring = getattr(args, 'k_norm_factoring', False)
                module.use_hadamard = getattr(args, 'use_hadamard', True)
                module.offline_v_hadamard = getattr(args, 'offline_v_hadamard', False)
