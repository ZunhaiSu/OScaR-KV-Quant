"""
OScaR KV Cache Quantization — Qwen3 attention forward.
Compatible with transformers >= 4.51 (Qwen3Attention API).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import logging

from .quarot_utils import init_quarot

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _get_kv_cache(past_key_value, layer_idx):
    if hasattr(past_key_value, 'key_cache'):
        return past_key_value.key_cache[layer_idx], past_key_value.value_cache[layer_idx]
    return past_key_value.layers[layer_idx].keys, past_key_value.layers[layer_idx].values


def _set_kv_cache(past_key_value, layer_idx, key_states, value_states):
    if hasattr(past_key_value, 'key_cache'):
        past_key_value.key_cache[layer_idx] = key_states
        past_key_value.value_cache[layer_idx] = value_states
    else:
        past_key_value.layers[layer_idx].keys = key_states
        past_key_value.layers[layer_idx].values = value_states


def qwen3_attention_forward_oscar(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    OScaR KV cache quantization for Qwen3 backbone (eager attention).

    Flow:
    1. QKV projection + QK normalization
    2. RoPE
    3. Hadamard rotation on Q, K (optional)
    4. Fake-quantize K, V with norm factoring
    5. Standard attention computation
    """
    past_key_value = past_key_value or kwargs.get("past_key_values", None)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    # Initialize quantizer on first prefill
    if q_len > 1:
        if not hasattr(self, 'quarot_quantizer'):
            init_quarot(self,
                        k_bits=getattr(self, 'k_bits', 4),
                        v_bits=getattr(self, 'v_bits', 4),
                        k_groupsize=getattr(self, 'k_groupsize', 32),
                        v_groupsize=getattr(self, 'v_groupsize', 32),
                        k_sym=getattr(self, 'k_sym', False),
                        v_sym=getattr(self, 'v_sym', False),
                        k_clip_ratio=getattr(self, 'k_clip_ratio', 1.0),
                        v_clip_ratio=getattr(self, 'v_clip_ratio', 1.0),
                        residual_length=getattr(self, 'residual_length', 0),
                        k_token_rotation=getattr(self, 'k_token_rotation', False),
                        k_norm_factoring=getattr(self, 'k_norm_factoring', False),
                        use_hadamard=getattr(self, 'use_hadamard', True),
                        offline_v_hadamard=getattr(self, 'offline_v_hadamard', False))
        else:
            self.quarot_quantizer.committed_k_len = 0
            self.quarot_quantizer.committed_v_len = 0

    # Qwen3: project + QK norm + reshape
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # OScaR: Hadamard rotation on Q, K
    if hasattr(self, 'quarot_quantizer'):
        query_states, key_states, value_states = self.quarot_quantizer.process_kv(
            query_states, key_states, value_states)

    # Update KV cache
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # GQA repeat
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if q_len > 1:
        # Prefill: flash attention
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states_fa = query_states.transpose(1, 2)
        key_states_fa = key_states.transpose(1, 2)
        value_states_fa = value_states.transpose(1, 2)

        sliding_window = getattr(self, 'sliding_window', None)

        attn_output = _flash_attention_forward(
            query_states_fa, key_states_fa, value_states_fa, None, q_len,
            dropout=0.0 if not self.training else self.attention_dropout,
            sliding_window=sliding_window,
            is_causal=True,
            use_top_left_mask=getattr(self, '_flash_attn_uses_top_left_mask', False),
        )
        attn_weights = None
    else:
        # Decode: eager attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    # Quantize KV cache after attention
    if hasattr(self, 'quarot_quantizer') and past_key_value is not None:
        key_states_cache, value_states_cache = _get_kv_cache(past_key_value, self.layer_idx)
        if q_len > 1:
            key_states_cache, value_states_cache = self.quarot_quantizer.quantize_prefill(
                key_states_cache, value_states_cache)
        else:
            key_states_cache, value_states_cache = self.quarot_quantizer.quantize_kv_cache(
                key_states_cache, value_states_cache)
        _set_kv_cache(past_key_value, self.layer_idx, key_states_cache, value_states_cache)

    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights
