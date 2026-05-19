"""
QuaRot KV Cache Quantization Utilities.
Adapted from QuaRot (https://github.com/spcl/QuaRot) fake_quant mode.

This module provides Hadamard rotation + low-bit quantization for KV cache compression.
Supports 2/4/8 bit quantization with per-channel (K) and per-token (V) modes.

Key idea:
  - Apply Hadamard rotation to Q,K after RoPE to remove outliers (makes distribution more uniform)
  - Quantize K per-channel and V per-token to low bits (fake quantization)
  - The Hadamard rotation preserves QK^T since Hadamard matrices are orthogonal

Quantization granularity (following KIVI):
  - K cache: per-channel quantization (statistics along seq_len dim, one scale per head_dim channel)
  - V cache: per-token quantization (statistics along head_dim dim, one scale per token)

Blockwise quantization (following KIVI's group_size):
  - K cache + blockwise: seq_len split into groups of group_size tokens, each group has its own
    per-channel scale. Finer-grained than global per-channel.
  - V cache + blockwise: head_dim split into groups of group_size channels, each group has its own
    per-token scale. Finer-grained than global per-token.
  - Activated by setting k_groupsize > 0 (for K) and v_groupsize > 0 (for V).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic quantization functions (from QuaRot/fake_quant/quant_utils.py)
# ============================================================

def get_minq_maxq(bits, sym):
    """Get min/max quantization range for given bit width."""
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)
    return minq, maxq


def sym_quant(x, scale, maxq):
    """Symmetric quantization."""
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale


def sym_dequant(q, scale):
    """Symmetric dequantization."""
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    """Symmetric quantize then dequantize (fake quantization)."""
    return sym_dequant(*sym_quant(x, scale, maxq))


def asym_quant(x, scale, zero, maxq):
    """Asymmetric quantization."""
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    """Asymmetric dequantization."""
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    """Asymmetric quantize then dequantize (fake quantization)."""
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


# ============================================================
# KV Quantizer (supports per-token and per-channel modes)
# ============================================================

class KVQuantizer:
    """
    Quantizer for KV cache states.
    Supports per-token, per-channel, and blockwise quantization modes.

    Modes (determined by per_channel and groupsize):
      - per_channel=True,  groupsize=-1: Per-channel (K cache default).
        Stats along entire seq_len dim. One scale per head_dim channel.
      - per_channel=True,  groupsize>0:  Per-channel + blockwise (K cache, KIVI-style).
        Seq_len split into groups of groupsize tokens. Each group has its own per-channel scale.
        Finer-grained than global per-channel.
      - per_channel=False, groupsize=-1: Per-token (V cache default).
        Stats along entire head_dim dim. One scale per token.
      - per_channel=False, groupsize>0:  Per-token + blockwise (V cache, KIVI-style).
        Head_dim split into groups of groupsize channels. Each group has its own per-token scale.
        Finer-grained than global per-token.
    """

    def __init__(self, bits=4, groupsize=-1, sym=True, clip_ratio=1.0, per_channel=False):
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.per_channel = per_channel
        _, self.maxq = get_minq_maxq(bits, sym)

    def quantize_dequantize(self, x):
        """
        Quantize and dequantize tensor x (fake quantization).

        Args:
            x: tensor of shape (bsz, num_heads, seq_len, head_dim)

        Returns:
            Fake-quantized tensor of same shape.

        Modes (determined by per_channel and groupsize):
          - per_channel=True,  groupsize=-1: per-channel (stats along entire seq_len)
          - per_channel=True,  groupsize>0:  per-channel + blockwise (group seq_len into blocks)
          - per_channel=False, groupsize=-1: per-token (stats along entire head_dim)
          - per_channel=False, groupsize>0:  per-token + blockwise (group head_dim into blocks)
        """
        if self.bits >= 16:
            return x

        x_dtype = x.dtype
        dev = x.device
        maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.per_channel and self.groupsize > 0:
            # ============================================================
            # Per-channel + blockwise: KIVI-style K cache quantization
            #
            # Following KIVI exactly:
            #   1. Transpose: (B, nh, T, D) → (B, nh, D, T)
            #   2. Group along last dim (T): (B*nh*D, num_groups, group_size)
            #   3. min/max along group_size dim (last dim)
            #   4. scale shape: (B, nh, D, num_groups)
            #   5. Quantize/dequantize, then transpose back
            # ============================================================
            bsz, num_heads, seq_len, head_dim = x.shape
            gs = self.groupsize

            # Step 1: Transpose to (B, nh, D, T) — same as KIVI
            x_trans = x.transpose(2, 3).contiguous()  # (B, nh, D, T)

            # Pad T (seq_len) if not divisible by group_size
            T = x_trans.shape[3]
            if T % gs != 0:
                pad_len = gs - (T % gs)
                x_trans = F.pad(x_trans, (0, pad_len), value=0)
            else:
                pad_len = 0

            T_padded = x_trans.shape[3]
            num_groups = T_padded // gs

            # Step 2: Reshape to (B*nh*D, num_groups, group_size) — same as KIVI
            reshaped = x_trans.reshape(bsz * num_heads * head_dim, num_groups, gs)

            # Step 3: min/max along last dim (group_size) — same as KIVI
            xmax = torch.amax(reshaped, dim=-1, keepdim=True) * self.clip_ratio
            xmin = torch.amin(reshaped, dim=-1, keepdim=True) * self.clip_ratio

            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                scale = xmax / maxq
                scale[tmp] = 1
                zero = torch.zeros_like(scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                scale = (xmax - xmin) / maxq
                zero = torch.round(-xmin / scale)

            # Step 4: Expand scale/zero and quantize
            # scale/zero shape: (B*nh*D, num_groups, 1) → (B*nh*D, num_groups, gs)
            scale = scale.expand_as(reshaped)
            zero = zero.expand_as(reshaped)

            if self.sym:
                result = sym_quant_dequant(reshaped, scale, maxq)
            else:
                result = asym_quant_dequant(reshaped, scale, zero, maxq)

            # Step 5: Reshape back to (B, nh, D, T_padded), remove padding, transpose back
            result = result.reshape(bsz, num_heads, head_dim, T_padded)

            if pad_len > 0:
                result = result[:, :, :, :seq_len]

            # Transpose back: (B, nh, D, T) → (B, nh, T, D)
            result = result.transpose(2, 3).contiguous()

            return result.to(x_dtype)

        elif self.per_channel:
            # ============================================================
            # Per-channel quantization (no blockwise): statistics along entire seq_len
            # x shape: (bsz, num_heads, seq_len, head_dim)
            # Each channel (head_dim dimension) gets its own scale/zero,
            # computed across all tokens in seq_len.
            # ============================================================
            bsz, num_heads, seq_len, head_dim = x.shape

            xmax = torch.amax(x, dim=2, keepdim=True) * self.clip_ratio  # (bsz, num_heads, 1, head_dim)
            xmin = torch.amin(x, dim=2, keepdim=True) * self.clip_ratio  # (bsz, num_heads, 1, head_dim)

            tmp_zero = torch.zeros(bsz, num_heads, 1, head_dim, device=dev)
            xmin = torch.minimum(xmin, tmp_zero)
            xmax = torch.maximum(xmax, tmp_zero)

            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                scale = xmax / maxq  # (bsz, num_heads, 1, head_dim)
                scale[tmp] = 1
                zero = torch.zeros_like(scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                scale = (xmax - xmin) / maxq  # (bsz, num_heads, 1, head_dim)
                zero = torch.round(-xmin / scale)

            # scale/zero broadcast: (bsz, num_heads, 1, head_dim) -> (bsz, num_heads, seq_len, head_dim)
            scale = scale.expand_as(x)
            zero = zero.expand_as(x)

        elif self.groupsize > 0:
            # ============================================================
            # Per-token + blockwise: group head_dim into blocks of groupsize
            # (following KIVI's V cache quantization pattern)
            #
            # x shape: (B, nh, T, D)
            # → reshape to (B, nh, T, num_groups, group_size)
            # → min/max along dim=-1 (within each group of channels)
            # → scale shape: (B, nh, T, num_groups, 1)
            # ============================================================
            bsz, num_heads, seq_len, head_dim = x.shape
            gs = self.groupsize
            assert head_dim % gs == 0, \
                f"head_dim {head_dim} must be divisible by groupsize {gs}"
            num_groups = head_dim // gs

            # Reshape: (B, nh, T, num_groups, group_size)
            reshaped = x.reshape(bsz, num_heads, seq_len, num_groups, gs)

            xmax = torch.amax(reshaped, dim=-1, keepdim=True) * self.clip_ratio
            xmin = torch.amin(reshaped, dim=-1, keepdim=True) * self.clip_ratio

            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                scale = xmax / maxq
                scale[tmp] = 1
                zero = torch.zeros_like(scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                scale = (xmax - xmin) / maxq
                zero = torch.round(-xmin / scale)

            # Expand: (B, nh, T, num_groups, 1) → (B, nh, T, num_groups, gs)
            scale = scale.expand_as(reshaped).reshape(init_shape)
            zero = zero.expand_as(reshaped).reshape(init_shape)

        else:
            # ============================================================
            # Per-token quantization (no blockwise): statistics along entire head_dim
            # Each token gets its own scale/zero.
            # ============================================================
            reshaped_x = x.reshape((-1, x.shape[-1]))

            tmp = torch.zeros(reshaped_x.shape[0], device=dev)
            xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
            xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio

            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                scale = (xmax / maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
                scale[tmp] = 1
                scale = scale.reshape(init_shape)
                zero = torch.zeros_like(scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = +1
                scale = (xmax - xmin) / maxq
                zero = torch.round(-xmin / scale)
                scale = scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
                zero = zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        if self.sym:
            result = sym_quant_dequant(x, scale, maxq)
        else:
            result = asym_quant_dequant(x, scale, zero, maxq)

        return result.to(x_dtype)


# ============================================================
# Hadamard rotation utilities
# ============================================================

def _is_pow2(n):
    """Check if n is a power of 2."""
    return (n > 0) and (n & (n - 1) == 0)


def hadamard_matrix(size, device=None, dtype=torch.float32):
    """Return the normalized Walsh-Hadamard matrix of shape (size, size)."""
    assert _is_pow2(size), f"Hadamard dimension must be a power of 2, got {size}"
    eye = torch.eye(size, device=device, dtype=dtype)
    return _hadamard_transform_pytorch(eye, scale=1.0 / math.sqrt(size))


@torch.no_grad()
def apply_exact_had_to_linear(module, had_dim=-1, output=False, R=None, R2=None):
    """
    Absorb a normalized Hadamard rotation into an nn.Linear weight.

    Args:
        module: linear layer to rotate.
        had_dim: block size for independent Hadamard rotations. Qwen3-8B uses
            head_dim=128 for V/O rotation.
        output: if True, rotate output features (W <- H W); otherwise rotate
            input features (W <- W H).
        R/R2: optional explicit rotation matrix. R2 is accepted for compatibility
            with torchao SpinQuant helpers.
    """
    assert isinstance(module, nn.Linear)
    if R is None:
        R = R2

    weight = module.weight.data
    dtype_orig = weight.dtype
    weight_f = weight.float()

    if had_dim == -1:
        had_dim = module.out_features if output else module.in_features
    assert _is_pow2(had_dim), f"Hadamard dimension must be a power of 2, got {had_dim}"

    if R is None:
        had = hadamard_matrix(had_dim, device=weight_f.device, dtype=torch.float32)
    else:
        had = R.to(device=weight_f.device, dtype=torch.float32)
        assert had.shape == (had_dim, had_dim), (
            f"Rotation shape {tuple(had.shape)} does not match had_dim={had_dim}"
        )

    if output:
        if module.out_features % had_dim != 0:
            raise ValueError(f"out_features={module.out_features} is not divisible by had_dim={had_dim}")
        # Rotate each output-head block: W' = H W. This is equivalent to
        # applying H to the linear output activation.
        reshaped = weight_f.reshape(module.out_features // had_dim, had_dim, module.in_features)
        rotated = torch.matmul(had, reshaped)
        module.weight.data = rotated.reshape_as(weight_f).to(dtype=dtype_orig)

        if module.bias is not None:
            bias_dtype = module.bias.data.dtype
            bias = module.bias.data.float().reshape(module.out_features // had_dim, had_dim)
            module.bias.data = torch.matmul(bias, had).reshape_as(module.bias.data).to(dtype=bias_dtype)
    else:
        if module.in_features % had_dim != 0:
            raise ValueError(f"in_features={module.in_features} is not divisible by had_dim={had_dim}")
        # Rotate each input-head block: W' = W H. This lets a rotated
        # attention output feed an equivalent downstream projection.
        reshaped = weight_f.reshape(module.out_features, module.in_features // had_dim, had_dim)
        rotated = torch.matmul(reshaped, had)
        module.weight.data = rotated.reshape_as(weight_f).to(dtype=dtype_orig)


@torch.no_grad()
def rotate_ov_proj(layer, had_dim, R=None):
    """Rotate one decoder layer's V projection output and O projection input."""
    attn = layer.self_attn
    apply_exact_had_to_linear(attn.v_proj, had_dim=had_dim, output=True, R=R)
    apply_exact_had_to_linear(attn.o_proj, had_dim=had_dim, output=False, R=R)


@torch.no_grad()
def rotate_qwen3_ov_proj(model, had_dim=None, R=None):
    """
    Apply offline V/O Hadamard rotation to every Qwen3 decoder layer.

    The model then produces/stores V in rotated head space, so runtime V-cache
    quantization must skip the extra H/quant/H wrapper.
    """
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("Could not find decoder layers at model.model.layers or model.layers")
    if had_dim is None:
        cfg = getattr(model, "config", None)
        had_dim = getattr(cfg, "head_dim", None)
        if had_dim is None:
            hidden_size = getattr(cfg, "hidden_size")
            num_heads = getattr(cfg, "num_attention_heads")
            had_dim = hidden_size // num_heads
    for layer in layers:
        rotate_ov_proj(layer, had_dim=had_dim, R=R)
    return model


# ============================================================
# Norm factoring helpers: extract per-token norm, normalize, restore
# ============================================================

def _norm_factor_extract(x, method):
    """
    Extract per-token norm and normalize x.

    Args:
        x: (B, nh, T, D)
        method: "norm_l2" | "norm_rsqrt" | "norm_max"

    Returns:
        (x_normed, norm_info) where norm_info is used by _norm_factor_restore.
        norm_info is always the norm value (for restore via multiplication).
    """
    if method == "norm_l2":
        # L2 norm: sqrt(sum(x^2)) across heads and head_dim
        k_norms = x.norm(dim=(1, 3), keepdim=True).clamp(min=1e-8)  # (B, 1, T, 1)
        return x / k_norms, k_norms
    elif method == "norm_rsqrt":
        # rsqrt: use x * rsqrt(sum(x^2)) to normalize (avoids sqrt+div)
        # Compute in float32 for numerical stability
        sq_sum = (x.float() * x.float()).sum(dim=(1, 3), keepdim=True)  # (B, 1, T, 1)
        inv_norm = torch.rsqrt(sq_sum.clamp(min=1e-8))  # (B, 1, T, 1)
        x_normed = x * inv_norm.to(x.dtype)
        # Store the norm (not inv_norm) for restore via multiplication
        k_norms = torch.sqrt(sq_sum.clamp(min=1e-8)).to(x.dtype)  # (B, 1, T, 1)
        return x_normed, k_norms
    elif method == "norm_max":
        # L∞ norm: max(|x|) across heads and head_dim
        k_norms = x.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8)  # (B, 1, T, 1)
        return x / k_norms, k_norms
    elif method == "norm_L1":
        # L1 norm (mean-abs): mean(|x|) across heads and head_dim
        k_norms = x.abs().mean(dim=(1, 3), keepdim=True).clamp(min=1e-8)  # (B, 1, T, 1)
        return x / k_norms, k_norms
    else:
        raise ValueError(f"Unknown norm factoring method: {method}")


def _norm_factor_restore(x, norm_info, method):
    """
    Restore original magnitudes after quantization.

    Args:
        x: (B, nh, T, D) quantized tensor
        norm_info: norm values returned by _norm_factor_extract
        method: "norm_l2" | "norm_rsqrt" | "norm_max"

    Returns:
        x with original magnitudes restored.
    """
    # All methods store the norm and restore via multiplication
    return x * norm_info


def _hadamard_transform_pytorch(x, scale=1.0):
    """
    Pure PyTorch implementation of the Walsh-Hadamard Transform (WHT).
    Uses the recursive butterfly (fast) algorithm, O(n log n).

    Args:
        x: tensor of shape (..., n) where n must be a power of 2
        scale: scaling factor applied to the result

    Returns:
        Hadamard-transformed tensor of same shape, multiplied by scale.
    """
    n = x.shape[-1]
    assert _is_pow2(n), f"Hadamard transform requires power-of-2 dimension, got {n}"

    # Butterfly stages
    h = 1
    while h < n:
        # Split into pairs of size h
        x_reshape = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_reshape[..., 0, :]  # even
        b = x_reshape[..., 1, :]  # odd
        x_reshape = torch.stack([a + b, a - b], dim=-2)
        x = x_reshape.view(*x.shape[:-1], n)
        h *= 2

    return x * scale


# Try to import fast_hadamard_transform (CUDA accelerated);
# fall back to pure PyTorch implementation if unavailable.
try:
    from fast_hadamard_transform import hadamard_transform as _cuda_hadamard_transform
    _HAS_FAST_HADAMARD = True
except ImportError:
    _HAS_FAST_HADAMARD = False


def hadamard_rotation(x, dtype=None):
    """
    Apply Hadamard rotation to tensor x along the last dimension.
    Uses fast_hadamard_transform (CUDA) if available, otherwise falls back
    to a pure PyTorch implementation.

    Args:
        x: tensor of shape (..., head_dim) where head_dim must be a power of 2
        dtype: optional dtype to cast to before transform (e.g., torch.float32 for stability)

    Returns:
        Hadamard-rotated tensor of same shape.
    """
    x_dtype = x.dtype
    n = x.shape[-1]

    if dtype is not None:
        x = x.to(dtype)
    else:
        x = x.float()

    scale = 1.0 / math.sqrt(n)

    if _HAS_FAST_HADAMARD:
        x = _cuda_hadamard_transform(x.contiguous(), scale=scale)
    else:
        x = _hadamard_transform_pytorch(x.contiguous(), scale=scale)

    return x.to(x_dtype)


def is_pow2(n):
    """Check if n is a power of 2."""
    return (n > 0) and (n & (n - 1) == 0)


# ============================================================
# Main QuaRot KV Cache Quantizer
# ============================================================

class QuaRotKVCacheQuantizer:
    """
    QuaRot KV Cache Quantizer (KIVI-style residual management).

    Applies Hadamard rotation to Q,K (after RoPE) and quantizes K,V to low bits.
    K rotation preserves attention scores since Hadamard matrices are orthogonal:
        (H @ Q) @ (H @ K)^T = Q @ H^T @ H @ K^T = Q @ K^T
    V is also Hadamard-rotated before quantization (inline, during quantize_kv_cache)
    to make the distribution more uniform, then inverse-rotated after dequantization:
        V_approx = H(quantize_dequantize(H(V)))  (H is self-inverse)

    Following KIVI (ICML 2024):
      - K cache: per-channel quantization (statistics along seq_len, one scale per head_dim channel)
      - V cache: per-token quantization (statistics along head_dim, one scale per token)
      - residual_length: buffer size for full-precision tokens (KIVI-style flush)

    KIVI-style residual management (avoids repeated fake-quantization):
      - Each token is fake-quantized exactly ONCE, then the result is written back
        to the KV cache. Subsequent decode steps do NOT re-quantize old tokens.
      - V cache (per-token): flush mode controlled by v_flush_mode parameter:
        "block" (Oscar default): same as K — flush entire buffer when buffer >= res_len.
        "token" (original KIVI): flush oldest 1 token when buffer > res_len,
        buffer always stays at ~res_len.
      - K cache (per-channel): buffer flush style (matching KIVI exactly) —
        new tokens accumulate in a full-precision buffer; when the buffer reaches
        residual_length, the ENTIRE buffer is fake-quantized at once and committed,
        then the buffer is cleared. This means: (1) per-channel statistics are computed
        over the full residual_length block, and (2) at the moment of flush, there are
        0 full-precision K tokens (same as KIVI).
        The k_groupsize parameter controls the quantization granularity (blockwise
        per-channel within each flushed block), NOT the flush timing.
      - K cache (global per-channel, k_groupsize = -1): falls back to re-quantizing
        all uncommitted tokens each step, since global per-channel statistics depend
        on the entire sequence and cannot be incrementally committed.

    Args:
        k_bits: quantization bits for K cache (2, 4, 8, or 16 for no quantization)
        v_bits: quantization bits for V cache (2, 4, 8, or 16 for no quantization)
        k_groupsize: groupsize for K quantization (-1 for global per-channel)
        v_groupsize: groupsize for V quantization (-1 for per-token)
        k_sym: symmetric quantization for K (True/False)
        v_sym: symmetric quantization for V (True/False)
        k_clip_ratio: clip ratio for K quantization
        v_clip_ratio: clip ratio for V quantization
        residual_length: buffer size — when full-precision K buffer reaches this
            length, flush all to quantized zone (KIVI-style). Must be divisible
            by k_groupsize when k_groupsize > 0.
        k_token_rotation: whether to apply blockwise token-dimension Hadamard
            rotation to K cache before per-channel quantization.  This uniformizes
            the L2-norm distribution across tokens within each flush block,
            reducing per-channel quantization error.
            Flow: transpose(2,3) → hadamard → transpose(2,3) → QDQ
                  → transpose(2,3) → hadamard → transpose(2,3)  (inverse)
            Requires residual_length to be a power of 2 (for Hadamard).
    """

    def __init__(
        self,
        k_bits=4,
        v_bits=4,
        k_groupsize=32,
        v_groupsize=32,
        k_sym=False,
        v_sym=False,
        k_clip_ratio=1.0,
        v_clip_ratio=1.0,
        residual_length=0,
        k_token_rotation=False,
        k_norm_factoring=False,
        use_hadamard=True,
        offline_v_hadamard=False,
        v_flush_mode="block",
    ):
        # K cache: per-channel quantization (per_channel=True)
        self.k_quantizer = KVQuantizer(bits=k_bits, groupsize=k_groupsize, sym=k_sym,
                                        clip_ratio=k_clip_ratio, per_channel=True)
        # V cache: per-token quantization (per_channel=False, default)
        self.v_quantizer = KVQuantizer(bits=v_bits, groupsize=v_groupsize, sym=v_sym,
                                        clip_ratio=v_clip_ratio, per_channel=False)
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_groupsize = k_groupsize
        self.residual_length = residual_length
        self.k_token_rotation = k_token_rotation
        # Normalize k_norm_factoring: True → "norm_rsqrt" (default), False → None
        if k_norm_factoring is True:
            self.k_norm_factoring = "norm_rsqrt"
        elif k_norm_factoring is False or k_norm_factoring is None:
            self.k_norm_factoring = None
        else:
            assert k_norm_factoring in ("norm_l2", "norm_rsqrt", "norm_max", "norm_L1"), \
                f"k_norm_factoring must be False/None/'norm_l2'/'norm_rsqrt'/'norm_max'/'norm_L1', got {k_norm_factoring}"
            self.k_norm_factoring = k_norm_factoring
        self.use_hadamard = use_hadamard
        self.offline_v_hadamard = offline_v_hadamard
        assert v_flush_mode in ("block", "token"), \
            f"v_flush_mode must be 'block' or 'token', got {v_flush_mode}"
        self.v_flush_mode = v_flush_mode

        # Validate: residual_length must be divisible by k_groupsize (same as KIVI)
        if k_groupsize > 0 and residual_length > 0:
            assert residual_length % k_groupsize == 0, \
                f"residual_length ({residual_length}) must be divisible by k_groupsize ({k_groupsize})"

        # Validate: when k_token_rotation is enabled, residual_length must be a power of 2
        # because Hadamard transform requires the last dimension to be a power of 2,
        # and the token-dimension rotation operates on blocks of residual_length tokens.
        if k_token_rotation and residual_length > 0:
            assert is_pow2(residual_length), \
                f"residual_length ({residual_length}) must be a power of 2 when k_token_rotation is enabled"

        # KIVI-style tracking: number of tokens already fake-quantized (committed)
        # Tokens at positions [0, committed_len) have been fake-quantized exactly once
        # and will NOT be re-quantized in future decode steps.
        self.committed_k_len = 0
        self.committed_v_len = 0

    def process_kv(self, query_states, key_states, value_states):
        """
        Apply Hadamard rotation to Q, K (no quantization here).

        Rotation is applied to ALL tokens (both prefill and decode) since Q must
        always be in the rotated space to compute correct attention with rotated K.
        Quantization is deferred to quantize_kv_cache() which is called after
        cache update, following KIVI's approach.

        Note: V rotation is NOT done here. V is Hadamard-rotated inline during
        quantization in quantize_kv_cache() (rotate → quant/dequant → inverse_rotate),
        so V is stored unrotated in the cache.

        Args:
            query_states: (bsz, num_heads, seq_len, head_dim)
            key_states: (bsz, num_kv_heads, seq_len, head_dim)
            value_states: (bsz, num_kv_heads, seq_len, head_dim)

        Returns:
            (query_states, key_states, value_states) with Q, K Hadamard-rotated.
            V is returned unchanged.
        """
        dtype = query_states.dtype

        if self.k_bits < 16 and self.use_hadamard:
            query_states = hadamard_rotation(query_states).to(dtype)
            key_states = hadamard_rotation(key_states).to(dtype)

        return query_states, key_states, value_states

    def quantize_kv_cache(self, key_states, value_states):
        """
        Fake-quantize the KV cache using KIVI-style residual management.

        Each token is fake-quantized exactly ONCE. Already-committed tokens are
        never re-quantized, avoiding accumulated rounding errors.

        Called after cache update, before attention computation.
        NOT called during prefill (first forward), so the first generated token
        sees full-precision K/V.

        V cache (per-token, mode depends on v_flush_mode):
            "block" (Oscar): flush entire buffer when buffer >= res_len, buffer → 0.
            "token" (original KIVI): flush oldest 1 token when buffer > res_len,
            buffer stays at ~res_len.

        K cache (per-channel, buffer flush — matching KIVI):
            New tokens accumulate in the full-precision buffer. When the buffer
            reaches residual_length, the ENTIRE buffer is fake-quantized at once
            and committed, then the buffer is cleared.
            This matches KIVI exactly: at the flush moment, 0 full-precision K
            tokens exist.
            The k_groupsize parameter controls quantization granularity within
            each flushed block (blockwise per-channel), NOT the flush timing.
            If k_groupsize=-1 (global per-channel), falls back to re-quantizing
            all non-committed tokens each step.

        Args:
            key_states: (bsz, num_kv_heads, total_seq_len, head_dim) — full KV cache
            value_states: (bsz, num_kv_heads, total_seq_len, head_dim)

        Returns:
            (key_states, value_states) with newly eligible tokens fake-quantized.
        """
        seq_len = key_states.shape[2]
        res_len = self.residual_length

        # ==================== V cache: per-token quantization ====================
        if self.v_bits < 16:
            v_buffer_len = seq_len - self.committed_v_len
            if self.v_flush_mode == "token":
                # --- Per-token flush (original KIVI): flush oldest 1 token when buffer > res_len ---
                # Buffer always stays at ~res_len. Per-token quantization is independent,
                # so single-token QDQ is valid.
                if v_buffer_len > res_len and res_len > 0:
                    flush_pos = self.committed_v_len  # oldest token in buffer

                    v_flush = value_states[:, :, flush_pos:flush_pos+1, :]
                    if self.use_hadamard and not self.offline_v_hadamard:
                        v_flush_rotated = hadamard_rotation(v_flush)
                        v_flush_quantized = self.v_quantizer.quantize_dequantize(v_flush_rotated)
                        v_flush_quantized = hadamard_rotation(v_flush_quantized)
                    else:
                        v_flush_quantized = self.v_quantizer.quantize_dequantize(v_flush)

                    value_states = torch.cat([
                        value_states[:, :, :flush_pos, :],       # already committed
                        v_flush_quantized,                        # 1 newly quantized token
                        value_states[:, :, flush_pos+1:, :],     # rest of buffer (FP16)
                    ], dim=2)
                    self.committed_v_len = flush_pos + 1
            else:
                # --- Block flush (Oscar default): flush entire buffer when buffer >= res_len ---
                if v_buffer_len >= res_len and res_len > 0:
                    flush_start = self.committed_v_len
                    flush_end = seq_len

                    v_flush = value_states[:, :, flush_start:flush_end, :]
                    if self.use_hadamard and not self.offline_v_hadamard:
                        v_flush_rotated = hadamard_rotation(v_flush)
                        v_flush_quantized = self.v_quantizer.quantize_dequantize(v_flush_rotated)
                        v_flush_quantized = hadamard_rotation(v_flush_quantized)
                    else:
                        v_flush_quantized = self.v_quantizer.quantize_dequantize(v_flush)

                    value_states = torch.cat([
                        value_states[:, :, :flush_start, :],  # already committed
                        v_flush_quantized,                     # newly quantized (entire buffer)
                    ], dim=2)
                    self.committed_v_len = flush_end

        # ==================== K cache: per-channel, buffer flush ====================
        # KIVI-style: accumulate tokens in full-precision buffer;
        # when buffer reaches residual_length, flush the ENTIRE buffer to quantized.
        if self.k_bits < 16:
            if self.k_groupsize > 0:
                # --- Blockwise per-channel: KIVI-style buffer flush ---
                # Buffer = tokens at positions [committed_k_len, seq_len)
                # When buffer reaches residual_length, flush ALL of them at once.
                k_buffer_len = seq_len - self.committed_k_len

                if k_buffer_len >= res_len and res_len > 0:
                    # ★ KIVI-style flush: quantize the entire buffer at once ★
                    # The buffer contains exactly residual_length tokens.
                    # k_groupsize controls per-channel block granularity within
                    # this flushed block (handled by KVQuantizer internally).
                    flush_start = self.committed_k_len
                    flush_end = seq_len  # flush everything, buffer becomes empty

                    k_flush = key_states[:, :, flush_start:flush_end, :]

                    if self.k_token_rotation:
                        # ★ Optional: blockwise token-dimension Hadamard rotation ★
                        k_flush = k_flush.transpose(2, 3)            # (B, nh, D, G)
                        k_flush = hadamard_rotation(k_flush)         # rotate along token dim
                        k_flush = k_flush.transpose(2, 3)            # (B, nh, G, D)

                    if self.k_norm_factoring:
                        k_flush, k_norm_info = _norm_factor_extract(k_flush, self.k_norm_factoring)

                    k_flush_quantized = self.k_quantizer.quantize_dequantize(k_flush)

                    if self.k_norm_factoring:
                        k_flush_quantized = _norm_factor_restore(k_flush_quantized, k_norm_info, self.k_norm_factoring)

                    if self.k_token_rotation:
                        # Inverse rotation: restore original token space
                        k_flush_quantized = k_flush_quantized.transpose(2, 3)  # (B, nh, D, G)
                        k_flush_quantized = hadamard_rotation(k_flush_quantized)  # H² = I
                        k_flush_quantized = k_flush_quantized.transpose(2, 3)  # (B, nh, G, D)

                    # Write back
                    key_states = torch.cat([
                        key_states[:, :, :flush_start, :],   # already committed, don't touch
                        k_flush_quantized,                    # newly fake-quantized (entire buffer)
                    ], dim=2)
                    self.committed_k_len = flush_end
                    # Buffer is now empty (0 full-precision K tokens), same as KIVI
                # else: buffer not full yet, keep accumulating (all remain full precision)
            else:
                # --- Global per-channel: must re-quantize all non-residual tokens ---
                # Global per-channel computes scale across entire seq_len, so when
                # new tokens arrive the scale changes. Cannot avoid re-quantization.
                # This is the original behavior (fallback).
                if res_len > 0 and seq_len > res_len:
                    quant_boundary = seq_len - res_len
                    k_early = key_states[:, :, :quant_boundary, :]
                    k_recent = key_states[:, :, quant_boundary:, :]
                    k_early = self.k_quantizer.quantize_dequantize(k_early)
                    key_states = torch.cat([k_early, k_recent], dim=2)
                else:
                    key_states = self.k_quantizer.quantize_dequantize(key_states)

        return key_states, value_states

    def quantize_prefill(self, key_states, value_states):
        """
        Fake-quantize the KV cache after prefill (matching KIVI's prefill behavior).

        In KIVI, after prefill attention is computed with full precision, the cache
        is immediately split into quantized zone + residual buffer:
          - K: tokens that form complete residual_length-sized blocks are quantized;
               the remainder stays full precision in the buffer.
          - V: tokens beyond the last residual_length are quantized;
               the last residual_length tokens stay full precision.

        This method should be called AFTER the prefill attention computation
        (so prefill sees full-precision KV), but BEFORE returning the cache
        (so subsequent decode steps start with a properly partitioned cache).

        Args:
            key_states: (bsz, num_kv_heads, prefill_seq_len, head_dim)
            value_states: (bsz, num_kv_heads, prefill_seq_len, head_dim)

        Returns:
            (key_states, value_states) with prefill tokens fake-quantized.
        """
        seq_len = key_states.shape[2]
        res_len = self.residual_length

        if res_len <= 0 or seq_len <= res_len:
            # Entire sequence fits in buffer, nothing to quantize
            return key_states, value_states

        # ==================== K cache: quantize complete blocks ====================
        if self.k_bits < 16:
            # KIVI prefill: quantize tokens [0 : seq_len - (seq_len % res_len)]
            # i.e., as many complete residual_length-sized blocks as possible.
            # The remainder stays in the full-precision buffer.
            if seq_len % res_len != 0:
                k_quant_end = seq_len - (seq_len % res_len)
            else:
                # All tokens form complete blocks → quantize all, buffer = None
                k_quant_end = seq_len

            if k_quant_end > 0:
                k_quant = key_states[:, :, :k_quant_end, :]

                if self.k_token_rotation:
                    # ★ Blockwise token-dimension Hadamard rotation (prefill) ★
                    bsz, nh, T_quant, D = k_quant.shape
                    num_blocks = T_quant // res_len
                    k_quant = k_quant.reshape(bsz, nh, num_blocks, res_len, D)
                    k_quant = k_quant.transpose(3, 4)
                    k_quant = hadamard_rotation(k_quant)
                    k_quant = k_quant.transpose(3, 4)
                    k_quant = k_quant.reshape(bsz, nh, T_quant, D)

                if self.k_norm_factoring:
                    k_quant, k_norm_info = _norm_factor_extract(k_quant, self.k_norm_factoring)

                k_quant = self.k_quantizer.quantize_dequantize(k_quant)

                if self.k_norm_factoring:
                    k_quant = _norm_factor_restore(k_quant, k_norm_info, self.k_norm_factoring)

                if self.k_token_rotation:
                    # Inverse rotation: restore original token space
                    bsz, nh, T_quant, D = k_quant.shape
                    num_blocks = T_quant // res_len
                    k_quant = k_quant.reshape(bsz, nh, num_blocks, res_len, D)
                    k_quant = k_quant.transpose(3, 4)
                    k_quant = hadamard_rotation(k_quant)  # H² = I (self-inverse)
                    k_quant = k_quant.transpose(3, 4)
                    k_quant = k_quant.reshape(bsz, nh, T_quant, D)

                key_states = torch.cat([
                    k_quant,
                    key_states[:, :, k_quant_end:, :],
                ], dim=2)
                self.committed_k_len = k_quant_end

        # ==================== V cache: quantize prefill tokens, keep FP buffer ====================
        if self.v_bits < 16:
            if self.v_flush_mode == "token":
                # --- Per-token flush (original KIVI): keep last res_len as FP buffer ---
                if seq_len > res_len:
                    v_quant_end = seq_len - res_len
                else:
                    v_quant_end = 0
            else:
                # --- Block flush (Oscar): block-aligned, same as K ---
                if seq_len % res_len != 0:
                    v_quant_end = seq_len - (seq_len % res_len)
                else:
                    v_quant_end = seq_len

            if v_quant_end > 0:
                v_quant = value_states[:, :, :v_quant_end, :]
                if self.use_hadamard and not self.offline_v_hadamard:
                    v_quant_rotated = hadamard_rotation(v_quant)
                    v_quant_quantized = self.v_quantizer.quantize_dequantize(v_quant_rotated)
                    v_quant_quantized = hadamard_rotation(v_quant_quantized)
                else:
                    v_quant_quantized = self.v_quantizer.quantize_dequantize(v_quant)
                value_states = torch.cat([
                    v_quant_quantized,
                    value_states[:, :, v_quant_end:, :],
                ], dim=2)
                self.committed_v_len = v_quant_end

        return key_states, value_states


def init_quarot(module, k_bits=4, v_bits=4, k_groupsize=32, v_groupsize=32,
                k_sym=False, v_sym=False, k_clip_ratio=1.0, v_clip_ratio=1.0,
                residual_length=0, k_token_rotation=False, k_norm_factoring=False,
                use_hadamard=True, offline_v_hadamard=False, v_flush_mode="block"):
    """
    Initialize QuaRot/KIVI KV Cache Quantizer on an attention module.
    Called during the first forward pass (prefill).

    Args:
        module: attention module to attach the quantizer to
        residual_length: number of most recent tokens to keep in full precision (0 = quantize all)
        k_token_rotation: whether to apply blockwise token-dimension Hadamard rotation
            to K cache before per-channel quantization (default: False)
        k_norm_factoring: whether to normalize per-token L2 norm before per-channel
            quantization, then restore after (default: False)
        use_hadamard: whether to apply Hadamard rotation (True=QuaRot, False=KIVI)
        v_flush_mode: "block" (Oscar, flush entire buffer) or "token" (original KIVI,
            flush oldest 1 token, buffer stays at res_len)
    """
    module.quarot_quantizer = QuaRotKVCacheQuantizer(
        k_bits=k_bits,
        v_bits=v_bits,
        k_groupsize=k_groupsize,
        v_groupsize=v_groupsize,
        k_sym=k_sym,
        v_sym=v_sym,
        k_clip_ratio=k_clip_ratio,
        v_clip_ratio=v_clip_ratio,
        residual_length=residual_length,
        k_token_rotation=k_token_rotation,
        k_norm_factoring=k_norm_factoring,
        use_hadamard=use_hadamard,
        offline_v_hadamard=offline_v_hadamard,
        v_flush_mode=v_flush_mode,
    )
