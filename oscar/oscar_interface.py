# Copyright (c) 2025, Dayou Du.

from typing import Optional, Union

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import oscar_cuda as oscar_cuda


def _should_pad_qpack_outputs(
    k_cache: torch.Tensor,
    quant_mode: str,
    num_bits: int,
    seqlen_k: int,
) -> bool:
    return (
        k_cache.is_cuda
        and k_cache.dim() == 4
        and quant_mode == "k-channel"
        and num_bits == 2
        and seqlen_k >= 128
    )


def _allocate_padded_qpack_outputs(
    k_pack: torch.Tensor,
    k_params: torch.Tensor,
    v_pack: torch.Tensor,
    v_params: torch.Tensor,
    group_size: int,
    num_bits: int,
    pad_tokens: int = 128,
):
    pack_nums = 16 // num_bits
    pad_pack_rows = pad_tokens // pack_nums
    pad_param_rows = pad_tokens // group_size

    k_pack_pad = torch.empty(
        (k_pack.shape[0], k_pack.shape[1] + pad_pack_rows, k_pack.shape[2], k_pack.shape[3]),
        device=k_pack.device,
        dtype=k_pack.dtype,
    )
    k_params_pad = torch.empty(
        (k_params.shape[0], k_params.shape[1] + pad_param_rows, k_params.shape[2], k_params.shape[3]),
        device=k_params.device,
        dtype=k_params.dtype,
    )
    v_pack_pad = torch.empty(
        (v_pack.shape[0], v_pack.shape[1] + pad_tokens, v_pack.shape[2], v_pack.shape[3]),
        device=v_pack.device,
        dtype=v_pack.dtype,
    )
    v_params_pad = torch.empty(
        (v_params.shape[0], v_params.shape[1], v_params.shape[2], v_params.shape[3] + pad_tokens),
        device=v_params.device,
        dtype=v_params.dtype,
    )
    return k_pack_pad, k_params_pad, v_pack_pad, v_params_pad


def _copy_qpack_outputs_back(
    k_pack_src: torch.Tensor,
    k_params_src: torch.Tensor,
    v_pack_src: torch.Tensor,
    v_params_src: torch.Tensor,
    k_pack_dst: torch.Tensor,
    k_params_dst: torch.Tensor,
    v_pack_dst: torch.Tensor,
    v_params_dst: torch.Tensor,
):
    k_pack_dst.copy_(k_pack_src[:, : k_pack_dst.shape[1]])
    k_params_dst.copy_(k_params_src[:, : k_params_dst.shape[1]])
    v_pack_dst.copy_(v_pack_src[:, : v_pack_dst.shape[1]])
    v_params_dst.copy_(v_params_src[:, :, :, : v_params_dst.shape[3]])


def preprocess_k_cache(
    key_states: torch.Tensor,
    apply_hadamard: bool = False,
    apply_norm: bool = False,
):
    if not apply_hadamard and not apply_norm:
        return key_states, None

    key_states_out, key_norm = oscar_cuda.preprocess_k_cache(
        key_states.contiguous(),
        apply_hadamard,
        apply_norm,
    )
    return key_states_out, (key_norm if key_norm.numel() > 0 else None)

def kvcache_pack_int(k_cache: torch.Tensor, k_pack: torch.Tensor, k_params: torch.Tensor,
                     v_cache: torch.Tensor, v_pack: torch.Tensor, v_params: torch.Tensor,
                     opt_block_table: Optional[torch.Tensor] = None,
                     cu_seqlens_k: torch.Tensor = None,
                     seqlen_k: int = 0,
                     quant_mode: str = "k-tensor",
                     group_size: int = 128,
                     num_bits: int = 4,
                     apply_k_hadamard: bool = False,
                     apply_k_norm: bool = False):
    
    batch_size, seqlen_k, nheads_k, d = k_cache.shape
    if seqlen_k == 0:
        return None

    use_padded_outputs = _should_pad_qpack_outputs(k_cache, quant_mode, num_bits, seqlen_k)
    if use_padded_outputs:
        k_pack_work, k_params_work, v_pack_work, v_params_work = _allocate_padded_qpack_outputs(
            k_pack, k_params, v_pack, v_params, group_size, num_bits
        )
    else:
        k_pack_work, k_params_work, v_pack_work, v_params_work = k_pack, k_params, v_pack, v_params

    V_unpad = v_cache.reshape(batch_size * seqlen_k, nheads_k, d)
    if apply_k_hadamard or apply_k_norm:
        if num_bits == 4:
            k_norm = oscar_cuda.kvcache_pack_int4_preprocess_k(
                k_cache.contiguous(), k_pack_work, k_params_work,
                v_cache.contiguous(), v_pack_work, v_params_work,
                opt_block_table,
                cu_seqlens_k,
                seqlen_k,
                quant_mode,
                group_size,
                apply_k_hadamard,
                apply_k_norm,
            )
        elif num_bits == 2:
            k_norm = oscar_cuda.kvcache_pack_int2_preprocess_k(
                k_cache.contiguous(), k_pack_work, k_params_work,
                v_cache.contiguous(), v_pack_work, v_params_work,
                opt_block_table,
                cu_seqlens_k,
                seqlen_k,
                quant_mode,
                group_size,
                apply_k_hadamard,
                apply_k_norm,
            )
        else:
            raise NotImplementedError(f"Unsupported num_bits={num_bits}. Expected 2 or 4.")

        if use_padded_outputs:
            _copy_qpack_outputs_back(
                k_pack_work, k_params_work, v_pack_work, v_params_work,
                k_pack, k_params, v_pack, v_params,
            )
        return k_norm if k_norm.numel() > 0 else None

    K_unpad = k_cache.reshape(batch_size * seqlen_k, nheads_k, d)

    if num_bits == 4:
        oscar_cuda.kvcache_pack_int4(K_unpad, k_pack_work, k_params_work,
                                          V_unpad, v_pack_work, v_params_work,
                                          opt_block_table,
                                          cu_seqlens_k,
                                          seqlen_k,
                                          quant_mode,
                                          group_size
                                         )
    elif num_bits == 2:
        oscar_cuda.kvcache_pack_int2(K_unpad, k_pack_work, k_params_work,
                                          V_unpad, v_pack_work, v_params_work,
                                          opt_block_table,
                                          cu_seqlens_k,
                                          seqlen_k,
                                          quant_mode,
                                          group_size
                                         )
    else:
        raise NotImplementedError(f"Unsupported num_bits={num_bits}. Expected 2 or 4.")

    if use_padded_outputs:
        _copy_qpack_outputs_back(
            k_pack_work, k_params_work, v_pack_work, v_params_work,
            k_pack, k_params, v_pack, v_params,
        )

    return None

def fwd_kvcache_int(q: torch.Tensor, 
                    k_pack: torch.Tensor, k_params: torch.Tensor, 
                    v_pack: torch.Tensor, v_params: torch.Tensor,
                    opt_k_new: Optional[torch.Tensor] = None,
                    opt_v_new: Optional[torch.Tensor] = None,
                    opt_seqlens_k: Optional[torch.Tensor] = None,
                    k_pack_new: torch.Tensor = None, k_params_new: torch.Tensor = None,
                    v_pack_new: torch.Tensor = None, v_params_new: torch.Tensor = None,
                    opt_block_table: Optional[torch.Tensor] = None,
                    softmax_scale: float = 1.0,
                    quant_mode: str = "k-tensor",
                    group_size: int = 128,
                    residual_block_size: int = 128,
                    new_lens: int = 0,
                    num_bits: int = 4,
                    opt_k_norm: Optional[torch.Tensor] = None,
                    opt_k_norm_new: Optional[torch.Tensor] = None):
    
    if num_bits == 4:
        out_bit, k_pack_new, k_params_new, v_pack_new, v_params_new = oscar_cuda.fwd_kvcache_int4(
            q,
            k_pack, k_params, 
            v_pack, v_params,
            opt_k_new, opt_v_new, opt_seqlens_k,
            k_pack_new, k_params_new, v_pack_new, v_params_new,
            opt_block_table,
            softmax_scale,
            quant_mode, 
            group_size,
            residual_block_size,
            new_lens,
            False,          # Added
            -1,             # Added
            -1,             # Added
            0.0,            # Added
            True,           # Added
            0,              # Added
            opt_k_norm,
            opt_k_norm_new
        )
    elif num_bits == 2:
        out_bit, k_pack_new, k_params_new, v_pack_new, v_params_new = oscar_cuda.fwd_kvcache_int2(
            q,
            k_pack, k_params,
            v_pack, v_params,
            opt_k_new, opt_v_new, opt_seqlens_k,
            k_pack_new, k_params_new, v_pack_new, v_params_new,
            opt_block_table,
            softmax_scale,
            quant_mode,
            group_size,
            residual_block_size,
            new_lens,
            False,          # Added
            -1,             # Added
            -1,             # Added
            0.0,            # Added
            True,           # Added
            0,              # Added
            opt_k_norm,
            opt_k_norm_new
        )
    else:
        raise NotImplementedError(f"Unsupported num_bits={num_bits}. Expected 2 or 4.")


    return out_bit, k_pack_new, k_params_new, v_pack_new, v_params_new
