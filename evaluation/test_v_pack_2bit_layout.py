#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oscar import fwd_kvcache_int, kvcache_pack_int, preprocess_k_cache  # noqa: E402


class VPack2BitLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for OScaR CUDA kernels")
        torch.cuda.set_device(0)

    def test_nontrivial_2bit_v_pattern_preserves_second_half_of_group(self) -> None:
        """Catch 2bit V layouts that repeat dims 0..15 into dims 16..31."""
        torch.manual_seed(7)
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        batch_size = 1
        seqlen_k = 256
        nheads_q = 32
        nheads_k = 8
        head_dim = 128
        group_size = 32
        num_bits = 2
        pack_num = 16 // num_bits
        residual_block_size = 128

        # This group is exactly representable by 2bit affine quantization
        # because min=0, max=3, scale=1, zero=0. The second half deliberately
        # differs from the first half, so a 16-d repeat is immediately visible.
        group_pattern = torch.tensor(
            [
                1, 3, 3, 3, 2, 3, 1, 1,
                2, 2, 0, 1, 2, 0, 1, 0,
                3, 2, 1, 3, 2, 3, 2, 3,
                2, 0, 0, 0, 0, 1, 1, 0,
            ],
            device=device,
            dtype=dtype,
        )
        v_pattern = group_pattern.repeat(head_dim // group_size)
        value_states = v_pattern.view(1, 1, 1, head_dim).expand(
            batch_size, seqlen_k, nheads_k, head_dim
        ).contiguous()

        # q=0 makes the attention probabilities uniform, so the output should
        # be exactly the V pattern after pack/decode. K is still supplied because
        # the fused kernel requires a packed K cache.
        query_states = torch.zeros(
            batch_size, 1, nheads_q, head_dim, device=device, dtype=dtype
        )
        key_states = torch.randn(
            batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype
        )
        query_states, _ = preprocess_k_cache(query_states, apply_hadamard=True, apply_norm=False)
        key_states, key_norm = preprocess_k_cache(key_states, apply_hadamard=True, apply_norm=True)

        k_pack = torch.zeros(
            (batch_size, seqlen_k // pack_num, nheads_k, head_dim),
            device=device,
            dtype=torch.uint16,
        )
        k_params = torch.zeros(
            (batch_size, seqlen_k // group_size, nheads_k, head_dim),
            device=device,
            dtype=torch.float32,
        )
        v_pack = torch.zeros(
            (batch_size, seqlen_k, nheads_k, head_dim // pack_num),
            device=device,
            dtype=torch.uint16,
        )
        v_params = torch.zeros(
            (batch_size, head_dim // group_size, nheads_k, seqlen_k),
            device=device,
            dtype=torch.float32,
        )
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32
        )

        kvcache_pack_int(
            key_states,
            k_pack,
            k_params,
            value_states,
            v_pack,
            v_params,
            None,
            cu_seqlens_k,
            seqlen_k,
            "k-channel",
            group_size,
            num_bits,
            False,
            False,
        )

        k_pack_new = torch.empty(
            (batch_size, residual_block_size // pack_num, nheads_k, head_dim),
            device=device,
            dtype=torch.uint16,
        )
        k_params_new = torch.empty(
            (batch_size, residual_block_size // group_size, nheads_k, head_dim),
            device=device,
            dtype=torch.float32,
        )
        v_pack_new = torch.empty(
            (batch_size, residual_block_size, nheads_k, head_dim // pack_num),
            device=device,
            dtype=torch.uint16,
        )
        v_params_new = torch.empty(
            (batch_size, head_dim // group_size, nheads_k, residual_block_size),
            device=device,
            dtype=torch.float32,
        )

        out, *_ = fwd_kvcache_int(
            query_states,
            k_pack,
            k_params,
            v_pack,
            v_params,
            None,
            None,
            None,
            k_pack_new,
            k_params_new,
            v_pack_new,
            v_params_new,
            None,
            1.0 / math.sqrt(head_dim),
            "k-channel",
            group_size,
            residual_block_size,
            0,
            num_bits,
            key_norm,
            None,
        )

        expected = value_states.repeat_interleave(nheads_q // nheads_k, dim=2).mean(
            dim=1, keepdim=True
        )
        diff = (out.float() - expected.float()).abs()
        max_diff = diff.max().item()
        self.assertLessEqual(
            max_diff,
            1e-3,
            "2bit V pack/decode corrupted a nontrivial 32-d pattern; "
            f"max_diff={max_diff}, "
            f"out[:32]={out[0, 0, 0, :32].float().cpu().tolist()}, "
            f"expected[:32]={expected[0, 0, 0, :32].float().cpu().tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
