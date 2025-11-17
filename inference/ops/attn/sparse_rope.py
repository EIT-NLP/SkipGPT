import os
import torch
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, Any

from ..utils import generate_autotune_config

os.environ['TRITON_PRINT_AUTOTUNING']='0'

@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_M=[1, 2, 4, 8],
            num_stages=[3, 4],
        )
    ),
    key=['MK', 'HQ', 'N'],
)
@triton.jit(do_not_specialize=['MQ'])
def triton_rope_qk_align_fwd(
    q: tl.tensor,
    k: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    out_q_cos: tl.tensor,
    out_q_sin: tl.tensor,
    out_k_cos: tl.tensor,
    out_k_sin: tl.tensor,
    batch_ids: tl.tensor,
    pos_ids: tl.tensor,
    B: tl.constexpr,
    L: tl.constexpr,
    MQ: tl.int64,
    MK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    mid = tl.program_id(0)
    head_id_q = tl.program_id(1)
    GROUP_SIZE = HQ // HK
    head_id_k = head_id_q // GROUP_SIZE

    offset_m = (mid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    batch_id = offset_m // L
    pos_id = offset_m % L

    offset_emb = pos_id * N
    offset_q = offset_m * HQ * N + head_id_q * N
    offset_k = offset_m * HK * N + head_id_k * N
    emb_mask = (batch_id < B) & (pos_id < L)

    cos_left = tl.load(cos + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
    cos_right = tl.load(cos + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)
    sin_left = tl.load(sin + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
    sin_right = tl.load(sin + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)

    k_data_left = tl.load(k + offset_k[:, None] + tl.arange(0, N // 2)[None, :], mask=(offset_m < MK)[:, None], other=0)
    k_data_right = tl.load(k + offset_k[:, None] + tl.arange(N // 2, N)[None, :], mask=(offset_m < MK)[:, None], other=0)

    k_left_cos = k_data_left * cos_left
    k_left_sin = -k_data_right * sin_left
    k_right_cos = k_data_right * cos_right
    k_right_sin = k_data_left * sin_right

    tl.store(out_k_cos + offset_k[:, None] + tl.arange(0, N // 2)[None, :], k_left_cos.to(out_k_cos.dtype.element_ty), mask=(offset_m < MK)[:, None])
    tl.store(out_k_cos + offset_k[:, None] + tl.arange(N // 2, N)[None, :], k_right_cos.to(out_k_sin.dtype.element_ty), mask=(offset_m < MK)[:, None])
    tl.store(out_k_sin + offset_k[:, None] + tl.arange(0, N // 2)[None, :], k_left_sin.to(out_k_sin.dtype.element_ty), mask=(offset_m < MK)[:, None])
    tl.store(out_k_sin + offset_k[:, None] + tl.arange(N // 2, N)[None, :], k_right_sin.to(out_k_sin.dtype.element_ty), mask=(offset_m < MK)[:, None])

    if mid * BLOCK_M < MQ:
        batch_id = tl.load(batch_ids + offset_m, mask=offset_m < MQ, other=B)
        pos_id = tl.load(pos_ids + offset_m, mask=offset_m < MQ, other=L)

        offset_emb = pos_id * N
        offset_q = offset_m * HQ * N + head_id_q * N
        emb_mask = (batch_id < B) & (pos_id < L)

        cos_left = tl.load(cos + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
        cos_right = tl.load(cos + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)
        sin_left = tl.load(sin + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
        sin_right = tl.load(sin + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)
        
        q_data_left = tl.load(q + offset_q[:, None] + tl.arange(0, N // 2)[None, :], mask=(offset_m < MQ)[:, None], other=0)
        q_data_right = tl.load(q + offset_q[:, None] + tl.arange(N // 2, N)[None, :], mask=(offset_m < MQ)[:, None], other=0)
        
        q_left_cos = q_data_left * cos_left
        q_left_sin = -q_data_right * sin_left
        q_right_cos = q_data_right * cos_right
        q_right_sin = q_data_left * sin_right

        store_mask = (offset_m < MQ)[:, None]

        tl.store(out_q_cos + offset_q[:, None] + tl.arange(0, N // 2)[None, :], q_left_cos.to(out_q_cos.dtype.element_ty), mask=store_mask)
        tl.store(out_q_cos + offset_q[:, None] + tl.arange(N // 2, N)[None, :], q_right_cos.to(out_q_cos.dtype.element_ty), mask=store_mask)
        tl.store(out_q_sin + offset_q[:, None] + tl.arange(0, N // 2)[None, :], q_left_sin.to(out_q_sin.dtype.element_ty), mask=store_mask)
        tl.store(out_q_sin + offset_q[:, None] + tl.arange(N // 2, N)[None, :], q_right_sin.to(out_q_sin.dtype.element_ty), mask=store_mask)

# @torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False)
def triton_rope_qk_align(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_ids: torch.Tensor,
    pos_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to dense k and sparse q, avoiding apply addition in triton kernel to match the pytorch precision.
    Args:
        q: torch.Tensor with shape [nnz, num_heads, head_dim]
        k: torch.Tensor with shape [batch_size, seq_len, num_heads, head_dim]
        sin: torch.Tensor with shape [1, seq_len, head_dim]
        cos: torch.Tensor with shape [1, seq_len, head_dim]
        batch_ids: torch.Tensor with shape [nnz]
        pos_ids: torch.Tensor with shape [nnz]
    Returns:
        out_q: torch.Tensor with shape [nnz, num_heads, head_dim]\\
        out_k: torch.Tensor with shape [batch_size, seq_len, num_heads, head_dim]
    """

    out_q_cos, out_q_sin = torch.empty_like(q), torch.empty_like(q)
    out_k_cos, out_k_sin = torch.empty_like(k), torch.empty_like(k)

    assert k.dim() == 4 and q.dim() == 3
    B, L = k.shape[0], k.shape[1]
    MK, N = k.shape[0] * k.shape[1], q.shape[-1]
    HQ, HK = q.shape[-2], k.shape[-2]
    MQ = q.shape[0]
    grid = lambda META: (triton.cdiv(MK, META['BLOCK_M']), HQ)
    
    triton_rope_qk_align_fwd[grid](
        q, k,
        cos,
        sin,
        out_q_cos, out_q_sin,
        out_k_cos, out_k_sin,
        batch_ids, pos_ids,
        B, L, MQ, MK, HQ, HK, N,
    )
    return out_q_cos + out_q_sin, out_k_cos + out_k_sin