import os
import torch
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, Any

from ..utils import generate_autotune_config

os.environ['TRITON_PRINT_AUTOTUNING']='0'

@triton.heuristics({
    "HAS_BIAS_DOWN": lambda args: args['b_down'] is not None,
})
@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_M=[32, 64, 128],
            BLOCK_N=[32, 64, 128],
            BLOCK_K=[32, 64],
            GROUP_SIZE=[4],
        )
    ),
    key=['N', 'K'],
)
@triton.jit(do_not_specialize=['M'])
def triton_sparse_mlp_fwd(
    x: tl.tensor,
    out: tl.tensor,
    indices: tl.tensor,
    w_down: tl.tensor,
    b_down: tl.tensor,
    M: tl.int64,
    N: tl.constexpr,
    K: tl.constexpr,
    O: tl.constexpr,
    HAS_BIAS_DOWN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    tmid, tnid = tl.program_id(0), tl.program_id(1)
    BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

    # compute indices in groups
    # Group 1: A[0:BM], A[BM:2BM], ... A[(G-1)*BM:G*BM] -> B[...,0:BN]
    mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)
    x_ptr = tl.make_block_ptr(
        x,
        shape=(M, K),
        strides=(K, 1),
        offsets=(mid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    w_down_ptr = tl.make_block_ptr(
        w_down,
        shape=(K, N),
        strides=(1, K),
        offsets=(0, nid * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    # 1. compute (x @ w_down)
    acc_down = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
        x_data = tl.load(x_ptr, boundary_check=(0, 1), padding_option='zero')
        w_down_data = tl.load(w_down_ptr, boundary_check=(0, 1), padding_option='zero')

        acc_down = tl.dot(x_data, w_down_data, acc=acc_down, allow_tf32=ALLOW_TF32)

        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        w_down_ptr = tl.advance(w_down_ptr, (BLOCK_K, 0))
    
    if HAS_BIAS_DOWN:
        b_down_ptr = tl.make_block_ptr(
            b_down,
            shape=(N,),
            strides=(1,),
            offsets=(nid * BLOCK_N,),
            block_shape=(BLOCK_N,),
            order=(0,)
        )
        acc_down += tl.load(b_down_ptr, boundary_check=(0,), padding_option='zero').to(tl.float32)

    idx_offset = indices + mid * BLOCK_M
    idx = tl.load(idx_offset + tl.arange(0, BLOCK_M), mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M, other=O)

    out_offset = out + idx[:, None] * N + nid * BLOCK_N
    tl.store(out_offset + tl.arange(0, BLOCK_N)[None, :], acc_down.to(out.dtype.element_ty), mask=(idx[:, None] < O) & (tl.arange(0, BLOCK_N)[None, :] + nid * BLOCK_N < N))

def triton_sparse_mlp(
    x: torch.Tensor,
    indices: torch.Tensor,
    mlp_down_weight: torch.Tensor,
    mlp_down_bias: Optional[torch.Tensor]=None,
    max_length: Optional[int]=None,
) -> torch.Tensor:
    """
    Compute the following equation:

    out[indices] = x @ w_down + b_down

    Args:
        x: [total, intermediate_size]
        mlp_down_weight: [intermediate_size, hidden_size]
        mlp_down_bias: [hidden_size]
        max_length: the original B * L, with indices[i] < max_length
    Returns:
        out: [max_length, hidden_size]
    """
    M, K, N = x.shape[0], x.shape[1], mlp_down_weight.shape[1]
    O = max_length if max_length else indices.max().item()
    out = torch.zeros((O, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    allow_tf32 = x.dtype == torch.float32
    triton_sparse_mlp_fwd[grid](
        x,
        out,
        indices,
        mlp_down_weight,
        mlp_down_bias,
        M, N, K, O,
        ALLOW_TF32=allow_tf32,
    )
    return out