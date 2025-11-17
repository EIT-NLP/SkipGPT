import os
import torch
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, Any

from .utils import generate_autotune_config

os.environ['TRITON_PRINT_AUTOTUNING']='0'

@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_M=[1, 2, 4, 8],
            num_stages=[3, 4],
        )
    ),
    key=['N', 'BLOCK_N'],
)
@triton.jit(do_not_specialize=['M'])
def triton_rmsnorm_fwd(
    x: tl.tensor,
    out: tl.tensor,
    w: tl.tensor,
    eps: tl.constexpr,
    M: tl.int64,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    mid = tl.program_id(0)
    x_ptr = tl.make_block_ptr(
        x,
        shape=(M, N),
        strides=(N, 1),
        offsets=(mid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x_data = tl.load(tl.advance(x_ptr, (0, i * BLOCK_N)), boundary_check=(0, 1), padding_option='zero')
        accum += x_data * x_data
    
    x_mean = tl.sum(accum, axis=1, keep_dims=True) / N + eps

    w_ptr = tl.make_block_ptr(
        w,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    out_ptr = tl.make_block_ptr(
        out,
        shape=(M, N),
        strides=(N, 1),
        offsets=(mid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    x_rsqrt = tl.math.rsqrt(x_mean)
    for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x_data = tl.load(tl.advance(x_ptr, (0, i * BLOCK_N)), boundary_check=(0, 1), padding_option='zero')
        w_data = tl.load(tl.advance(w_ptr, (i * BLOCK_N,)), boundary_check=(0,), padding_option='zero')
        x_norm = x_data * x_rsqrt * w_data[None, :]

        tl.store(tl.advance(out_ptr, (0, i * BLOCK_N)), x_norm.to(x.dtype.element_ty), boundary_check=(0, 1))

def triton_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    An naive implementation of RMSNorm for the final dimension
    """
    original_shape = x.shape
    if x.dim() > 2: x_flatten = x.flatten(0, -2)
    else: x_flatten = x

    M, N = x_flatten.shape
    assert N == w.shape[0]

    out = torch.empty_like(x_flatten)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    triton_rmsnorm_fwd[grid](
        x,
        out,
        w,
        eps,
        M, N,
        BLOCK_N=min(triton.next_power_of_2(N), 1024)
    )
    return out.reshape(original_shape)


@triton.jit
def triton_sparse_rmsnorm_fwd(
    x: tl.tensor,
    mask: tl.tensor,
    mask_cumsum: tl.tensor,
    out: tl.tensor,
    indices: tl.tensor,
    w: tl.tensor,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    mid = tl.program_id(0)
    mask_data = tl.load(mask + mid)
    idx = tl.load(mask_cumsum + mid - 1, mask=mid > 0, other=0)

    if mask_data != 0:
        row_offset = mid * N
        accum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
            col_offset = i * BLOCK_N + tl.arange(0, BLOCK_N)
            x_data = tl.load(x + row_offset + col_offset, mask=col_offset < N, other=0.0)
            accum += x_data * x_data
        
        x_mean = tl.sum(accum) / N + eps
        x_rsqrt = tl.math.rsqrt(x_mean)

        for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
            col_offset = i * BLOCK_N + tl.arange(0, BLOCK_N)
            x_data = tl.load(x + row_offset + col_offset, mask=col_offset < N, other=0.0)
            w_data = tl.load(w + col_offset, mask=col_offset < N, other=0.0)
            x_norm = x_data * x_rsqrt * w_data

            out_row_offset = idx * N
            tl.store(out + out_row_offset + col_offset, x_norm.to(x.dtype.element_ty), mask=col_offset < N)
            tl.store(indices + idx, mid)

def triton_sparse_rmsnorm(
    x: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    An implementation of sparse RMSNorm for the final dimension
    Args:
        x: torch.Tensor with shape [M, N]
        mask: torch.Tensor with shape [M], 1 -> execute, 0 -> skip
    Returns:
        out: torch.Tensor with shape [O, N], where O = mask.sum()\\
        indices: torch.Tensor with shape [O], where RMSNorm(x[indices]) = out
    """
    M, N = x.shape
    assert N == w.shape[0]
    assert mask.shape[0] == M and mask.dim() == 1

    mask_cumsum = torch.cumsum(mask, dim=0)
    O = mask_cumsum[-1].item()
    out = torch.empty((O, N), dtype=x.dtype, device=x.device)
    indices = torch.empty((O,), dtype=torch.int64, device=x.device)
    grid = lambda META: (M,)
    triton_sparse_rmsnorm_fwd[grid](
        x,
        mask,
        mask_cumsum,
        out,
        indices,
        w,
        eps,
        N,
        BLOCK_N=min(triton.next_power_of_2(N), 1024)
    )
    return out, indices


@triton.jit
def triton_sparse_rmsnorm_before_attn_fwd(
    x: tl.tensor,
    mask: tl.tensor,
    mask_cumsum: tl.tensor,
    out: tl.tensor,
    out_dense: tl.tensor,
    indices: tl.tensor,
    batch_indices: tl.tensor,
    seq_indices: tl.tensor,
    w: tl.tensor,
    eps: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    mid = tl.program_id(0)
    mask_data = tl.load(mask + mid)
    idx = tl.load(mask_cumsum + mid - 1, mask=mid > 0, other=0)
    batch_id = mid // L
    seq_id = mid % L

    row_offset = mid * D
    out_row_offset = idx * D
    accum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in tl.range(0, tl.cdiv(D, BLOCK_N)):
        col_offset = i * BLOCK_N + tl.arange(0, BLOCK_N)
        x_data = tl.load(x + row_offset + col_offset, mask=col_offset < D, other=0.0)
        accum += x_data * x_data
    
    x_mean = tl.sum(accum) / D + eps
    x_rsqrt = tl.math.rsqrt(x_mean)

    for i in tl.range(0, tl.cdiv(D, BLOCK_N)):
        col_offset = i * BLOCK_N + tl.arange(0, BLOCK_N)
        x_data = tl.load(x + row_offset + col_offset, mask=col_offset < D, other=0.0)
        w_data = tl.load(w + col_offset, mask=col_offset < D, other=0.0)
        x_norm = x_data * x_rsqrt * w_data

        tl.store(out_dense + row_offset + col_offset, x_norm.to(x.dtype.element_ty), mask=col_offset < D)
        if mask_data != 0:
            tl.store(out + out_row_offset + col_offset, x_norm.to(x.dtype.element_ty), mask=col_offset < D)
            tl.store(indices + idx, mid)
            tl.store(batch_indices + idx, batch_id)
            tl.store(seq_indices + idx, seq_id)

def triton_sparse_rmsnorm_before_attn(
    x: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    An implementation of sparse RMSNorm before query sparse attention
    Args:
        x: torch.Tensor with shape [B, L, D]
        mask: torch.Tensor with shape [B, L], 1 -> execute, 0 -> skip
    Returns:
        out: torch.Tensor with shape [O, N], where O = mask.sum(), corresponding to query\\
        out_dense: torch.Tensor with shape [B, L, D], corresponding to key/value\\
        indices: torch.Tensor with shape [O], where RMSNorm(x.flatten()[indices]) = out\\
        batch_indices: torch.Tensor with shape [O], where batch_indices[i] is out[i]'s batch_id\\
        seq_indices: torch.Tensor with shape [O], where seq_indices[i] is out[i]'s position
    """
    assert x.dim() == 3

    original_shape = x.shape
    B, L, D = x.shape
    assert mask.shape == x.shape[:-1]

    x_flatten = x.flatten(0, 1)
    mask_flatten = mask.flatten()

    mask_cumsum = torch.cumsum(mask_flatten, dim=0)
    O = mask_cumsum[-1].item()
    out = torch.empty((O, D), dtype=x.dtype, device=x.device)
    out_dense = torch.empty_like(x_flatten)
    indices = torch.empty((O,), dtype=torch.int64, device=x.device)
    batch_indices = torch.empty((O,), dtype=torch.int64, device=x.device)
    seq_indices = torch.empty((O,), dtype=torch.int64, device=x.device)
    grid = lambda META: (B * L,)
    triton_sparse_rmsnorm_before_attn_fwd[grid](
        x_flatten,
        mask_flatten,
        mask_cumsum,
        out,
        out_dense,
        indices,
        batch_indices,
        seq_indices,
        w,
        eps,
        L, D,
        BLOCK_N=min(triton.next_power_of_2(D), 1024)
    )
    return out, out_dense.reshape(original_shape), indices, batch_indices, seq_indices