import os
import torch
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, Any

from ..utils import generate_autotune_config

os.environ['TRITON_PRINT_AUTOTUNING']='0'

@triton.jit
def triton_prepare_chunk_indices(
    chunk_num: tl.tensor,
    chunk_cumsum: tl.tensor,
    out: tl.tensor,
):
    bid = tl.program_id(0)

    num = tl.load(chunk_num + bid)
    prev_count = tl.load(chunk_cumsum + bid - 1, mask=bid > 0, other=0)

    for i in tl.range(num):
        tl.store(out + (prev_count + i) * 2 + 1, i)
        tl.store(out + (prev_count + i) * 2, bid)

def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    impl: Optional[str]='triton',
):
    """
    Generate [cdiv(seqlen, chunk_size), 2] indices tensor for chunk id in batch (col 1) and batch id (col 0).

    From Flash-Linear-Attention.
    """
    if impl == 'torch':
        chunk_ids = torch.cat([torch.arange(_) for _ in triton.cdiv(cu_seqlens[1:] - cu_seqlens[:-1], chunk_size).tolist()])
        return torch.stack([chunk_ids.eq(0).cumsum(0) - 1, chunk_ids], 1).to(cu_seqlens)
    elif impl == 'triton':
        chunk_num = triton.cdiv(cu_seqlens[1:] - cu_seqlens[:-1], chunk_size)
        chunk_cumsum = chunk_num.cumsum(0)
        M = chunk_cumsum[-1].item()
        B = cu_seqlens.shape[0] - 1

        out = torch.empty([M, 2], dtype=torch.int32, device=cu_seqlens.device)
        grid = lambda META: (B,)
        triton_prepare_chunk_indices[grid](
            chunk_num,
            chunk_cumsum,
            out,
        )
        return out

@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_N=[32, 64, 128],
            num_stages=[2, 3, 4]
        )
    ),
    key=['LK', 'HQ', 'D', 'BLOCK_M'],
)
@triton.jit
def prefill_query_sparse_attn_fwd(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    cu_seqlens_q: tl.tensor,
    pos_indices: tl.tensor,
    pad_offset: tl.tensor,
    chunk_indices: tl.tensor,
    out: tl.tensor,
    LK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    qk_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    hid, cid = tl.program_id(1), tl.program_id(0)

    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    qk_scale *= 1.44269504 # 1/log(2)
    hid_kv = hid // G

    batch_id = tl.load(chunk_indices + cid * 2)
    chunk_id = tl.load(chunk_indices + cid * 2 + 1)

    start_q, end_q = tl.load(cu_seqlens_q + batch_id), tl.load(cu_seqlens_q + batch_id + 1)
    seq_offset_q = start_q + chunk_id * BLOCK_M

    pad_offset_k = tl.load(pad_offset + batch_id)
    start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
    key_length = end_k - start_k

    pos_idx_q = tl.load(pos_indices + seq_offset_q + tl.arange(0, BLOCK_M), mask=(seq_offset_q + tl.arange(0, BLOCK_M)) < end_q, other=-1)
    max_pos_q = tl.max(pos_idx_q)

    query_range_mask = (seq_offset_q + tl.arange(0, BLOCK_M)) < end_q
    query_data = tl.load(
        q + ((seq_offset_q + tl.arange(0, BLOCK_M)) * HQ * D + hid * D)[:, None] + tl.arange(0, D)[None, :],
        mask=query_range_mask[:, None],
        other=0.0
    )

    split_k_range = tl.cdiv(max_pos_q - pad_offset_k + 1, BLOCK_N)
    for tile_k in tl.range(0, split_k_range):
        key_range_mask = ((tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
        key_data = tl.load(
            k + ((start_k + tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + hid_kv * D)[:, None] + tl.arange(0, D)[None, :],
            mask=key_range_mask[:, None],
            other=0.0
        )
        value_data = tl.load(
            v + ((start_k + tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + hid_kv * D)[:, None] + tl.arange(0, D)[None, :],
            mask=key_range_mask[:, None],
            other=0.0
        )

        qk = tl.dot(query_data, key_data.T) * qk_scale
        
        # causal mask for each query pos
        causal_mask = pos_idx_q[:, None] >= (tile_k * BLOCK_N + pad_offset_k + tl.arange(0, BLOCK_N))[None, :]
        qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

        score_max_new = tl.maximum(score_max, tl.max(qk, 1))
        score_scale = tl.exp2(score_max - score_max_new)
        qk = tl.exp2(qk - score_max_new[:, None])
        score_sum = score_sum * score_scale + tl.sum(qk, 1)
        acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
        score_max = score_max_new
    
    acc /= score_sum[:, None]
    tl.store(
        out + ((seq_offset_q + tl.arange(0, BLOCK_M)) * HQ * D + hid * D)[:, None] + tl.arange(0, D)[None, :],
        acc.to(out.dtype.element_ty),
        mask=query_range_mask[:, None]
    )

@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_M=[16, 32],
            BLOCK_N=[32, 64, 128],
            num_stages=[2, 3, 4]
        )
    ),
    key=['HQ', 'D'],
)
@triton.jit(do_not_specialize=['LK'])
def decode_query_sparse_attn_fwd(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    batch_indices: tl.tensor,
    pad_offset: tl.tensor,
    out: tl.tensor,
    LK: tl.int64,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    qk_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    hid, cid = tl.program_id(1), tl.program_id(0)

    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    qk_scale *= 1.44269504 # 1/log(2)
    hid_kv = hid // G

    batch_id = tl.load(batch_indices + cid)

    start_q, end_q = cid, cid + 1
    seq_offset_q = start_q

    pad_offset_k = tl.load(pad_offset + batch_id)
    start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
    key_length = end_k - start_k

    query_range_mask = (seq_offset_q + tl.arange(0, BLOCK_M)) < end_q
    query_data = tl.load(
        q + ((seq_offset_q + tl.arange(0, BLOCK_M)) * HQ * D + hid * D)[:, None] + tl.arange(0, D)[None, :],
        mask=query_range_mask[:, None],
        other=0.0
    )

    split_k_range = tl.cdiv(key_length, BLOCK_N)
    for tile_k in tl.range(0, split_k_range):
        key_range_mask = ((tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
        key_data = tl.load(
            k + ((start_k + tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + hid_kv * D)[:, None] + tl.arange(0, D)[None, :],
            mask=key_range_mask[:, None],
            other=0.0
        )
        value_data = tl.load(
            v + ((start_k + tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + hid_kv * D)[:, None] + tl.arange(0, D)[None, :],
            mask=key_range_mask[:, None],
            other=0.0
        )

        qk = tl.dot(query_data, key_data.T) * qk_scale
        qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))

        score_max_new = tl.maximum(score_max, tl.max(qk, 1))
        score_scale = tl.exp2(score_max - score_max_new)
        qk = tl.exp2(qk - score_max_new[:, None])
        score_sum = score_sum * score_scale + tl.sum(qk, 1)
        acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
        score_max = score_max_new
    
    acc /= score_sum[:, None]
    tl.store(
        out + ((seq_offset_q + tl.arange(0, BLOCK_M)) * HQ * D + hid * D)[:, None] + tl.arange(0, D)[None, :],
        acc.to(out.dtype.element_ty),
        mask=query_range_mask[:, None]
    )

@triton.autotune(
    configs=generate_autotune_config(
        dict(
            BLOCK_N=[16, 32, 64, 128],
            num_stages=[2, 3, 4]
        )
    ),
    key=['HQ', 'D'],
)
@triton.jit(do_not_specialize=['LK'])
def decode_query_sparse_attn_fwd_gemv(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    batch_indices: tl.tensor,
    pad_offset: tl.tensor,
    out: tl.tensor,
    LK: tl.int64,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    qk_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    hid, cid = tl.program_id(1), tl.program_id(0)

    score_max = -float('inf')
    score_sum = float(0)
    acc = tl.zeros([D], dtype=tl.float32)

    qk_scale *= 1.44269504 # 1/log(2)
    hid_kv = hid // G

    batch_id = tl.load(batch_indices + cid)

    start_q, end_q = cid, cid + 1
    seq_offset_q = start_q

    pad_offset_k = tl.load(pad_offset + batch_id)
    start_k, end_k = batch_id * LK + pad_offset_k, (batch_id + 1) * LK
    key_length = end_k - start_k

    query_data = tl.load(q + (seq_offset_q * HQ * D + hid * D) + tl.arange(0, D))[None, :]

    split_k_range = tl.cdiv(key_length, BLOCK_N)
    for tile_k in tl.range(0, split_k_range):
        key_range_mask = ((tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
        key_data = tl.load(
            k + ((start_k + tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + hid_kv * D)[:, None] + tl.arange(0, D)[None, :],
            mask=key_range_mask[:, None],
            other=0.0
        )
        value_data = tl.load(
            v + ((start_k + tile_k * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + hid_kv * D)[:, None] + tl.arange(0, D)[None, :],
            mask=key_range_mask[:, None],
            other=0.0
        )

        # qk size [BLOCK_N]
        qk = tl.sum(query_data * key_data, axis=-1, dtype=tl.float32) * qk_scale
        qk = tl.where(key_range_mask, qk, -float('inf'))

        # scalar
        score_max_new = tl.maximum(score_max, tl.max(qk))
        score_scale = tl.exp2(score_max - score_max_new)
        qk = tl.exp2(qk - score_max_new)
        score_sum = score_sum * score_scale + tl.sum(qk)
        acc = acc * score_scale + tl.sum(qk.to(q.dtype.element_ty) * value_data.T, axis=-1, dtype=tl.float32)
        score_max = score_max_new
    
    acc /= score_sum
    tl.store(
        out + (seq_offset_q * HQ * D + hid * D) + tl.arange(0, D),
        acc.to(out.dtype.element_ty)
    )

def prefill_query_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    pos_indices: torch.Tensor,
    pad_offset: torch.Tensor,
    chunk_size: Optional[int]=32,
    chunk_indices_impl: Optional[str]='triton',
):
    """
    An implementation of query sparse attention (causal).

    The main pipeline is from FLA's prefill flash-attn kernel.

    Args:
        q: torch.Tensor with shape [nnz_q, num_heads_q, head_dim]
        k: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
        v: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
        cu_seqlens_q: torch.Tensor with shape [batch_size + 1,]
        pos_indices: torch.Tensor with shape [nnz_q,], the exact position id of each query token
        pad_offset: torch.Tensor with shape [batch_size,], the left padding offset for key and value
        chunk_size: int, the BLOCK_M for query chunking
    Returns:
        out: torch.Tensor with shape [nnz_q, num_heads, head_dim]
    """

    NQ, HQ, D = q.shape

    B, LK, HK, _ = k.shape
    G = HQ // HK

    assert cu_seqlens_q[-1] == NQ
    chunk_indices = prepare_chunk_indices(cu_seqlens_q, chunk_size, chunk_indices_impl)

    C = chunk_indices.shape[0]
    out = torch.empty_like(q)
    grid = lambda META: (C, HQ)
    prefill_query_sparse_attn_fwd[grid](
        q, k, v,
        cu_seqlens_q,
        pos_indices,
        pad_offset,
        chunk_indices,
        out,
        LK, HQ, HK, D, G,
        qk_scale=D**-0.5,
        BLOCK_M=chunk_size
    )
    return out

def decode_query_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_indices: torch.Tensor,
    pad_offset: torch.Tensor,
    use_tensor_core: Optional[bool]=False
):
    """
    An implementation of query sparse attention (causal).

    The main pipeline is from FLA's prefill flash-attn kernel.

    Args:
        q: torch.Tensor with shape [nnz_q, num_heads_q, head_dim] (equal to [valid_batch_size, num_heads_q, head_dim])
        k: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
        v: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
        batch_indices: torch.Tensor with shape [nnz_q,]
        pad_offset: torch.Tensor with shape [batch_size,], the left padding offset for key and value
        chunk_size: int, the BLOCK_M for query chunking
        use_tensor_core: bool, whether to use tensor core (padding gemm) or not (gemv)
    Returns:
        out: torch.Tensor with shape [nnz_q, num_heads, head_dim]
    """

    C, HQ, D = q.shape
    assert k.dim() == 4

    B, LK, HK, _ = k.shape
    G = HQ // HK

    out = torch.empty_like(q)
    grid = lambda META: (C, HQ)

    if use_tensor_core:
        decode_query_sparse_attn_fwd[grid](
            q, k, v,
            batch_indices,
            pad_offset,
            out,
            LK, HQ, HK, D, G,
            qk_scale=D**-0.5,
        )
    else:
        decode_query_sparse_attn_fwd_gemv[grid](
            q, k, v,
            batch_indices,
            pad_offset,
            out,
            LK, HQ, HK, D, G,
            qk_scale=D**-0.5,
        )
    return out

def query_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pad_mask: torch.Tensor,
    query_mask: torch.Tensor,
    indices: Optional[torch.Tensor]=None,
    chunk_size: Optional[int]=32,
    chunk_indices_impl: Optional[str]='triton',
):
    """
    Apply causal query sparse attention with dense kv for both prefill and decode

    Args:
        q: torch.Tensor with shape [nnz_q, num_heads_q, head_dim]
        k: torch.Tensor with shape [batch_size, seqlen_k, num_heads_k, head_dim]
        v: torch.Tensor with shape [batch_size, seqlen_k, num_heads_k, head_dim]
        pad_mask: torch.Tensor with shape [batch_size, seqlen_k]
        query_mask: torch.Tensor with shape [batch_size, seqlen_q]
        indices: torch.Tensor with shape [nnz_q,], the exact position id of each query token
        chunk_size: int, the BLOCK_M for query chunking
        chunk_indices_impl: str, the implementation of chunk indices calculation
    Returns:
        out: torch.Tensor with shape [nnz_q, num_heads, head_dim]
    """
    assert pad_mask.shape == k.shape[:2]
    pad_offset = k.shape[1] - pad_mask.sum(-1)

    if query_mask.shape[1] == 1 and query_mask.shape[1] != pad_mask.shape[1]: # decode
        if indices is None: indices = torch.nonzero(query_mask, as_tuple=True)[0]
        out = decode_query_sparse_attn(
            q, k, v,
            indices,
            pad_offset,
        )
    
    else: # prefill
        seqlens_q = query_mask.sum(-1)
        cu_seqlens_q = torch.nn.functional.pad(seqlens_q.cumsum(-1), (1, 0), mode='constant', value=0)
        if indices is None: indices = torch.nonzero(query_mask, as_tuple=True)[-1]
        out = prefill_query_sparse_attn(
            q, k, v,
            cu_seqlens_q,
            indices,
            pad_offset,
            chunk_size=chunk_size,
            chunk_indices_impl=chunk_indices_impl,
        )
    
    return out