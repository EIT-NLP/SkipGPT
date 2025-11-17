import os
import torch
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, Any

from ..utils import generate_autotune_config

os.environ['TRITON_PRINT_AUTOTUNING']='0'

@triton.heuristics({
    "HAS_BIAS_UP": lambda args: args['b_up'] is not None,
    "HAS_BIAS_GATE": lambda args: args['b_gate'] is not None,
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
def triton_glu_fwd(
    x: tl.tensor,
    out: tl.tensor,
    w_up: tl.tensor,
    w_gate: tl.tensor,
    b_up: tl.tensor,
    b_gate: tl.tensor,
    M: tl.int64,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS_UP: tl.constexpr,
    HAS_BIAS_GATE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    ACTIVATION: tl.constexpr,
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
    # linear.weight.T is col major
    w_up_ptr = tl.make_block_ptr(
        w_up,
        shape=(K, N),
        strides=(1, K),
        offsets=(0, nid * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )
    w_gate_ptr = tl.make_block_ptr(
        w_gate,
        shape=(K, N),
        strides=(1, K),
        offsets=(0, nid * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # compute (x @ w_up), (x @ w_gate)
    for i in tl.range(0, tl.cdiv(K, BLOCK_K)):
        x_data = tl.load(x_ptr, boundary_check=(0, 1), padding_option='zero')
        w_up_data = tl.load(w_up_ptr, boundary_check=(0, 1), padding_option='zero')
        w_gate_data = tl.load(w_gate_ptr, boundary_check=(0, 1), padding_option='zero')

        acc_up = tl.dot(x_data, w_up_data, acc=acc_up, allow_tf32=ALLOW_TF32)
        acc_gate = tl.dot(x_data, w_gate_data, acc=acc_gate, allow_tf32=ALLOW_TF32)

        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        w_up_ptr = tl.advance(w_up_ptr, (BLOCK_K, 0))
        w_gate_ptr = tl.advance(w_gate_ptr, (BLOCK_K, 0))
    
    if HAS_BIAS_UP:
        b_up_ptr = tl.make_block_ptr(
            b_up,
            shape=(N,),
            strides=(1,),
            offsets=(nid * BLOCK_N,),
            block_shape=(BLOCK_N,),
            order=(0,)
        )
        acc_up += tl.load(b_up_ptr, boundary_check=(0,), padding_option='zero').to(tl.float32)

    if HAS_BIAS_GATE:
        b_gate_ptr = tl.make_block_ptr(
            b_gate,
            shape=(N,),
            strides=(1,),
            offsets=(nid * BLOCK_N,),
            block_shape=(BLOCK_N,),
            order=(0,)
        )
        acc_gate += tl.load(b_gate_ptr, boundary_check=(0,), padding_option='zero').to(tl.float32)
    
    if ACTIVATION == 'silu':
        acc_gate *= tl.sigmoid(acc_gate)
    elif ACTIVATION == 'relu':
        acc_gate = tl.maximum(acc_gate, 0.0)
    
    acc_up *= acc_gate

    out_ptr = tl.make_block_ptr(
        out,
        shape=(M, N),
        strides=(N, 1),
        offsets=(mid * BLOCK_M, nid * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    tl.store(out_ptr, acc_up.to(out.dtype.element_ty), boundary_check=(0, 1))

def triton_glu(
    x: torch.Tensor,
    mlp_up_weight: torch.Tensor,
    mlp_gate_weight: torch.Tensor,
    mlp_up_bias: Optional[torch.Tensor]=None,
    mlp_gate_bias: Optional[torch.Tensor]=None,
    activation: Optional[str]='silu',
) -> torch.Tensor:
    """
    Compute the following equation:

    x = act(x @ w_gate + b_gate) * (x @ w_up + b_up)

    Args:
        x: [total, hidden_size]
        mlp_up_weight: [hidden_size, intermediate_size]
        mlp_gate_weight: [hidden_size, intermediate_size]
        mlp_up_bias: [intermediate_size]
        mlp_gate_bias: [intermediate_size]
        activation: str in ['silu', 'relu']
    Returns:
        out: [total, hidden_size]
    """
    assert mlp_up_weight.shape == mlp_gate_weight.shape

    M, K, N = x.shape[0], x.shape[1], mlp_up_weight.shape[1]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    allow_tf32 = x.dtype == torch.float32
    triton_glu_fwd[grid](
        x,
        out,
        mlp_up_weight,
        mlp_gate_weight,
        mlp_up_bias,
        mlp_gate_bias,
        M, N, K,
        ALLOW_TF32=allow_tf32,
        ACTIVATION=activation,
    )
    return out