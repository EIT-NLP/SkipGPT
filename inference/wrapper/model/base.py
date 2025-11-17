import torch
import torch.nn as nn

from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from typing import Optional, Dict, List, Tuple, Any
from einops import rearrange

from ...ops import *

_use_top_left_mask = flash_attn_supports_top_left_mask()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor with size [batch_size, seq_len, num_heads, head_dim].
        k (`torch.Tensor`): The key tensor with size [batch_size, seq_len, num_heads, head_dim].
        cos (`torch.Tensor`): The cosine part of the rotary embedding, with size [1, seq_len, head_dim].
        sin (`torch.Tensor`): The sine part of the rotary embedding, with size [1, seq_len, head_dim].
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 直接从输入维度到输出权重预测
        self.weight_predictor = nn.Linear(embed_dim, 2) # 4096->2
        
        # 使用 He Kaiming 初始化
        nn.init.kaiming_uniform_(self.weight_predictor.weight, nonlinearity='linear')
        
        # 初始化 bias 为 0
        if self.weight_predictor.bias is not None:
            nn.init.zeros_(self.weight_predictor.bias)

    def forward(self, x):
        # 保存输入的原始数据类型
        original_type = x.dtype
        
        # 计算权重预测
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype))
        
        return weights.to(original_type)


class AttentionWrapper(nn.Module):
    """Wrapper for token-level routing, with query keep sparse in [nnz, hidden_size] and key keep dense in [batch_size, seq_len, hidden_size]"""
    def __init__(self, config, block, input_layernorm):
        super().__init__()
        self.config = config
        self._forward_impl = getattr(config, '_forward_impl', 'torch')
        self.layer_idx = block.layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.input_layernorm = input_layernorm
        self.q_proj = block.q_proj
        self.k_proj = block.k_proj
        self.v_proj = block.v_proj
        self.o_proj = block.o_proj

        self.q_norm = getattr(block, "q_norm", None)
        self.k_norm = getattr(block, "k_norm", None)

        layer_types = getattr(config, "layer_types", None)
        layer_type = layer_types[self.layer_idx] if layer_types else None
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None
        self.block = block
    
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        sparse_mask: Optional[torch.Tensor]=None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._forward_impl == 'torch':
            query = self.input_layernorm(hidden_states)

            query_states = self.q_proj(query)
            key_states = self.k_proj(query)
            value_states = self.v_proj(query)

            query_states, key_states, value_states = list(map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), [query_states, key_states, value_states]))

            if self.q_norm: query_states = self.q_norm(query_states)
            if self.k_norm: key_states = self.k_norm(key_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length=query_states.shape[1],
                is_causal=self.is_causal,
                dropout=0.0,
                softmax_scale=self.scaling,
                sliding_window=self.sliding_window,
                use_top_left_mask=_use_top_left_mask,
                attn_implementation=self.config._attn_implementation,
                layer_idx=self.layer_idx,
                **kwargs,
            )
            attn_output = rearrange(attn_output, '... h d -> ... (h d)').contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output
        elif self._forward_impl == 'triton':
            query, key, indices, batch_ids, pos_ids = triton_sparse_rmsnorm_before_attn(
                x=hidden_states,
                mask=sparse_mask,
                w=self.input_layernorm.weight,
                eps=self.input_layernorm.variance_epsilon,
            )

            query_states = self.q_proj(query)
            key_states = self.k_proj(key)
            value_states = self.v_proj(key)

            query_states, key_states, value_states = list(map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), [query_states, key_states, value_states]))

            if self.q_norm: query_states = triton_rmsnorm(query_states, self.q_norm.weight, self.q_norm.variance_epsilon)
            if self.k_norm: key_states = triton_rmsnorm(key_states, self.k_norm.weight, self.k_norm.variance_epsilon)

            cos, sin = position_embeddings
            query_states, key_states = triton_rope_qk_align(query_states, key_states, cos, sin, batch_ids, pos_ids)

            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
            attn_output = query_sparse_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                sparse_mask,
                indices=batch_ids if sparse_mask.shape[1] == 1 and sparse_mask.shape[1] != attention_mask.shape[1] else pos_ids,
            )
            attn_output = rearrange(attn_output, '... h d -> ... (h d)')
            attn_output = triton_sparse_mlp(attn_output, indices, self.o_proj.weight.T, self.o_proj.bias, sparse_mask.shape[0] * sparse_mask.shape[1])
            return rearrange(attn_output, '(b l) ... -> b l ...', b=sparse_mask.shape[0], l=sparse_mask.shape[1])
        
        else: raise NotImplementedError


class MLPWrapper(nn.Module):
    """Wrapper for token-level routing, with input keep sparse in [nnz, hidden_size]"""
    def __init__(self, config, block, post_attention_layernorm):
        super().__init__()
        self.config = config
        self._forward_impl = getattr(config, '_forward_impl', 'torch')
        self.post_attention_layernorm = post_attention_layernorm
        self.gate_proj = block.gate_proj
        self.up_proj = block.up_proj
        self.down_proj = block.down_proj
        assert config.hidden_act in ['silu', 'relu'] # do not support other activations yet
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        sparse_mask: Optional[torch.Tensor]=None,
    ):
        if self._forward_impl == 'torch':
            out = self.post_attention_layernorm(hidden_states)
            return self.down_proj(self.act_fn(self.gate_proj(out)) * self.up_proj(out))
        elif self._forward_impl == 'triton':
            out, indices = triton_sparse_rmsnorm(
                x=hidden_states.flatten(0, -2),
                mask=sparse_mask.flatten(),
                w=self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
            )
            out = triton_glu(
                x=out,
                mlp_up_weight=self.up_proj.weight.T,
                mlp_gate_weight=self.gate_proj.weight.T,
                mlp_up_bias=self.up_proj.bias,
                mlp_gate_bias=self.gate_proj.bias,
                activation='silu',
            )
            out = triton_sparse_mlp(
                x=out,
                indices=indices,
                mlp_down_weight=self.down_proj.weight.T,
                mlp_down_bias=self.down_proj.bias,
                max_length=sparse_mask.numel(),
            )
            return rearrange(out, '(b l) ... -> b l ...', b=sparse_mask.shape[0])
        else:
            raise NotImplementedError