import os
import torch
import torch.nn as nn

from copy import deepcopy
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.llama import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
)
from transformers.utils import auto_docstring
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Tuple, Dict, List, Any

from .cache import DynamicCacheSeqFirst
from .base import TokenRouter, AttentionWrapper, MLPWrapper
from ...ops import triton_rmsnorm

class LlamaMoDDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        block: LlamaDecoderLayer,
    ):
        super().__init__()
        self.config = config
        self._forward_impl = getattr(config, '_forward_impl', 'torch')
        self.hidden_size = config.hidden_size
        self.self_attn = AttentionWrapper(config, block.self_attn, block.input_layernorm)
        self.mlp = MLPWrapper(config, block.mlp, block.post_attention_layernorm)

        self.router_attention = TokenRouter(self.hidden_size)
        self.router_mlp = TokenRouter(self.hidden_size)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> torch.Tensor:
        is_benchmark = kwargs.get("is_benchmark", False)
        # Routing
        if self.config.sparse:
            attn_route = self.router_attention(hidden_states)
            mlp_route = self.router_mlp(hidden_states)

            if attn_route in kwargs:
                attn_route = kwargs.get("attn_route")
            if mlp_route in kwargs:
                mlp_route = kwargs.get("mlp_route")

            route_mask_attn = attn_route.argmax(dim=-1).to(torch.bool)
            route_mask_mlp = mlp_route.argmax(dim=-1).to(torch.bool)

            query_mask_attn = route_mask_attn.logical_not()
            query_mask_mlp = route_mask_mlp.logical_not()
            if hidden_states.shape[1] == attention_mask.shape[1]:
                query_mask_attn = query_mask_attn.logical_and(attention_mask)

            residual = hidden_states
            # Self Attention
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                sparse_mask=query_mask_attn,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values if not is_benchmark else deepcopy(past_key_values),
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states.masked_fill_(route_mask_attn.unsqueeze(-1), 0.0)
            hidden_states += residual

            # Fully Connected
            residual = hidden_states
            if query_mask_mlp.eq(1).any():
                if self._forward_impl == 'torch':
                    hidden_states[query_mask_mlp] += self.mlp(hidden_states[query_mask_mlp])
                elif self._forward_impl == 'triton':
                    hidden_states = self.mlp(hidden_states, query_mask_mlp)
                    hidden_states += residual
            else:
                hidden_states = residual
        else:
            query_mask_attn = torch.ones(*hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
            residual = hidden_states
            # Self Attention
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                sparse_mask=query_mask_attn,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values if not is_benchmark else deepcopy(past_key_values),
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states += residual

            residual = hidden_states
            hidden_states = self.mlp(hidden_states)
            hidden_states += residual

        return hidden_states

@auto_docstring
class LlamaMoDPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaMoDDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False

    _can_record_outputs = {
        "hidden_states": LlamaMoDDecoderLayer,
        "attentions": AttentionWrapper,
    }

class LlamaMoDModel(LlamaMoDPreTrainedModel):
    def __init__(
        self,
        config: LlamaConfig,
        block: LlamaModel,
    ):
        super().__init__(config)
        assert config._attn_implementation in ['flash_attention_2', 'flash_attention_3']

        self.config = config
        self._forward_impl = getattr(config, '_forward_impl', 'torch')
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = block.embed_tokens
        self.layers = nn.ModuleList(
            [LlamaMoDDecoderLayer(config, blk) for blk in block.layers]
        )
        self.norm = block.norm
        self.rotary_emb = block.rotary_emb
        self.gradient_checkpointing = block.gradient_checkpointing

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, DynamicCacheSeqFirst):
            past_key_values = DynamicCacheSeqFirst(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        if attention_mask is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1] + past_seen_tokens),
                device=inputs_embeds.device,
                dtype=torch.bool,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if self._forward_impl == 'triton':
            hidden_states = triton_rmsnorm(hidden_states, self.norm.weight, self.norm.variance_epsilon)
        else:
            hidden_states = self.norm(hidden_states)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )