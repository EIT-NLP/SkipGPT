import os
import triton
import torch
import torch.nn as nn

from functools import partial
from triton.testing import do_bench
from typing import Optional, Tuple, Dict, List, Any

from .wrapper.model.cache import DynamicCacheSeqFirst

def set_kv_cache(
    config,
    dummy_k: Optional[torch.Tensor]=None,
    dummy_v: Optional[torch.Tensor]=None,
    append_layer: Optional[int]=None,
) -> DynamicCacheSeqFirst:
    cache = DynamicCacheSeqFirst(config=config)

    if dummy_k is not None:
        _k = dummy_k.clone().detach()
        _v = dummy_v.clone().detach()

        _, _ = cache.update(_k, _v, layer_idx=append_layer)
    return cache

def inference_benchmark(
    model,
    config,
    batch_size: Optional[int]=32,
    sparsity: Optional[float]=0.25,
    mode: Optional[str]='prefill'
):
    @triton.testing.perf_report((
        triton.testing.Benchmark(
            x_names=['seqlen'],
            x_vals=[2 ** i for i in range(3, 15)],
            x_log=True,
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
            line_vals=['dense', 'torch-sparse', 'triton-sparse'],  # Possible values for `line_arg`.
            line_names=['dense', 'torch-sparse', 'triton-sparse'],  # Label name for the lines.
            styles=[('red', '-'), ('coral', '-'), ('violet', '-')],  # Line styles.
            ylabel='TFLOPS',  # Label name for the y-axis.
            plot_name=f'inference-benchmark-bsz{batch_size}-sparsity{int(sparsity*100)}-{mode}',  # Name for the plot. Used also as a file name for saving the plot.
            args={'bsz': batch_size, 'sparsity': sparsity, 'mode': mode},  # Values for function arguments not in `x_names` and `y_name`.
        )
    ))
    def benchmark(bsz, seqlen, sparsity, mode, provider):
        device = 'cuda'
        dtype = torch.bfloat16
        hidden_size = config.hidden_size

        seqlen_q = seqlen if mode == 'prefill' else 1
        seqlen_k = seqlen

        attn_route = torch.rand((bsz, seqlen_q), device=device) < sparsity
        mlp_route = torch.rand((bsz, seqlen_q), device=device) < sparsity

        dummy_input = torch.randint(low=0, high=4096, size=(bsz, seqlen_q), device=device)
        inputs_embeds = model.model.embed_tokens(dummy_input)

        attention_mask = torch.ones((bsz, seqlen_k), dtype=torch.bool, device=device)

        dummy_k = torch.randn((bsz, seqlen_k-1, config.num_key_value_heads, config.head_dim), dtype=dtype, device=device) if seqlen_q == 1 else None
        dummy_v = torch.randn((bsz, seqlen_k-1, config.num_key_value_heads, config.head_dim), dtype=dtype, device=device) if seqlen_q == 1 else None

        past_key_values = set_kv_cache(config, dummy_k, dummy_v, 0)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        if provider == 'dense':
            model.model.layers[0]._forward_impl = 'torch'
            model.model.layers[0].config.sparse = False
            model.model.layers[0].self_attn._forward_impl = 'torch'
            model.model.layers[0].mlp._forward_impl = 'torch'
        elif provider == 'torch-sparse':
            model.model.layers[0]._forward_impl = 'torch'
            model.model.layers[0].config.sparse = True
            model.model.layers[0].self_attn._forward_impl = 'torch'
            model.model.layers[0].mlp._forward_impl = 'torch'
        elif provider == 'triton-sparse':
            model.model.layers[0]._forward_impl = 'triton'
            model.model.layers[0].config.sparse = True
            model.model.layers[0].self_attn._forward_impl = 'triton'
            model.model.layers[0].mlp._forward_impl = 'triton'
        else:
            raise ValueError(f"Unknown provider {provider}")

        test_fun = partial(
            model.model.layers[0].forward,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            attn_route=attn_route,
            mlp_route=mlp_route,
            is_benchmark=True,
        )

        quantiles = [0.5, 0.2, 0.8]
        try:
            with torch.no_grad():
                ms, min_ms, max_ms = do_bench(test_fun, quantiles=quantiles)
                print(f"pass {provider} in [{bsz}, {seqlen}] with sparsity {sparsity}, {mode} mode")
        except Exception as e:
            return 0, 0, 0

        tflops = lambda ms: (
            (bsz * seqlen_q * hidden_size * (2 * config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim) + # attn qkvo proj
            (bsz * config.head_dim * config.num_attention_heads * seqlen_q * seqlen_k) + # attn
            (bsz * seqlen_q * hidden_size * config.intermediate_size * 3 + bsz * seqlen_q * config.intermediate_size) # glu
        ) * 2 * 1e-12 / (ms * 1e-3)

        return tflops(ms), tflops(min_ms), tflops(max_ms)
    return benchmark