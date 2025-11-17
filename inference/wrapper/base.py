import os
import regex as re
import torch
import torch.nn as nn

from typing import Optional, Tuple, Dict, Any
from argparse import Namespace
from transformers import PreTrainedModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM, PretrainedConfig, PreTrainedTokenizer
from transformers.models import (
    LlamaForCausalLM
)

from .model import *

def apply_wrapper(args: Namespace) -> Tuple[PreTrainedModel, PretrainedConfig, PreTrainedTokenizer]:
    model_path = args.model_path
    router_path = args.router_path
    lora_path = args.lora_path
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = 'flash_attention_2'
    config._forward_impl = args.forward_impl
    config.sparse = args.sparse

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, dtype='auto')

    model = model.eval()

    # load router and lora
    router = None
    lora = None
    if os.path.exists(router_path):
        router = torch.load(router_path, map_location='cpu', weights_only=True)
    if os.path.exists(lora_path):
        lora = torch.load(lora_path, map_location='cpu', weights_only=True)

    if isinstance(model, LlamaForCausalLM):
        model.model = LlamaMoDModel(config, model.model)
    else:
        raise NotImplementedError

    # clean prefix
    if router is not None:
        new_router = {}
        for k, v in router.items():
            name = re.findall(r'[\w\W]*?(model.layers.[\d]+.[\w\W]+)', k)[-1]
            new_router[name] = v

        model.load_state_dict(new_router, strict=False)
    
    if lora is not None:
        new_lora = {}
        for k, v in lora.items():
            name = re.findall(r'[\w\W]*?(model.layers.[\d]+.[\w\W]+)', k)[-1]
            name = name.replace('.block', '')
            component = re.findall(r'[\w\W]*?(lora_A|lora_B)', name)[-1]
            prefix = name.split('.lora')[0]
            if prefix not in new_lora:
                new_lora[prefix] = {}
            new_lora[prefix][component] = v

        param_dict = {}
        for k, v in model.named_parameters(): param_dict[k] = v
        for k, v in new_lora.items():
            if k + '.weight' in param_dict:
                lora_weight = ((v['lora_B'].float() @ v['lora_A'].float()) * (args.lora_alpha / args.lora_r)).to(v['lora_A'].dtype)
                param_dict[k + '.weight'].data.add_(lora_weight)
    
    return model, config, tokenizer