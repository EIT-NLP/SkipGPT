import os
import torch

from copy import deepcopy
from argparse import Namespace, ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the finetuned model.pth")
    parser.add_argument("--output_path", type=str, default=None, help="Finetune stage")
    parser.add_argument("--stage", type=str, default='stage1', help="Finetune stage")

    args = parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))
    part = torch.load(
        args.ckpt_path,
        map_location='cpu', weights_only=True
    )

    router_dict = {}
    lora_dict = {}
    for k in part.keys():
        if 'router_attention' in k or 'router_mlp' in k:
            router_dict[k] = deepcopy(part[k])
        elif 'lora_A' in k or 'lora_B' in k:
            lora_dict[k] = deepcopy(part[k])
    
    if args.output_path is None:
        output_path = os.path.join(base_path, f"{args.stage}")
    else:
        output_path = args.output_path
    
    if len(router_dict) > 0: torch.save(router_dict, os.path.join(output_path, f"router.pth"))
    if len(lora_dict) > 0: torch.save(lora_dict, os.path.join(output_path, f"lora.pth"))
