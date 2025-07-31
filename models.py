from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from router_attn_mlp import apply_router_attn_mlp
import torch
def model_import(device, args):
    # 如果是评估模式（不做训练）
    if args.eval:
        # 如果是加载已经微调过的模型（通过torch.load加载.pt或.pth）
        if args.finetuned:
            orignal_model = torch.load(args.model_path).to(device)  # 直接加载保存的模型对象
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # 加载预训练分词器（从指定模型名）
            tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token，防止缺失 pad_token 报错
        else:
            # 加载预训练模型（从 huggingface 结构目录）
            orignal_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,              # 使用 bfloat16 精度减少显存
                attn_implementation="flash_attention_2",  # 使用更快的注意力机制
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            tokenizer.pad_token = tokenizer.eos_token
        return orignal_model, tokenizer  # 返回模型和分词器

    else:
        # 非评估模式，即训练模式下加载模型
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # 根据指定方法对模型结构进行修改（如插入 Router 或其他模块）
        if args.method == "router_attn_mlp":
            model = apply_router_attn_mlp(model, args).to(device)

        elif args.method == "router_all":
            model = apply_router_all(model, args).to(device)

        elif args.method == "mod_twice":
            model = apply_mod_twice(model, args).to(device)

        elif args.method == "mod":
            model = apply_mod(model, args).to(device)

        # “joint” 方法优先级高于上面几个（可看作一种特殊组合）
        if args.method == "joint":
            model = apply_router_attn_mlp(model, args).to(device)

        # 加载已经训练好的模型参数（后训练模式）
        elif args.method == "post_training":
            model = torch.load(
                "../mod/Llama-2-7b_router_attn_mlp_0_loraFalse/checkpoint-3200/model.pth"
            ).to(device)

        # 使用 DeepSpeed 下的训练模型（load_state_dict 且允许部分不匹配）
        elif args.method == "post_training_deepspeed":
            model = apply_router_attn_mlp(model, args).to(device)
            model.load_state_dict(
                torch.load("../anhao_zhao/mod/Llama-2-7b_router_attn_mlp_0_loraFalse/checkpoint-3200/model.pth"),
                strict=False  # 允许权重键值不完全匹配
            )

        # 加载 tokenizer 并设置 pad_token
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer  # 返回修改后的模型和分词器
