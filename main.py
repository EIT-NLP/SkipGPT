import torch
from models import model_import
from train_processing_two_stages import train_processing_two_stages
import argparse
import os
import random
import numpy as np  
from transformers import set_seed
def parse_args():
    parser = argparse.ArgumentParser(description="dynamic layer skipping.")
    
    # 添加命令行参数
    parser.add_argument("--lora", action="store_true", help="Use LoRA") # 使用 LoRA
    parser.add_argument("--sparsity", type=int, default=0.2, help="Sparsity of the routers") 
    parser.add_argument("--capacity", type=float, default=0.75, help="Capacity of the routers") # 路由器的容量，只与mod的复现有关
    parser.add_argument("--model_name", type=str, default="Llama-2-13b", help="Name of the model to load") # 模型名称
    parser.add_argument("--model_path", type=str, default="Llama-2-13b", help="Path to the model to load") # 模型路径
    parser.add_argument("--finetuned", action="store_true", help="Load a finetuned model") # 是否加载微调后的模型
    parser.add_argument("--method", type=str, default="router_attn_mlp", choices=['router_attn_mlp', 'mod_twice', 'mod', "router_all","post_training","original","joint","post_training_deepspeed"], help="choosed method") # 动态层跳过的方法
    parser.add_argument("--dataset_path", type=str, default="datasets/RedPajama-Data-1T-Sample", help="Path to the dataset") # 数据集路径
    parser.add_argument("--initial_temperature", type=float, default=5.0, help="Initial temperature for the gumbel softmax") # gumbe softmax的初始温度
    parser.add_argument("--final_temperature", type=float, default=1.0, help="Final temperature for the gumbel softmax") # gumbe softmax的最终温度
    parser.add_argument("--warmup_ratio", type=float, default=0, help="Warmup ratio for the temperature") # 温度的预热比例
    parser.add_argument("--max_steps_stage", type=int, default=10000, help="Maximum training steps") # 最大训练步数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--bf16", type=bool, default=True, help="Use mixed precision training") # 使用混合精度训练
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy") # 评估策略
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation steps") # 评估步数
    parser.add_argument("--save_strategy", type=str, default="no", help="Save strategy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps") # 梯度累积步数
    parser.add_argument("--logging_strategy", type=str, default="steps", help="Logging strategy")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--eval", action="store_true", help="Train or evaluate the model") # 训练或评估模型
    parser.add_argument("--eval_dataset_ratio", type=float, default=0.001, help="Ratio of the dataset to use for evaluation") # 用于评估的数据集比例
    parser.add_argument("--max_length", type=int, default=4096, help="the max context length") # 最大上下文长度
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from distributed launcher")
    parser.add_argument("--train_method", type=str, default="target_sparsity", choices=["target_sparsity", "curriculum"], help="Training method") # 训练方法
    args = parser.parse_args()
    return args

def wandb_config():
    # 设置你自己的WANDB_API_KEY
    os.environ['WANDB_API_KEY'] = "personal_wandb_id" 

    # 设置 WANDB_CONFIG_DIR 以使用自定义的配置目录
    os.environ['WANDB_CONFIG_DIR'] = './wandb_config'

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU 和 GPU 上的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)               # 为当前GPU设置种子
        torch.cuda.manual_seed_all(seed)           # 如果使用多个GPU，设置所有GPU的种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

def main():
    args = parse_args()
    
    # 设置wandb
    wandb_config()

    # 设置种子
    set_seeds(43)
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 导入模型
    if args.eval:
        orignal_model, tokenizer = model_import(device, args)
        args.max_length=orignal_model.config.max_position_embeddings
        
    else:
        model, tokenizer = model_import(device, args)
        # args.max_length=model.config.max_position_embeddings

    print(f"Maximum context size: {args.max_length}")


    if args.eval:
        train_processing_two_stages(args, tokenizer, orignal_model)

    else:
        train_processing_two_stages(args, tokenizer, model)

if __name__ == "__main__":
    main()