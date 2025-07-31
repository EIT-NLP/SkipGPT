from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from log_sparsity import SparsityLoggingCallback, SparsityLoggingAllCallback, CustomWandbCallback
from custom_trainer import CustomTrainer_router_attn_mlp, CustomTrainer_router_all
from torch_save import SaveCheckpointCallback, CustomEvalCallback, SaveCheckpointCallback_post_training, SaveCheckpointCallback_original
import math
import torch
import wandb
import os
from transformers import AdamW
from dataset_processing import dataset_processing, dataset_processing_llama13

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
def train_processing_two_stages(args, tokenizer, model):
  
    if args.model_path == "Llama-2-13b":
        # Load the dataset
        train_dataset, valid_dataset = dataset_processing_llama13(args, tokenizer)
    else:
        # Load the dataset
        train_dataset, valid_dataset = dataset_processing(args, tokenizer)

    # Initialize the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

    training_args = TrainingArguments(
    output_dir= f"dynamic_layer_skipping_{args.model_name}",
    evaluation_strategy= args.evaluation_strategy,
    eval_steps= args.eval_steps,  # 每100步评估一次，可以根据需要调整
    save_strategy= args.save_strategy,
    learning_rate= args.learning_rate, 
    lr_scheduler_type= args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    max_steps= args.max_steps_stage,
    weight_decay= args.weight_decay,
    bf16= args.bf16,
    per_device_train_batch_size= args.per_device_train_batch_size,
    per_device_eval_batch_size= args.per_device_eval_batch_size,
    gradient_accumulation_steps= args.gradient_accumulation_steps,
    max_grad_norm= args.max_grad_norm,
    logging_strategy= args.logging_strategy,
    logging_steps= args.logging_steps,
    logging_dir=f'./logs/{args.method}_{args.sparsity}_{args.model_path}_{args.learning_rate}_{args.lr_scheduler_type}',
    seed=43,
    report_to="wandb",
    # local_rank = 0,
    deepspeed= "deepspeed_config.json",
    )
    
    # 函数用于冻结模型的一些参数
    def freeze_parameters(model):
        for name, param in model.named_parameters():
            if 'router' not in name:  # 这里根据参数名称选择需要冻结的部分
                param.requires_grad = False

    
    # 解冻所有参数
    def unfreeze_parameters(model):
        for param in model.parameters():
            param.requires_grad = True

    # 冻结一些参数+lora
    def unfreeze_parameters_lora(model):
        # for name, param in model.named_parameters():
        #     if 'router' not in name:  # 这里根据参数名称选择需要冻结的部分
        #         param.requires_grad = False
        lora_config = LoraConfig(
            r=16,  # 低秩矩阵的秩
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # 指定应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        # 再次确保 router 参数的 requires_grad 为 True
        for name, param in model.named_parameters():
            if 'router' in name:
                param.requires_grad = True

        return model
    
    # lora
    def lora(model):
        lora_config = LoraConfig(
            r=16,  # 低秩矩阵的秩
            lora_alpha=32,
            target_modules=["q_proj", "v_proj","gate_proj"],  # 指定应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        return model


    if args.eval== False:
        # Initialize wandb for the second run
        wandb.init(project=f"dynamic_layer_skipping_{args.model_name}_{args.method}") 
        # Log the hyperparameters
        wandb.config.update(vars(args))
        if args.method=="router_attn_mlp" and args.lora==False:
            # Unfreeze the parameters
            freeze_parameters(model)
        elif args.method=="router_attn_mlp" and args.lora==True:
            # Unfreeze the parameters
            model=unfreeze_parameters_lora(model)

        elif args.method=="mod_twice" or args.method=="mod":
            # Unfreeze the parameters
            model=unfreeze_parameters_lora(model)
        
        elif "post_training" in args.method:
            model = lora(model)
        
        elif args.method == "router_all" and args.lora==False:
            # Unfreeze the parameters
            freeze_parameters(model)

        elif args.method == "router_all" and args.lora==True:
            # Unfreeze the parameters
            model= unfreeze_parameters_lora(model)
        
        elif args.method == "original" and args.lora==True:
            model= lora(model)

        elif args.method == "joint":
            model=unfreeze_parameters_lora(model)



        
        if args.method=="router_attn_mlp":
            # Initialize the trainer for the second stage
            trainer = CustomTrainer_router_attn_mlp(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                custom_args=args,  # 传递自定义的 args
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[SparsityLoggingCallback(f'./logs/{args.method}_{args.sparsity}_{args.model_path}_{args.learning_rate}_{args.lr_scheduler_type}'),
                SaveCheckpointCallback(
                save_steps=200,  # 每1000步保存一次，可以根据需要修改
                model_name=args.model_path,
                method=args.method,
                sparsity=args.sparsity,
                lora=args.lora
            )],
            )
        
        elif args.method=="joint":
            # Initialize the trainer for the second stage
            trainer = CustomTrainer_router_attn_mlp(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[SparsityLoggingCallback(f'./logs/{args.method}_{args.sparsity}_{args.model_path}_{args.learning_rate}_{args.lr_scheduler_type}'),SaveCheckpointCallback(
                save_steps=200,  # 每1000步保存一次，可以根据需要修改
                model_name=args.model_path,
                method=args.method,
                sparsity=args.sparsity,
                lora=args.lora
            )],
            )
        elif  args.method=="router_all":
            trainer = CustomTrainer_router_all(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[SparsityLoggingAllCallback(f'./logs/{args.method}_{args.sparsity}_{args.model_path}_{args.learning_rate}_{args.lr_scheduler_type}'), SaveCheckpointCallback(
                save_steps=200,  # 每1000步保存一次，可以根据需要修改
                model_name=args.model_path,
                method=args.method,
                sparsity=args.sparsity,
                lora=args.lora
            )],
            )
        elif args.method=="mod_twice" or args.method=="mod":
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[SaveCheckpointCallback(
                save_steps=200,  # 每1000步保存一次，可以根据需要修改
                model_name=args.model_path,
                method=args.method,
                sparsity=args.capacity,
                lora=args.lora
            )],
            )
        
        elif "post_training" in args.method:
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[SaveCheckpointCallback_post_training(
                save_steps=200,  # 每1000步保存一次，可以根据需要修改
                model_name=args.model_path,
                method=args.method,
                sparsity=args.sparsity,
            )],
            )
        
        elif args.method == "original":
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[SaveCheckpointCallback_original(
                save_steps=1000,  # 每1000步保存一次，可以根据需要修改
                model_name=args.model_path,
            )],
            )
        # trainer.model()
        # Train the model for the second stage
        trainer.train()
        
        # os.makedirs(f"{args.model_name}_{args.method}_{args.sparsity}", exist_ok=True)
        # file_path = f"{args.model_name}_{args.method}_{args.sparsity}/model.pth"
        # # # 训练结束后保存模型和分词器
        # torch.save(model, file_path)
        wandb.finish()  # End the second run
   
    else:
        wandb.init(project=f"dynamic_layer_skipping", mode="offline") 
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=[SparsityLoggingCallback()],
        )
        #Calculate and report on perplexity
        initial_results = trainer.evaluate()
        print(initial_results)
        print(f"Baseline {args.model_name} Results: Perplexity: {math.exp(initial_results['eval_loss']):.2f}")