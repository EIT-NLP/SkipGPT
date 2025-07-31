from transformers import Trainer
from router_attn_mlp import router_attn_mlp_llama, router_attn_mlp_gemma
from transformers import AdamW
import numpy as np

class CustomTrainer1(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)
        # 初始化router损失
        attn_router_loss = 0.0
        mlp_router_loss = 0.0
        num_layers = 0  # 用来记录层数

        if "Llama" in model.__class__.__name__:
            # 遍历模型的每一层，收集router决定计算的平均值
            for layer in model.model.layers:
                if isinstance(layer, router_attn_mlp_llama):
                    attn_router_loss += layer.attn_router_zero_prob
                    mlp_router_loss += layer.mlp_router_zero_prob
                    num_layers += 1  # 记录层数

        
        if num_layers > 0:
            # 在层数上取平均
            attn_router_loss /= num_layers
            mlp_router_loss /= num_layers

        # 计算正则化项
        # total_router_loss = - (attn_router_loss + mlp_router_loss) / 2
        total_router_loss= -0.05*attn_router_loss - mlp_router_loss
        
        return total_router_loss
    
class CustomTrainer_router_attn_mlp(Trainer):
    def __init__(self, *args, **kwargs):
        self.custom_args = kwargs.pop('custom_args', None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # 标准损失计算
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        # 仅在训练时添加正则化损失
        if self.model.training:
            # 初始化router损失
            attn_router_loss = 0.0
            mlp_router_loss = 0.0
            num_layers = 0  # 用来记录层数

            if "Llama" in model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.layers:
                    if isinstance(layer, router_attn_mlp_llama):
                        attn_router_loss += layer.attn_router_zero_prob
                        mlp_router_loss += layer.mlp_router_zero_prob
                        num_layers += 1  # 记录层数
            
            elif "Gemma" in model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.layers:
                    if isinstance(layer, router_attn_mlp_gemma):
                        attn_router_loss += layer.attn_router_zero_prob
                        mlp_router_loss += layer.mlp_router_zero_prob
                        num_layers += 1  # 记录层数
            
            
            elif "PeftModel" in model.__class__.__name__ and "Llama" in model.model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.model.layers:
                    if isinstance(layer, router_attn_mlp_llama):
                        attn_router_loss += layer.attn_router_zero_prob
                        mlp_router_loss += layer.mlp_router_zero_prob
                        num_layers += 1  # 记录层数

            elif  "DeepSpeedEngine" in model.__class__.__name__ and "Llama" in model.module.model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.module.model.layers:
                    if isinstance(layer, router_attn_mlp_llama):
                        attn_router_loss += layer.attn_router_zero_prob
                        mlp_router_loss += layer.mlp_router_zero_prob
                        num_layers += 1

            if num_layers > 0:
                # 在层数上取平均
                attn_router_loss /= num_layers
                mlp_router_loss /= num_layers
            
            total_router_loss = (attn_router_loss + mlp_router_loss) / 2
            
            if self.custom_args.train_method=="curriculum":
                
                # 获取当前训练的步骤数
                current_step = self.state.global_step

                if current_step < 500:
                    target_value = 1.0
                elif current_step < 1000 and current_step >= 500:
                    target_value = 0.9
                elif current_step < 1500 and current_step >= 1000:
                    target_value = 0.8
                elif current_step < 2000 and current_step >= 1500:
                    target_value = 0.7
                elif current_step < 2500 and current_step >= 2000:
                    target_value = 0.6
                elif current_step < 3000 and current_step >= 2500:
                    target_value = 0.5
                elif current_step < 3500 and current_step >= 3000:
                    target_value = 0.4
                elif current_step < 4000 and current_step >= 3500:
                    target_value = 0.3
                elif current_step < 4500 and current_step >= 4000:
                    target_value = 0.2
                else:
                    target_value = 0.1

                loss = loss + 20*abs(target_value-total_router_loss)# 可以调节正则化项的权重
            
            elif self.custom_args.train_method=="target_sparsity":
                loss = loss + 20*abs((1-self.custom_args.sparsity)-total_router_loss)# 可以调节正则化项的权重
        return loss
    

class CustomTrainer_router_all(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 标准损失计算
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        # 仅在训练时添加正则化损失
        if self.model.training:
            # 初始化router损失
            router_loss = 0.0
            num_layers = 0  # 用来记录层数

            if "Llama" in model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.layers:
                    if isinstance(layer, router_all_llama):
                        router_loss += layer.block_router_zero_prob
                        num_layers += 1  # 记录层数
            
            elif "Gemma" in model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.layers:
                    if isinstance(layer, router_all_gemma):
                        router_loss += layer.router_zero_prob
                        num_layers += 1  # 记录层数

            elif "PeftModel" in model.__class__.__name__ and "Llama" in model.model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.model.layers:
                    if isinstance(layer, router_all_llama):
                        router_loss += layer.block_router_zero_prob
                        num_layers += 1

            
            elif  "DistributedDataParallel" in model.__class__.__name__ and "Llama" in model.module.model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.module.model.model.layers:
                    if isinstance(layer, router_all_llama):
                        router_loss+= layer.block_router_zero_prob
                        num_layers += 1
                        
            if num_layers > 0:
                # 在层数上取平均
                router_loss /= num_layers
            
            # 获取当前训练的步骤数
            current_step = self.state.global_step

            if current_step < 500:
                target_value = 1.0
            elif current_step < 1000 and current_step >= 500:
                target_value = 0.9
            elif current_step < 1500 and current_step >= 1000:
                target_value = 0.8
            elif current_step < 2000 and current_step >= 1500:
                target_value = 0.7
            elif current_step < 2500 and current_step >= 2000:
                target_value = 0.6
            elif current_step < 3000 and current_step >= 2500:
                target_value = 0.5
            elif current_step < 3500 and current_step >= 3000:
                target_value = 0.4
            elif current_step < 4000 and current_step >= 3500:
                target_value = 0.3
            elif current_step < 4500 and current_step >= 4000:
                target_value = 0.2
            else:
                target_value = 0.1

            # 合并总损失
            loss = loss + 10*abs(0.7-router_loss)# 可以调节正则化项的权重
        return loss
    

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 标准损失计算
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        # 仅在训练时添加正则化损失
        if self.model.training:
            # 初始化router损失
            attn_router_loss = 0.0
            mlp_router_loss = 0.0
            num_layers = 0  # 用来记录层数

            if "Llama" in model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.layers:
                    if isinstance(layer, router_attn_mlp_llama):
                        attn_router_loss += layer.attn_router_zero_prob
                        mlp_router_loss += layer.mlp_router_zero_prob
                        num_layers += 1  # 记录层数

            if num_layers > 0:
                # 在层数上取平均
                attn_router_loss /= num_layers
                mlp_router_loss /= num_layers

            # 计算正则化项
            total_router_loss = (attn_router_loss + mlp_router_loss) / 2
            
            # 合并总损失
            loss = loss -  4*total_router_loss  # 可以调节正则化项的权重
        return loss

class Trainer_CustomOptimizer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        # 覆盖优化器，不管 self.optimizer 是否已经存在
        self.optimizer = learning_rate_group(self.model)  # 使用自定义的优化器
        # 使用父类提供的调度器创建逻辑
        if self.lr_scheduler is None:
            self.lr_scheduler = self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
            
    def compute_loss(self, model, inputs, return_outputs=False):
        # 标准损失计算
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        # 仅在训练时添加正则化损失
        if self.model.training:
            # 初始化router损失
            attn_router_loss = 0.0
            mlp_router_loss = 0.0
            num_layers = 0  # 用来记录层数

            if "Llama" in model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.layers:
                    if isinstance(layer, router_attn_mlp_llama):
                        attn_router_loss += layer.attn_router_zero_prob
                        mlp_router_loss += layer.mlp_router_zero_prob
                        num_layers += 1  # 记录层数
            
            elif "Gemma" in model.__class__.__name__:
                # 遍历模型的每一层，收集router决定计算的平均值
                for layer in model.model.layers:
                    if isinstance(layer, router_attn_mlp_gemma):
                        attn_router_loss += layer.attn_router_zero_prob
                        mlp_router_loss += layer.mlp_router_zero_prob
                        num_layers += 1  # 记录层数

            if num_layers > 0:
                # 在层数上取平均
                attn_router_loss /= num_layers
                mlp_router_loss /= num_layers

            # 计算正则化项
            total_router_loss = (attn_router_loss + mlp_router_loss) / 2
            
            # 合并总损失
            loss = loss + 4*abs(0.5-total_router_loss)# 可以调节正则化项的权重
        return loss

def learning_rate_group(model):
    # 定义一个空的参数组列表
    param_groups = []

    # 迭代模型的所有参数，并根据需求设置学习率
    for name, param in model.named_parameters():
        if "router" in name:  # 如果参数的名称中包含 "router"
            param_groups.append({'params': [param], 'lr': 5e-5})  # 设置较低学习率
        else:
            param_groups.append({'params': [param], 'lr': 2e-6})  # 设置较高学习率

    # 创建 AdamW 优化器并设置全局的 weight_decay
    optimizer = AdamW(param_groups, lr=2e-5, weight_decay=0.01)  # 全局的 weight_decay

    return optimizer