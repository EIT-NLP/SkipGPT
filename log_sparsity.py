from router_attn_mlp import router_attn_mlp_llama, router_attn_mlp_gemma
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
import wandb
import torch
import math
class SparsityLoggingCallback(TrainerCallback):
    def __init__(self, log_dir='logs'):
        self.writer = SummaryWriter(log_dir=log_dir)
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model', None)
        if model is not None:
            total_attn_sparsity = 0
            total_mlp_sparsity = 0
            num_layers = 0
            if "Llama" in model.__class__.__name__:
                # 遍历模型中的所有层并累积稀疏度
                for layer in model.model.layers:  
                    if isinstance(layer, router_attn_mlp_llama):
                        attn_sparsity, mlp_sparsity = layer.compute_sparsity()
                        total_attn_sparsity += attn_sparsity
                        total_mlp_sparsity += mlp_sparsity
                        num_layers += 1

                # 计算平均稀疏度
                average_attn_sparsity = total_attn_sparsity / num_layers if num_layers > 0 else 0
                average_mlp_sparsity = total_mlp_sparsity / num_layers if num_layers > 0 else 0

                # 记录到 wandb
                wandb.log({
                    "Average Attention Sparsity": average_attn_sparsity,
                    "Average MLP Sparsity": average_mlp_sparsity,
                    "epoch": state.epoch
                })

                # 使用 SummaryWriter 记录指标
                self.writer.add_scalar("Average_Attention_Sparsity", average_attn_sparsity, state.global_step)
                self.writer.add_scalar("Average_MLP_Sparsity", average_mlp_sparsity, state.global_step)
                self.writer.add_scalar("epoch", state.epoch, state.global_step)

                # 重置所有层的稀疏度计数
                for layer in model.model.layers:
                    if isinstance(layer, router_attn_mlp_llama):
                        layer.reset_sparsity_counts()

            elif "Gemma" in model.__class__.__name__:
                # 遍历模型中的所有层并累积稀疏度
                for layer in model.model.layers:  
                    if isinstance(layer, router_attn_mlp_gemma):
                        attn_sparsity, mlp_sparsity = layer.compute_sparsity()
                        total_attn_sparsity += attn_sparsity
                        total_mlp_sparsity += mlp_sparsity
                        num_layers += 1

                # 计算平均稀疏度
                average_attn_sparsity = total_attn_sparsity / num_layers if num_layers > 0 else 0
                average_mlp_sparsity = total_mlp_sparsity / num_layers if num_layers > 0 else 0

                # 记录到 wandb
                wandb.log({
                    "Average Attention Sparsity": average_attn_sparsity,
                    "Average MLP Sparsity": average_mlp_sparsity,
                    "epoch": state.epoch
                })
                
                # 使用 SummaryWriter 记录指标
                self.writer.add_scalar("Average_Attention_Sparsity", average_attn_sparsity, state.global_step)
                self.writer.add_scalar("Average_MLP_Sparsity", average_mlp_sparsity, state.global_step)
                self.writer.add_scalar("epoch", state.epoch, state.global_step)


                # 重置所有层的稀疏度计数
                for layer in model.model.layers:
                    if isinstance(layer, router_attn_mlp_gemma):
                        layer.reset_sparsity_counts()

            elif "PeftModel" in model.__class__.__name__ and "Llama" in model.model.__class__.__name__:
                # Iterate over all layers and accumulate sparsity
                for layer in model.model.model.layers:  
                    if isinstance(layer, router_attn_mlp_llama):
                        attn_sparsity, mlp_sparsity = layer.compute_sparsity()
                        total_attn_sparsity += attn_sparsity
                        total_mlp_sparsity += mlp_sparsity
                        num_layers += 1

                # 计算平均稀疏度
                average_attn_sparsity = total_attn_sparsity / num_layers if num_layers > 0 else 0
                average_mlp_sparsity = total_mlp_sparsity / num_layers if num_layers > 0 else 0

                # 记录到 wandb
                wandb.log({
                    "Average Attention Sparsity": average_attn_sparsity,
                    "Average MLP Sparsity": average_mlp_sparsity,
                    "epoch": state.epoch
                })
                
                # 使用 SummaryWriter 记录指标
                self.writer.add_scalar("Average_Attention_Sparsity", average_attn_sparsity, state.global_step)
                self.writer.add_scalar("Average_MLP_Sparsity", average_mlp_sparsity, state.global_step)
                self.writer.add_scalar("epoch", state.epoch, state.global_step)


                # 重置所有层的稀疏度计数
                for layer in model.model.model.layers:
                    if isinstance(layer, router_attn_mlp_llama):
                        layer.reset_sparsity_counts()

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()
            
class SparsityLoggingAllCallback(TrainerCallback):
    def __init__(self, log_dir='logs'):
        self.writer = SummaryWriter(log_dir=log_dir)
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model', None)

        if model is not None:
            total_sparsity = 0
            num_layers = 0
            if "Llama" in model.__class__.__name__:
                # Iterate over all layers and accumulate sparsity
                for layer in model.model.layers:
                    if isinstance(layer, router_all_llama):
                        sparsity = layer.compute_sparsity()
                        total_sparsity += sparsity
                        num_layers += 1

                # Calculate average sparsity
                average_sparsity = total_sparsity / num_layers if num_layers > 0 else 0

                # Log to wandb
                wandb.log({
                    "Average Layer Sparsity": average_sparsity,
                    "epoch": state.epoch
                })

                # 使用 SummaryWriter 记录指标
                self.writer.add_scalar("Average_Layer_Sparsity", average_sparsity, state.global_step)
                self.writer.add_scalar("epoch", state.epoch, state.global_step)

                # Reset sparsity counts for all layers
                for layer in model.model.layers:
                    if isinstance(layer, router_all_llama):
                        layer.reset_sparsity_counts()

            elif "Gemma" in model.__class__.__name__:
                # Iterate over all layers and accumulate sparsity
                for layer in model.model.layers:
                    if isinstance(layer, router_all_gemma):
                        sparsity = layer.compute_sparsity()
                        total_sparsity += sparsity
                        num_layers += 1

                # Calculate average sparsity
                average_sparsity = total_sparsity / num_layers if num_layers > 0 else 0

                # Log to wandb
                wandb.log({
                    "Average Layer Sparsity": average_sparsity,
                    "epoch": state.epoch
                })

                # Reset sparsity counts for all layers
                for layer in model.model.layers:
                    if isinstance(layer, router_all_gemma):
                        layer.reset_sparsity_counts()

            elif "PeftModel" in model.__class__.__name__ and "Llama" in model.model.__class__.__name__:
                # Iterate over all layers and accumulate sparsity
                for layer in model.model.model.layers:
                    if isinstance(layer, router_all_llama):
                        sparsity = layer.compute_sparsity()
                        total_sparsity += sparsity
                        num_layers += 1

                # Calculate average sparsity
                average_sparsity = total_sparsity / num_layers if num_layers > 0 else 0

                # Log to wandb
                wandb.log({
                    "Average Layer Sparsity": average_sparsity,
                    "epoch": state.epoch
                })

                # 使用 SummaryWriter 记录指标
                self.writer.add_scalar("Average_Layer_Sparsity", average_sparsity, state.global_step)
                self.writer.add_scalar("epoch", state.epoch, state.global_step)

                # Reset sparsity counts for all layers
                for layer in model.model.model.layers:
                    if isinstance(layer, router_all_llama):
                        layer.reset_sparsity_counts()

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()

# 自定义的 Wandb 回调
class CustomWandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 如果 metrics 中包含 eval_loss，则将其日志记录到 wandb
        if "eval_loss" in metrics:
            wandb.log({"eval_loss": metrics["eval_loss"], "step": state.global_step})

# 自定义学习率调度
class ConstantCosineSchedulerCallback(TrainerCallback):
    """
    自定义回调，实现先保持学习率恒定，然后进行余弦衰减。
    """

    def __init__(self, constant_steps, total_steps):
        """
        Args:
            constant_steps (int): 保持学习率恒定的步数。
            total_steps (int): 总的训练步数。
        """
        self.constant_steps = constant_steps
        self.total_steps = total_steps

    def on_train_begin(self, args, state, control, **kwargs):
        """
        在训练开始时初始化调度器。
        """
        optimizer = kwargs['optimizer']
        self.current_step = 0
        self.constant_steps = self.constant_steps
        self.total_steps = self.total_steps

        def lr_lambda(current_step):
            if current_step < self.constant_steps:
                return 1.0  # 保持学习率恒定
            else:
                progress = (current_step - self.constant_steps) / (self.total_steps - self.constant_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))  # 余弦衰减

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def on_step_end(self, args, state, control, **kwargs):
        """
        在每个训练步骤结束时更新学习率。
        """
        self.scheduler.step()
        self.current_step += 1
        return control