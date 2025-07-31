from transformers import TrainerCallback
import os
import torch
class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"{model_name}_{method}_{sparsity}_lora{lora}"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):

        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                
                # 保存模型
                torch.save(model.state_dict(), file_path)
                print(f"Model saved at step {state.global_step}")



class SaveCheckpointCallback_post_training(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity):
        self.save_steps = save_steps
        self.save_dir = f"{model_name}_{method}_{sparsity}_post_training"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.bin")
                
                # 保存模型
                torch.save(model.state_dict(), file_path)
                print(f"Model saved at step {state.global_step}")

class SaveCheckpointCallback_original(TrainerCallback):
    def __init__(self, save_steps, model_name):
        self.save_steps = save_steps
        self.save_dir = f"{model_name}_original_lora"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.bin")
                
                # 保存模型
                torch.save(model.state_dict(), file_path)
                print(f"Model saved at step {state.global_step}")

class CustomEvalCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 每隔 200 个 global_step 进行评估
        if state.global_step % 100 == 0 and state.global_step > 0:
            trainer = kwargs['trainer']
            eval_results = trainer.evaluate()
            print(f"Evaluation at step {state.global_step}: {eval_results}")