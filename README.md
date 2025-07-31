<div align="center">

# SkipGPT: Dynamic Layer Pruning Reinvented with Token Awareness and Module Decoupling

![BoolQ](https://img.shields.io/badge/Dataset-BoolQ-blue)
![PIQA](https://img.shields.io/badge/Dataset-PIQA-blue)
![HellaSwag](https://img.shields.io/badge/Dataset-HellaSwag-blue)
![ARC-easy ](https://img.shields.io/badge/Dataset-ARC--easy-blue)
![ARC-challenge](https://img.shields.io/badge/Dataset-ARC--challenge-blue)
![OpenbookQA](https://img.shields.io/badge/Dataset-OpenbookQA-blue)
![WikiText2](https://img.shields.io/badge/Dataset-WikiText2-blue)
![PTB](https://img.shields.io/badge/Dataset-PTB-blue)

![LLaMA2-7B](https://img.shields.io/badge/Model-LLaMA2--7B-21C2A4)
![LLaMA2-13B](https://img.shields.io/badge/Model-LLaMA2--13B-21C2A4)
![LLaMA3.1-8B](https://img.shields.io/badge/Model-LLaMA3.1--8B-21C2A4)

📰 [Paper](https://arxiv.org/pdf/2506.04179)

</div>

## Introduction
Large language models (LLMs) deliver impressive performance but remain computationally expensive and structurally inefficient. Existing layer pruning methods often overlook two key aspects of pruning dynamics: token-wise computational variability (**horizontal dynamics**) and the distinct roles of MLP vs. attention modules (**vertical dynamics**).

<p align="center">
  <img src="image/introduction.png" width="60%" />
  <p align="center">An overview of SkipGPT. Unlike conventional static structured pruning, SkipGPT dynamically prunes layers by considering both horizontal and vertical dynamics. In horizontal dynamics, different tokens receive varying computational allocations. In vertical dynamics, the MLP and attention modules are decoupled to account for their distinct roles within each layer.</p>
</p>

**SkipGPT** is a dynamic pruning framework that tackles both. It adaptively prunes layers on a per-token basis and decouples MLP and attention pruning for fine-grained control. _To stabilize training, SkipGPT introduces a Two-Stage Training Paradigm_: first tuning a routing module while freezing the base model, then restoring performance via lightweight LoRA fine-tuning.
SkipGPT reduces up to **40**% of parameters without compromising—and sometimes even improving—performance. It also enables interpretability: revealing attention’s higher redundancy and dynamic computation needs as sequence length grows. These insights push forward both LLM efficiency and architectural understanding.


## ⚙️ Environment Setup

Create and activate an environment named `skipgpt`:

```bash
conda create -n skipgpt python=3.10
conda activate skipgpt
pip install -r requirements.txt
```

## Two-stage Training Paradigm
### Router Tuning
In the initial stage of training, we focus exclusively on tuning the router while keeping all other model parameters frozen.
```bash
python main.py \
--model_path models/Llama-3.1-8B \
--method router_attn_mlp \
--dataset_path datasets/RedPajama-Data-1T-Sample \
--warmup_ratio 0.1 \
--max_steps_stage 10000 \
--learning_rate 2e-3 \
--lr_scheduler_type cosine
```
### LoRA Fine-Tuning (Optional)
While the router tuning stage is sufficient to preserve most of the model’s performance, we provide an optional LoRA fine-tuning stage for those aiming to fully restore performance to the original model level.
```bash
python main.py \
--lora \
--model_path models/Llama-3.1-8B  \
--method post_training \
--dataset_path datasets/RedPajama-Data-1T-Sample \
--warmup_ratio 0.1 \
--max_steps_stage 10000 \
--learning_rate 2e-4 \
--lr_scheduler_type cosine
```
Note: To continue training, the checkpoint from the router stage must be loaded in models.py.

## Citation
```latex
@inproceedings{zhaoskipgpt,
  title={SkipGPT: Each Token is One of a Kind},
  author={Zhao, Anhao and Ye, Fanghua and Fan, Yingqi and Tong, Junlong and Xiong, Jing and Fei, Zhiwei and Su, Hui and Shen, Xiaoyu},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

