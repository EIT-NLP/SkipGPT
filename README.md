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

![GPT-2 XL](https://img.shields.io/badge/Model-GPT2--XL-21C2A4)
![GPT-J](https://img.shields.io/badge/Model-GPT--J-21C2A4)
![Llama-2-7B](https://img.shields.io/badge/Model-Llama--2--7B-21C2A4)
![Llama-2-13B](https://img.shields.io/badge/Model-Llama--2--13B-21C2A4)
![Falcon-40B](https://img.shields.io/badge/Model-Falcon--40B-21C2A4)

📰 [Paper](https://www.arxiv.org/abs/2407.17011)

</div>

## 1. Introduction
Large language models (LLMs) exhibit remarkable in-context learning (ICL) capabilities. However, the underlying working mechanism of ICL remains poorly understood. Recent research presents two conflicting views on ICL: One attributes it to LLMs' inherent ability of task recognition, deeming label correctness and shot numbers of demonstrations as not crucial; the other emphasizes the impact of similar examples in the demonstrations, stressing the need for label correctness and more shots. 

<p align="center">
  <img src="image/introduction.jpg" width="60%" />
  <p align="center">An overview of the proposed two-dimensional coordinate system for ICL.</p>
</p>

In this work, we provide a **Two-Dimensional Coordinate System** that unifies both views into a systematic framework. The framework explains the behavior of ICL through two orthogonal variables: *whether LLMs can recognize the task* and *whether similar examples are presented in the demonstrations*. We propose the peak inverse rank metric to detect the task recognition ability of LLMs and study LLMs' reactions to different definitions of similarity. Based on these, we conduct extensive experiments to elucidate how ICL functions across each quadrant on multiple representative classification tasks. Finally, we extend our analyses to generation tasks, showing that our coordinate system can also be used to interpret ICL for generation tasks effectively.

