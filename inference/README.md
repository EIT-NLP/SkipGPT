## The inference implementation of SkipGPT
---

Here is the inference implementation of SkipGPT, which provides `dense`, `torch-sparse`, and `triton-sparse` pipelines for evaluation

- **dense** is the vanilla huggingface implementation of LLM, with minor modifications in KV Cache layout (seq first) for `FlashAttention`
- **torch-sparse** is the implementation of SkipGPT, which only provided sparsity in FFN calculation
- **triton-sparse** is the triton-based implementation of SkipGPT, with fully sparse pipeline based on custom attention kernel

### Requirements
- Triton 3.4.0+
- lm-evaluation-harness

### Usage
- `./ckpt` includes the router and lora checkpoints in stage 1 and 2. Or you can run `python ./ckpt/convert.py --ckpt_path ${YOUR_CKPT_PATH} --output_path ${OUTPUT_PATH} --stage stage1` to convert the finetuned ckpt into seperate router and lora ckpt.
- `./inference/wrapper` includes the wrapper for specific HF models, as well as the basic `AttentionWrapper` and `MLPWrapper`
- `./inference/ops` includes the triton kernels for `triton-sparse` forward pipeline
- `./inference.py` is the main entry for SkipGPT's inference, you can run:
  ```bash
  python inference.py \
  --model_path ${YOUR_HF_MODEL_PATH} \
  --router_path ${THE_TRAINED_ROUTER_CKPT_PATH} \
  --lora_path ${THE_TRAINED_LORA_CKPT_PATH} \
  --save_path ${THE_SAVE_PATH} \
  --forward_impl torch \
  --task wikitext \
  --batch_size ${BATCH_SIZE}
  ```
  for `dense` evaluation on `lm-evaluation-harness`'s `wikitext2` task, or
  ```bash
  python inference.py \
  --model_path ${YOUR_HF_MODEL_PATH} \
  --router_path ${THE_TRAINED_ROUTER_CKPT_PATH} \
  --lora_path ${THE_TRAINED_LORA_CKPT_PATH} \
  --save_path ${THE_SAVE_PATH} \
  --forward_impl torch \
  --sparse \
  --task wikitext \
  --batch_size ${BATCH_SIZE}
  ```
  for `torch-sparse`, or
  ```bash
  python inference.py \
  --model_path ${YOUR_HF_MODEL_PATH} \
  --router_path ${THE_TRAINED_ROUTER_CKPT_PATH} \
  --lora_path ${THE_TRAINED_LORA_CKPT_PATH} \
  --save_path ${THE_SAVE_PATH} \
  --forward_impl triton \
  --sparse \
  --task wikitext \
  --batch_size ${BATCH_SIZE}
  ```
  for `triton-sparse`.
  For the inference throughput benchmarking, you can run:
  ```bash
  python inference.py \
  --model_path ${YOUR_HF_MODEL_PATH} \
  --save_path ${THE_SAVE_PATH} \
  --task inference_throughput \
  --batch_size ${BATCH_SIZE} \
  --sparsity 0.25 \
  --inference_mode prefill/decode
  ```
  for evaluating the inference throughput of different pipelines based on triton's `do_bench()`

### Note
The current implementation addresses an issue where Triton triggers kernel re-tuning and recompilation whenever any input constexpr values changes. In SkipGPT, since the sequence length (seqlen) or the (batch_size * seqlen) varies across layers, this could—in severe cases—result in every layer requiring kernel recompilation. To mitigate this, the `do_not_specialize` flag is set, which prevents excessive recompilation. However, this approach also restricts the optimization potential of autotune, leading to performance degradation of `triton-sparse` under several benchmark settings.

Additionally, when the processing time per batch is short, variations in token numbers across different batches can similarly trigger kernel re-tuning during the prefill stage. This behavior likewise impacts the overall runtime in `lm-evaluation-harness`.