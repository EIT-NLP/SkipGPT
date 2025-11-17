import torch
import time

from argparse import Namespace, ArgumentParser
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from inference.wrapper import apply_wrapper
from inference.benchmark import inference_benchmark

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="./llm/lama-3.1-8B", help="Path to the HF model to load")
    parser.add_argument("--router_path", type=str, default="./ckpt/stage1/router.pth", help="Path to the router to load")
    parser.add_argument("--lora_path", type=str, default="./ckpt/stage1/lora.pth", help="Path to the lora to load")
    parser.add_argument("--save_path", type=str, default="./inference", help="Path to save the inference benchmark results")

    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--forward_impl", type=str, choices=['torch', 'triton'], default='triton')
    parser.add_argument("--sparse", action='store_true')

    parser.add_argument("--task", type=str, default="inference_throughput")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sparsity", type=float, default=0.25)
    parser.add_argument("--inference_mode", type=str, default="prefill")

    args = parser.parse_args()

    model, config, tokenizer = apply_wrapper(args)
    model = model.to('cuda')

    if args.task != "inference_throughput":
        llm = HFLM(
            model,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
            batch_size=args.batch_size,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tasks = [args.task]

        torch.cuda.synchronize()
        start = time.perf_counter()
        results = simple_evaluate(
            llm,
            tasks=tasks
        )
        torch.cuda.synchronize()
        duration = time.perf_counter() - start

        if results is not None:
            print(args)
            print(make_table(results))
            print(f"Duration: {duration:.4f}s")
    else:
        assert args.sparsity > 0 and args.sparsity < 1
        bench = inference_benchmark(model, config, batch_size=args.batch_size, sparsity=args.sparsity, mode=args.inference_mode)
        bench.run(print_data=True, show_plots=False, save_path=args.save_path)
