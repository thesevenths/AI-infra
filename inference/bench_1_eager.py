"""
层1：PyTorch (HF transformers) eager —— 基线

特点：eager 算子逐个 launch，无融合、无批处理调度。
并发档位(>1)用「静态批处理 + 等长序列」模拟：这正是 HF generate 的批处理
方式——全批同步前进，谁先出 EOS 也得陪跑到最长那条。

用法：
  python bench_1_eager.py --concurrency 1,4 --repeats 3
"""
import torch

from llm_bench import build_parser, run_suite, warmup


def main():
    args = build_parser("1_eager").parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,          # 统一 fp16 保证各层同精度
        attn_implementation="sdpa",
    )
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARN: 未检测到 CUDA，将测在 CPU 上，数字仅用于验证脚本")
    model = model.to(device).eval()

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    shapes = [(b, 128, 8) for b in concurrencies]
    warmup(model, tokenizer, device, shapes)

    run_suite("1_eager", model, tokenizer, device, args)


if __name__ == "__main__":
    main()
