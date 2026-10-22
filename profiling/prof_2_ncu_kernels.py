"""
Experiment 4 / 层B：给 Nsight Compute 用的最小 kernel 载荷

不重复加载全套 bench：只做一次 prefill + N 步 decode（batch=1, eager），
让 ncu 的 kernel replay 开销可控。本脚本自己**不调用** ncu——由
run_ncu_local.ps1（或云端命令行）在外面包：

  ncu --set basic --launch-skip <跳过 prefill 的 launch 数> \
      --launch-count 40 --target-processes all \
      python profiling/prof_2_ncu_kernels.py

launch 数与模型层数相关（Qwen-0.6B 28 层 ≈ 每 forward 60 个 launch），
prefill 的 lm_head 是全序列 logits（logits_to_keep=1，只算末位，无额外）。
默认 skip=70 ≈ prefill(约60) + 少量，从第一步 decode 的 kernel 开始采。

用法：
  python profiling/prof_2_ncu_kernels.py --steps 10
"""
import argparse
import sys
from pathlib import Path

import torch

INFERENCE_DIR = Path(__file__).resolve().parent.parent / "inference"
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_bench import DEFAULT_MODEL, make_input  # noqa: E402


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description="ncu payload: single-request decode")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    assert torch.cuda.is_available(), "ncu 载荷需要 CUDA"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = dict(torch_dtype=torch.float16, attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        model = model.cuda().eval()
    except OSError as e:
        # Windows 提交内存不足（os error 1455）：流式低内存加载
        print(f"[payload] 常规加载失败({e})，改用流式加载 ...", flush=True)
        from stream_load import load_hf_model
        model = load_hf_model(args.model, "cuda")

    total_len = args.input_len + args.steps
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=total_len, device="cuda",
                        dtype=torch.float16)
    input_ids = make_input(tokenizer, "cuda", 1, args.input_len, seed=0)
    attention_mask = torch.ones(1, total_len, dtype=torch.long, device="cuda")
    pos = torch.arange(args.input_len, device="cuda").unsqueeze(0)

    # prefill（ncu 用 --launch-skip 跳过这段）
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=pos, cache_position=torch.arange(args.input_len, device="cuda"),
                past_key_values=cache, use_cache=True, logits_to_keep=1)
    tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    print("prefill done", flush=True)

    # decode：每步之间 sync，保证 ncu replay 时 launch 序列可预期
    for i in range(args.steps):
        cur_pos = args.input_len + i
        out = model(input_ids=tok,
                    position_ids=torch.full((1, 1), cur_pos, dtype=torch.long, device="cuda"),
                    cache_position=torch.tensor([cur_pos], device="cuda"),
                    attention_mask=attention_mask,
                    past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        print(f"decode step {i} done", flush=True)
    print("payload finished", flush=True)


if __name__ == "__main__":
    main()
