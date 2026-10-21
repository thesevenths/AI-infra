"""
层3：vLLM —— 只换服务架构（PagedAttention + continuous batching），内核仍是 PyTorch

平台限制（重要）：
  vLLM 只支持 Linux。Windows 本机需要 WSL2：
      wsl --install
      # 在 WSL2 (Ubuntu) 内: pip install vllm && python bench_3_vllm.py ...
  2GB 级显卡上大概率 OOM：CUDA context + 权重 + 引擎开销后已没有多少
  KV space。本脚本参数按"云端 A100/4090 也能用同一命令"设计，
  小显存机器请减小 --max-model-len。

测量方法与层1/2 对齐：
  - TTFT      : AsyncLLMEngine 流式输出，第 1 个 token 到达的时刻
  - TPOT      : (末 token 时刻 - 第 1 token 时刻) / (token 数 - 1)
  - Throughput: 总输出 token / 全部请求完成的墙钟时间
  - VRAM      : nvidia-smi 峰值 + vLLM 自报的 KV cache blocks

用法：
  python bench_3_vllm.py --concurrency 1,4,16,32 --repeats 1
"""

import asyncio
import sys
import time
from argparse import ArgumentParser

from llm_bench import (DEFAULT_MODEL, WORKLOADS, RunResult, make_input,
                       nvidia_smi_used_mib, save_and_print_table)


async def one_request(engine, prompt_ids, output_len, sem, req_id):
    """单条请求：走引擎流式接口，记录首 token 与各 token 到达时刻。"""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=output_len,
                        ignore_eos=True)  # ignore_eos: 固定输出长度，对齐层1/2
    async with sem:
        t0 = time.perf_counter()
        token_times = []
        n = 0
        async for out in engine.generate(
                {"prompt_token_ids": prompt_ids}, sp, request_id=str(req_id)):
            n = len(out.outputs[0].token_ids)
            while len(token_times) < n:
                token_times.append(time.perf_counter())
        if not token_times:
            return None
        ttft_ms = (token_times[0] - t0) * 1e3
        tpot_ms = ((token_times[-1] - token_times[0]) * 1e3
                   / max(1, len(token_times) - 1))
        return ttft_ms, tpot_ms, token_times[-1] - t0


async def bench_loop(engine, tokenizer, args):
    """所有负载在同一个事件循环里跑完：AsyncLLMEngine 绑定创建时的 loop，
    不能每个档位 asyncio.run 一次。"""
    concurrencies = [int(x) for x in args.concurrency.split(",")]
    selected = []
    if "A" in args.workloads:
        selected.append(("A_short_in_long_out", WORKLOADS["A_short_in_long_out"]))
    if "B" in args.workloads:
        selected.append(("B_long_in_short_out", WORKLOADS["B_long_in_short_out"]))

    results, req_id = [], 0
    for wname, wl in selected:
        for b in concurrencies:
            res = RunResult(method="3_vllm", workload=wname, concurrency=b,
                            repeats=args.repeats)
            seen_any = False
            for r in range(args.repeats):
                sem = asyncio.Semaphore(b)
                tasks = []
                for _ in range(b):
                    req_id += 1
                    ids = make_input(tokenizer, "cpu", 1,
                                     wl["input_len"], seed=r)[0].tolist()
                    tasks.append(one_request(engine, ids, wl["output_len"],
                                             sem, req_id))
                t0 = time.perf_counter()
                got = await asyncio.gather(*tasks)
                wall = time.perf_counter() - t0
                got = [g for g in got if g]
                if not got:
                    continue
                seen_any = True
                for ttft, tpot, _ in got:
                    res.add_sample(ttft, tpot, wall, b, wl["output_len"])
                res.vram_smi_peak_mib = max(res.vram_smi_peak_mib,
                                            nvidia_smi_used_mib())
            if seen_any:
                results.append(res.finalize())
                print(f"  {wname} batch={b}: TTFT {res.ttft_ms:.1f}ms | "
                      f"TPOT {res.tpot_ms:.1f}ms | "
                      f"{res.throughput_tps:.1f} tok/s", flush=True)
    return results


def main():
    if sys.platform == "win32":
        sys.exit("vLLM 不支持 Windows 原生运行，请在 WSL2/Linux 中执行本脚本。")
    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit("未安装 vllm: pip install vllm")

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from transformers import AutoTokenizer

    p = ArgumentParser(description="Experiment 2 / layer-3 vLLM bench")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", default="1,4,16,32")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--workloads", default="A,B")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    # 与层1/2 同精度：老卡（SM<8.0）只能 float16；A100+ 可 --dtype bfloat16
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model, dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,  # 关掉 CUDA graph，保证"层3 vs 层2"只隔离架构变量
    ))
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = asyncio.run(bench_loop(engine, tokenizer, args))
    save_and_print_table("3_vllm", results)


if __name__ == "__main__":
    main()
