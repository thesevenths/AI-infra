"""
PyTorch Profiler 算子级瓶颈定位

回答的问题：慢在哪个算子？kernel 数量 / 空隙 / 访存拷贝占多少？GPU 忙不忙？

与 throughput benchmark（inference/bench_1、bench_2）同一模型、同一负载
（Qwen-0.6B fp16, workload A），但 prefill 与 decode 两条路径**分开采集**
——它们慢的原因完全不同，混在一起会被平均掉。对每个变体输出：
  1. key_averages 表（按 self_cuda_time_total 排序）-> results/*.txt
  2. chrome trace JSON（chrome://tracing / Perfetto 打开，eager 的
     "锯齿" vs cudagraphs 的"实心块"肉眼可见）      -> results/*.json
  3. 4 个关键指标（脚本自动计算）：
       - 每 decode 步 kernel 总数
       - 单 kernel 平均时长
       - GPU busy 率（kernel 时间轴并集 / 墙钟，空隙=launch 等待）
       - memcpy/memset 次数与字节数

用法（在 inference 环境里跑，脚本自行把 llm_bench 加入 sys.path）：
  python profiling/prof_1_torch_profiler.py --variant both --steps 30
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# 复用 Experiment 2 的测量库（不修改 inference/ 下任何代码）
INFERENCE_DIR = Path(__file__).resolve().parent.parent / "inference"
sys.path.insert(0, str(INFERENCE_DIR))
from llm_bench import DEFAULT_MODEL, make_input  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# decode 循环与 llm_bench.hf_measure 保持同一口径（StaticCache、
# attention_mask 全长 1、logits_to_keep、mark_step_begin），区别是
# 每步用 profiler record_function 包住，以便逐段导出。
MARK_STEP = (torch.compiler.cudagraph_mark_step_begin
             if "torch._inductor.cudagraph_trees" in sys.modules else None)


def union_interval(intervals):
    """合并重叠区间后返回总时长（µs），用于 GPU busy 率。"""
    total, cur_end = 0.0, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            total += end - start
            cur_end = end
        elif end > cur_end:
            total += end - cur_end
            cur_end = end
    return total


def compute_metrics(trace_path, wall_s, steps):
    """从导出的 chrome trace 计算 4 个关键指标 + 便于写报告的原始数字。

    注意：不能用 prof.events() 算——CPU op 事件也带聚合的 device time，
    会把 kernel 数/总时长重复计入 2~3 倍；trace 里 cat==kernel /
    gpu_memcpy / gpu_memset 才是精确的设备侧记录。
    """
    with open(trace_path, encoding="utf-8") as f:
        evs = json.load(f)["traceEvents"]
    kern = [e for e in evs if e.get("cat") == "kernel"]
    mcpy = [e for e in evs if e.get("cat") == "gpu_memcpy"]
    mset = [e for e in evs if e.get("cat") == "gpu_memset"]

    def iv(e):
        return (e["ts"], e["ts"] + max(e.get("dur", 0), 0))

    busy_us = union_interval([iv(e) for e in kern + mcpy + mset])
    kernel_us = sum(e.get("dur", 0) for e in kern)
    mb = sum(e.get("args", {}).get("bytes", 0) for e in mcpy + mset)

    # 逐步统计：用 record_function("decode_step_i") 的 GPU 侧区间，
    # 中位数自然排除 graph 捕获/重编译的首步（稳态 TPOT）。
    step_ann = sorted((e for e in evs if e.get("cat") == "gpu_user_annotation"
                       and str(e.get("name", "")).startswith("decode_step_")),
                      key=lambda e: e["ts"])
    per_step = []
    if step_ann and steps:
        all_dev = sorted(kern + mcpy + mset, key=lambda e: e["ts"])
        for a in step_ann:
            s0, s1 = a["ts"], a["ts"] + a.get("dur", 0)
            seg = [e for e in all_dev if e["ts"] >= s0 and e["ts"] < s1]
            per_step.append({
                "wall_ms": a.get("dur", 0) / 1e3,
                "busy": union_interval([iv(e) for e in seg]) / max(a.get("dur", 1), 1),
                "kernels": len(seg),
                "kernel_ms": sum(e.get("dur", 0) for e in seg) / 1e3,
            })
    med = lambda xs: statistics.median(xs) if xs else None
    step_walls = [p["wall_ms"] for p in per_step]
    steady = {"steady_tpot_ms": med(step_walls),
              "steady_busy": med([p["busy"] for p in per_step]),
              "steady_kernels_per_step": med([p["kernels"] for p in per_step]),
              "steady_kernel_ms_per_step": med([p["kernel_ms"] for p in per_step])}

    by_name = {}
    for e in kern:
        nm = e["name"]
        d = by_name.setdefault(nm, [0, 0.0])
        d[0] += 1
        d[1] += e.get("dur", 0)
    # 设备核级热点（mangled kernel name）。这是从 chrome trace 的 cat==kernel
    # 事件直接聚合的——与脚本打印的 key_averages 表格里**设备核行**完全对应
    # （例如 gemv 在 trace 聚合与表格里都是同一数值），因此 JSON 字段与报告
    # 表格自洽。算子级（aten::*）归因见 results/*.table.txt。
    top_kernels = sorted(((n, c, round(us, 1)) for n, (c, us) in by_name.items()),
                         key=lambda x: -x[2])[:12]
    m = {
        "wall_ms": wall_s * 1e3,
        "wall_per_step_ms": (wall_s * 1e3 / steps) if steps else None,
        "gpu_busy_ratio": busy_us / (wall_s * 1e6) if wall_s > 0 else 0.0,
        "kernel_time_sum_ms": kernel_us / 1e3,
        "num_kernels": len(kern),
        "kernels_per_step": (len(kern) / steps) if steps else None,
        "avg_kernel_us": (kernel_us / len(kern)) if kern else 0.0,
        "num_memcpy": len(mcpy),
        "num_memset": len(mset),
        "memcpy_mb": mb / 2**20,
        "unique_kernel_names": len({e["name"] for e in kern}),
        "top_kernels_by_total_us": top_kernels,
        "steady": steady,
    }
    return m


@torch.inference_mode()
def run_prefill(model, tokenizer, device, batch, input_len, cache, tag,
                out_shape=None):
    """prefill 一次 forward，包在 record_function 里，整体套 profiler。"""
    input_ids = make_input(tokenizer, device, batch, input_len, seed=0)
    total_len = input_len + (out_shape or 0)
    attention_mask = torch.ones(batch, total_len, dtype=torch.long, device=device)
    pos = torch.arange(input_len, device=device).unsqueeze(0).expand(batch, -1)
    if MARK_STEP:
        MARK_STEP()
    return model(input_ids=input_ids, attention_mask=attention_mask,
                 position_ids=pos.contiguous(),
                 cache_position=torch.arange(input_len, device=device),
                 past_key_values=cache, use_cache=True, logits_to_keep=1)


@torch.inference_mode()
def run_decode_steps(step_model, tokenizer, device, batch, input_len, steps,
                     cache):
    """从 prefill 后的 cache 接着走 `steps` 步 decode；每步一个 record_function。"""
    attention_mask = torch.ones(batch, input_len + steps,
                                dtype=torch.long, device=device)
    g = torch.Generator(device="cpu").manual_seed(0)
    tok = torch.randint(1000, tokenizer.vocab_size - 10, (batch, 1), generator=g).to(device)
    for i in range(steps):
        cur_pos = input_len + i
        pos = torch.full((batch, 1), cur_pos, dtype=torch.long, device=device)
        with torch.profiler.record_function(f"decode_step_{i}"):
            if MARK_STEP:
                MARK_STEP()
            out = step_model(input_ids=tok, position_ids=pos,
                             cache_position=torch.tensor([cur_pos], device=device),
                             attention_mask=attention_mask,
                             past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()


def profile_one(name, fn, steps=0):
    """跑一次 fn，套 profiler；导出表/trace/指标，返回 metrics dict。"""
    RESULT_SUBDIR.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    table = prof.key_averages().table(sort_by="self_cuda_time_total",
                                      row_limit=40)
    (RESULT_SUBDIR / f"{name}.table.txt").write_text(table, encoding="utf-8")
    trace_path = str(RESULT_SUBDIR / f"{name}.trace.json")
    prof.export_chrome_trace(trace_path)
    metrics = compute_metrics(trace_path, wall, steps)
    print(f"--- {name} ---")
    for k, v in metrics.items():
        if k != "top_kernels_by_total_us":
            print(f"    {k}: {v}")
    print("    top kernels (name, count, total_us):")
    for n, c, us in metrics["top_kernels_by_total_us"]:
        print(f"      {us/1e3:9.1f} ms x{c:<6} {n[:110]}")
    top = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12)
    print(top)
    return metrics


def build_model(model_path, device):
    """加载模型。

    本机（办公笔记本）提交内存吃紧时，Windows 下 from_pretrained 对
    safetensors 的整体 mmap 会抛 os error 1455（页面文件太小）；回退到
    stream_load.load_hf_model：纯顺序读 + 逐参数上卡，峰值内存只占单
    tensor 大小（仅限性能测量用途，见 stream_load.py 文档）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = dict(torch_dtype=torch.float16, attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model = model.to(device).eval()
    except OSError as e:
        print(f"[build_model] 常规加载失败({e})，改用流式低内存加载 ...", flush=True)
        from stream_load import load_hf_model
        model = load_hf_model(model_path, device)
    return model, tokenizer


def make_cache(model, device, batch, total_len):
    from transformers import StaticCache
    return StaticCache(config=model.config, max_batch_size=batch,
                       max_cache_len=total_len, device=device,
                       dtype=torch.float16)


@torch.inference_mode()
def reset_cache(cache):
    # cache buffer 在 inference_mode 下创建/写入，reset() 原位清零也必须在
    # 同一模式下（否则报 "Inplace update to inference tensor"）。
    cache.reset()


def variant_eager(model, tokenizer, device, input_len, steps):
    """层1 eager：prefill/decode 都是原模型。"""
    out = {}
    cache = make_cache(model, device, 1, input_len + steps)
    # 先跑一次预热（cuBLAS workspace/选核、lazy init 不能混进采集窗口）
    run_prefill(model, tokenizer, device, 1, input_len, cache, "warm", out_shape=steps)
    reset_cache(cache)
    out["prefill"] = profile_one("eager_prefill",
                                 lambda: run_prefill(model, tokenizer, device, 1, input_len, cache, "prefill", out_shape=steps))
    cache = make_cache(model, device, 1, input_len + steps)
    run_prefill(model, tokenizer, device, 1, input_len, cache, "warm", out_shape=steps)
    out["decode"] = profile_one("eager_decode",
                                lambda: run_decode_steps(model, tokenizer, device, 1, input_len, steps, cache),
                                steps=steps)
    return out


def variant_compiled(model, tokenizer, device, input_len, steps):
    """层2 cudagraphs：只编译 decode（与 bench_2 同设置，auto 降级）。"""
    from llm_bench import apply_inductor_workaround
    apply_inductor_workaround()
    cap = torch.cuda.get_device_capability(0)
    backend = "cudagraphs" if cap < (7, 0) else "inductor"
    print(f"[compile] backend={backend} (SM{cap[0]}.{cap[1]})")
    global MARK_STEP
    compile_model = torch.compile(model, backend=backend, dynamic=False)
    if backend == "cudagraphs":
        from torch._inductor import cudagraph_trees  # noqa: F401
        MARK_STEP = torch.compiler.cudagraph_mark_step_begin

    out = {}
    # 编译/图捕获放进 warmup，不进 profiler（与 bench_2 相同：prefill 走 eager）
    cache = make_cache(model, device, 1, input_len + steps)
    run_prefill(model, tokenizer, device, 1, input_len, cache, "warm", out_shape=steps)
    run_decode_steps(compile_model, tokenizer, device, 1, input_len, 2, cache)
    # 采集窗口只包含 decode 循环：prefill 在 profiler 外完成，不污染每步指标
    run_prefill(model, tokenizer, device, 1, input_len, cache, "p", out_shape=steps)
    torch.cuda.synchronize()
    out["decode"] = profile_one(
        "compiled_decode",
        lambda: run_decode_steps(compile_model, tokenizer, device, 1, input_len, steps, cache),
        steps=steps)
    return out


def main():
    p = argparse.ArgumentParser(description="Operator-level bottleneck attribution via PyTorch Profiler")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--variant", default="both", choices=["eager", "compiled", "both"])
    p.add_argument("--input-len", type=int, default=128, help="prefill 输入长度（workload A）")
    p.add_argument("--steps", type=int, default=30, help="decode 采集步数")
    args = p.parse_args()

    global RESULT_SUBDIR
    RESULT_SUBDIR = RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARN: 无 CUDA，profiler 只有 CPU 事件，busy 指标无意义")
    model, tokenizer = build_model(args.model, device)

    summary = {"model": args.model, "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "input_len": args.input_len, "decode_steps": args.steps}
    if args.variant in ("eager", "both"):
        summary["eager"] = variant_eager(model, tokenizer, device, args.input_len, args.steps)
    if args.variant in ("compiled", "both"):
        try:
            summary["compiled"] = variant_compiled(model, tokenizer, device, args.input_len, args.steps)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            summary["compiled"] = {"skipped_reason": f"CUDA OOM: {str(e)[:200]}"}
            print(f"[compiled] 显存不足，跳过（本机 2GB 卡常见）：{str(e)[:200]}")

    out = RESULT_SUBDIR / "prof_metrics.json"
    prev = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    merged = {**prev, **summary}
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"metrics -> {out}")


RESULT_SUBDIR = RESULTS_DIR

if __name__ == "__main__":
    main()
