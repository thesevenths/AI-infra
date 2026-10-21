"""
measurement.py -- Experiment 2 (Inference) 公共测量库

四个指标的统一实现，所有 bench 脚本共用，保证跨方法可比：
  - TTFT       : time-to-first-token，prefill 一次 forward 的耗时（ms）
  - TPOT       : time-per-output-token，decode 阶段每 token 平均耗时（ms）
  - Throughput : output tokens/s（系统级 = 并发数 x 单请求速率）
  - VRAM       : 显存峰值（torch allocator 记账 + nvidia-smi 外部读数）

测量方法：
  不用 generate() 黑盒，而是手动 prefill + 逐 token decode 循环，
  以 perf_counter + cuda.synchronize 打点，得到真实的 TTFT/TPOT。

负载设定：
  workload A: 短入长出 (in=128,  out=128)
  workload B: 长入短出 (in=512,  out=32)   # 长度针对 2GB 入门级显卡做了缩减
  并发（batch）按 --concurrency 扫描。
"""

import json
import statistics
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

DEFAULT_MODEL = "F:/model/Qwen-0.6B"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# 两个固定负载：短入长出 / 长入短出
WORKLOADS = {
    "A_short_in_long_out": {"input_len": 128, "output_len": 128},
    "B_long_in_short_out": {"input_len": 512, "output_len": 32},
}


def sync_wall(fn, *args, **kwargs):
    """计时一个 GPU 工作负载：前后 sync，墙钟时间才可信（CPU 下为 no-op）。"""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if cuda:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0)


def nvidia_smi_used_mib():
    """外部显存读数（含 CUDA context，torch allocator 看不到的那部分）。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return max(int(x) for x in r.stdout.strip().splitlines() if x.strip().isdigit())
    except Exception:
        return -1


@dataclass
class RunResult:
    method: str
    workload: str
    concurrency: int
    repeats: int
    ttft_ms: float = 0.0            # p50
    ttft_p95_ms: float = 0.0
    tpot_ms: float = 0.0            # p50
    tpot_p95_ms: float = 0.0
    throughput_tps: float = 0.0     # output tokens/s（系统级）
    vram_alloc_peak_mib: float = 0.0    # torch allocator 记账
    vram_smi_peak_mib: float = 0.0      # nvidia-smi 看到的进程总占用
    extras: dict = field(default_factory=dict)

    def add_sample(self, ttft_ms, tpot_ms, wall_s, batch, out_tokens):
        self._ttfts = getattr(self, "_ttfts", [])
        self._tpots = getattr(self, "_tpots", [])
        self._ttfts.append(ttft_ms)
        self._tpots.append(tpot_ms)
        # 系统吞吐：batch 条请求各自产出 out_tokens
        self._thpts = getattr(self, "_thpts", [])
        self._thpts.append(batch * out_tokens / wall_s)

    def finalize(self):
        def p50(xs): return statistics.median(xs)
        def p95(xs):
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(0.95 * len(xs)))]
        self.ttft_ms, self.ttft_p95_ms = p50(self._ttfts), p95(self._ttfts)
        self.tpot_ms, self.tpot_p95_ms = p50(self._tpots), p95(self._tpots)
        self.throughput_tps = p50(self._thpts)
        if torch.cuda.is_available():
            self.vram_alloc_peak_mib = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        return self


# ---------------------------------------------------------------- HF 测量内核
# 层1/层2 共用的手动 prefill+decode。两个关键设计决策：
#  1) 使用 StaticCache 而非 DynamicCache。DynamicCache 每步以 torch.cat
#     增长缓存张量，其"新地址输出"模式与 torch.compile cudagraph trees 的
#     generation 复写机制不兼容；StaticCache 预分配、原位写入，跨步地址恒定。
#  2) prefill 始终走 eager 模型，decode 步才交给（可选的）decode_model。
#     StaticCache 的缓存 buffer 在首次 eager update 时分配并被标记为静态
#     地址，因此编译 prefill 与 CUDA Graphs 组合不可靠；业界 serving 框架
#     同样只对 decode 图化。代价是编译收益只体现在 TPOT，TTFT 不变。
# cache 可由调用方按 (batch, total_len) 复用（reset 原位清零、地址不变），
# 避免 cudagraph trees 因新地址触发重录。

def make_input(tokenizer, device, batch, input_len, seed=0):
    """构造 batch x input_len 的固定长度输入（随机 token，长度即负载）。"""
    g = torch.Generator().manual_seed(seed)
    vocab = tokenizer.vocab_size
    ids = torch.randint(1000, vocab - 10, (batch, input_len), generator=g)
    return ids.to(device)


@torch.inference_mode()
def hf_measure(model, tokenizer, device, batch, input_len, output_len, seed=0,
               decode_models=None, caches=None):
    """测量一次「prefill + 固定步数 decode」，返回 (ttft_ms, tpot_ms, wall_s)。

    batch 内等长输入，attention_mask 全程为全长 1s（静态 shape）。

    decode_models / caches：可选的 (batch, total_len) -> 对象 映射，
    层2 为每个 shape 准备独立 compile wrapper 与独立 StaticCache；
    缺省时直接使用 model 并为本次调用新建 cache（层1 路径）。
    """
    from transformers import StaticCache

    total_len = input_len + output_len
    input_ids = make_input(tokenizer, device, batch, input_len, seed)
    attention_mask = torch.ones(batch, total_len, dtype=torch.long, device=device)
    cache = None if caches is None else caches.get((batch, total_len))
    if cache is None:
        cache = StaticCache(config=model.config, max_batch_size=batch,
                            max_cache_len=total_len, device=device,
                            dtype=torch.float16)
    else:
        cache.reset()   # 复用同一组 buffer，保持 data_ptr 不变

    # cudagraph trees 参与执行时（层2），每个 generation 开始前必须
    # mark_step_begin，否则图节点内存随 decode 步数单调增长；
    # eager 路径（层1）不涉及图，无需该调用。
    mark = (torch.compiler.cudagraph_mark_step_begin
            if "torch._inductor.cudagraph_trees" in sys.modules else None)
    pos = torch.arange(input_len, device=device).unsqueeze(0).expand(batch, -1)
    if mark:
        mark()
    out, ttft_s = sync_wall(
        model, input_ids=input_ids, attention_mask=attention_mask,
        position_ids=pos.contiguous(), cache_position=torch.arange(input_len, device=device),
        past_key_values=cache, use_cache=True,
        # 只算末位 logits（greedy 采样只用它），与 vLLM 等生产框架一致；
        # 全序列 logits 在长输入下会产生巨大的瞬时显存峰值。
        logits_to_keep=1,
    )
    logits = out.logits if hasattr(out, "logits") else out[0]
    next_tok = logits[:, -1:, :].argmax(-1)

    step_model = (decode_models or {}).get((batch, total_len), model)
    step_times = []
    for i in range(output_len - 1):
        cur_pos = input_len + i
        pos = torch.full((batch, 1), cur_pos, dtype=torch.long, device=device)
        if mark:
            mark()
        out, dt = sync_wall(
            step_model, input_ids=next_tok, position_ids=pos,
            cache_position=torch.tensor([cur_pos], device=device),
            attention_mask=attention_mask,
            past_key_values=cache, use_cache=True,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        next_tok = logits[:, -1:, :].argmax(-1)
        step_times.append(dt)

    ttft_ms = ttft_s * 1e3
    tpot_ms = statistics.mean(step_times) * 1e3 if step_times else 0.0
    return ttft_ms, tpot_ms, ttft_s + sum(step_times)


def warmup(model, tokenizer, device, shapes, decode_models=None, caches=None):
    """预热：触发 lazy init / JIT / torch.compile。不计入任何指标。"""
    print(f"  warmup {len(shapes)} shape(s) ...", flush=True)
    for (b, i, o) in shapes:
        hf_measure(model, tokenizer, device, b, i, o, seed=99,
                   decode_models=decode_models, caches=caches)


# ---------------------------------------------------------------- 兼容性补丁

class _DuplicateTemplateTolerantFinder:
    """容忍 torch 2.14.x Windows 安装中的 Triton 模板重名问题。

    该发行版的 _inductor/kernel/ 下新旧两套文件以相同模板名、不同源码
    各自注册，import 时断言 'duplicate template name'。本钩子包住
    torch._inductor.kernel.* 模块的 exec，在其执行期间临时容忍重名：
    源码哈希不同的后注册者改名 '__dup_ok'，模块内其余 lowering/符号
    照常可用。被改名的均为冷门模板，不影响本实验的 SDPA 路径。"""

    _busy = False

    def find_spec(self, fullname, path=None, target=None):
        import hashlib
        import importlib.util

        if not fullname.startswith("torch._inductor.kernel") or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None

        outer_exec = spec.loader.exec_module

        def exec_module(module):
            sa = sys.modules.get("torch._inductor.select_algorithm")
            TT = getattr(sa, "TritonTemplate", None) if sa else None
            if TT is None:
                return outer_exec(module)
            orig_init = TT.__init__

            def tolerant(self_, name, grid=None, source=None, *a, **kw):
                if source is None:
                    source = kw.get("source")
                existing = TT.all_templates.get(name)
                if existing is not None and isinstance(source, str):
                    h = hashlib.sha256(source.encode("utf-8")).hexdigest()
                    if existing.src_hash != h:
                        name = f"{name}__dup_ok"
                return orig_init(self_, name, grid, source, *a, **kw)

            TT.__init__ = tolerant
            try:
                return outer_exec(module)
            finally:
                TT.__init__ = orig_init

        spec.loader.exec_module = exec_module
        return spec


def apply_inductor_workaround():
    """安装钩子并完成 torch._inductor 的首次导入；幂等。"""
    import importlib

    if "torch._inductor.compile_fx" in sys.modules:
        return "already-imported"
    if not any(isinstance(f, _DuplicateTemplateTolerantFinder)
               for f in sys.meta_path):
        sys.meta_path.insert(0, _DuplicateTemplateTolerantFinder())
    importlib.import_module("torch._inductor.compile_fx")
    return "hook-installed"


# ---------------------------------------------------------------- 运行框架

def build_parser(method_name, epilog=""):
    p = ArgumentParser(
        description=f"Experiment 2 / {method_name} benchmark", epilog=epilog,
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--concurrency", default="1,4",
                   help="逗号分隔的并发(batch)列表")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workloads", default="A,B", help="A, B 或 A,B")
    return p


def run_suite(method_name, model, tokenizer, device, args, warm_shapes=None,
              decode_models=None, caches=None):
    """对每个 workload x 并发跑 repeats 次，产出 RunResult 表并落盘。"""
    concurrencies = [int(x) for x in args.concurrency.split(",")]
    selected = []
    if "A" in args.workloads:
        selected.append(("A_short_in_long_out", WORKLOADS["A_short_in_long_out"]))
    if "B" in args.workloads:
        selected.append(("B_long_in_short_out", WORKLOADS["B_long_in_short_out"]))

    # 显存基线（模型已加载，权重已计入 allocator 记账）
    base_alloc = torch.cuda.memory_allocated() / 2**20 if device == "cuda" else 0
    print(f"[{method_name}] weights allocated: {base_alloc:.0f} MiB")

    results = []
    for wname, wl in selected:
        for b in concurrencies:
            torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
            res = RunResult(method=method_name, workload=wname,
                            concurrency=b, repeats=args.repeats)
            for r in range(args.repeats):
                ttft, tpot, wall = hf_measure(
                    model, tokenizer, device, b,
                    wl["input_len"], wl["output_len"], seed=r,
                    decode_models=decode_models, caches=caches,
                )
                res.add_sample(ttft, tpot, wall, b, wl["output_len"])
                smi = nvidia_smi_used_mib()
                res.vram_smi_peak_mib = max(res.vram_smi_peak_mib, smi)
            results.append(res.finalize())
            print(f"  {wname} batch={b}: TTFT {res.ttft_ms:.1f}ms | "
                  f"TPOT {res.tpot_ms:.1f}ms | {res.throughput_tps:.1f} tok/s | "
                  f"VRAM(smi) {res.vram_smi_peak_mib:.0f}MiB", flush=True)

    save_and_print_table(method_name, results)
    return results


def save_and_print_table(method_name, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = [asdict(r) for r in results]
    for d in out:  # 清掉私有字段
        for k in list(d):
            if k.startswith("_"):
                del d[k]
    path = RESULTS_DIR / f"{method_name}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================= 结果汇总 =================")
    hdr = f"{'workload':<22}{'batch':>5}{'TTFT p50':>10}{'TPOT p50':>10}" \
          f"{'tok/s':>9}{'VRAM MiB':>10}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r.workload:<22}{r.concurrency:>5}{r.ttft_ms:>9.1f}ms"
              f"{r.tpot_ms:>9.1f}ms{r.throughput_tps:>9.1f}"
              f"{r.vram_smi_peak_mib:>10.0f}")
    print(f"saved -> {path}")
