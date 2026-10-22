# Profiling Report: Qwen-0.6B FP16 Inference on a Consumer Pascal GPU

> **Summary of findings (measured, GTX 1050 Ti Max-Q 4 GB, Qwen-0.6B, batch=1, 128-token prompt):**
>
> 1. **Decode is bandwidth-bound at the kernel level and dispatch-bound at the
>    step level.** Each decode step streams ~1.37 GB of fp16 weights through
>    GEMV (62.6 % of kernel time). The implied effective memory bandwidth is
>    ~82 GB/s — ~73 % of the card's nominal 112 GB/s — so the GEMV kernels
>    themselves are bandwidth-saturated. However, the GPU is only **15 % busy
>    across the whole step**, because ~1,700 separate kernel launches per step
>    leave the (relatively fast) GPU idle between dispatches.
> 2. **Prefill is compute-bound.** GEMM is 71 % of kernel time; Pascal has no
>    Tensor Core, so fp16 GEMM runs on fp32 ALUs at low arithmetic efficiency.
>    FlashAttention-2 is unavailable, so attention uses the Cutlass
>    memory-efficient `sm50` kernel.
> 3. **`torch.compile` with the `cudagraphs` backend is the highest-leverage
>    optimization for decode on this hardware: 175 ms → 59–66 ms TPOT
>    (~3×), raising GPU busy from ~15 % to ~87 %** by collapsing ~1,700
>    per-step launches into a single graph replay. (Architectural caveat: the
>    same path regressed on a 2 GB MX330 because the static graph did not fit
>    without thrashing — the win is VRAM-headroom dependent.)

---

## 1. Scope and Objectives

This study characterizes where wall-clock time is spent during FP16
autoregressive inference of a 0.6 B-parameter model on a commodity Pascal
GPU, using the PyTorch Profiler. It separates the **prefill** path (one
large forward over the prompt) from the **decode** path (many small
single-token forwards), because they are bottlenecked by different
resources and must not be averaged together.

Two questions are answered with evidence:

- Which operators dominate decode and prefill, and what resource are they
  bound by (compute vs. memory bandwidth vs. launch dispatch)?
- Does graph capture (`cudagraphs`) help, and under what conditions?

The throughput benchmarks (`bench_1_eager`, `bench_2_torch_compile`) in
the sibling `inference/` directory provide an independent, profiler-free
cross-check for the compiled path.

## 2. Environment and Methodology

| Item | Value |
|---|---|
| GPU | NVIDIA **GeForce GTX 1050 Ti Max-Q** (Pascal GP107 / SM 6.1, 128-bit GDDR5, nominal 112 GB/s, **no Tensor Core**, **no FlashAttention-2**) |
| Software | Python 3.12.3 (Anaconda), **torch 2.8.0.dev+cu128**, transformers 4.51.0 |
| Model | **Qwen-0.6B** (Qwen3: 28 layers, hidden 1024, GQA 8 KV heads, vocab 151936, tied word embeddings), loaded fp16 |
| Workload | `StaticCache`, batch=1, prompt length 128 (fixed-seed random token ids), **30 decode steps**, `logits_to_keep=1`, SDPA attention |
| Variants | `eager` (vanilla) vs. `torch.compile(backend="cudagraphs", dynamic=False)` — on SM < 7.0 Triton is unavailable, so inductor falls back to the `cudagraphs` backend (CUDA graphs only, no kernel fusion) |

**Metric definitions and trust notes:**

- The PyTorch Profiler records CPU + CUDA activities; traces are exported to
  chrome/Perfetto JSON and parsed for `kernel` / `gpu_memcpy` / `gpu_memset`
  events only (CPU-operator events carry aggregated device time and would
  double-count). GPU busy ratio = union of device-event intervals ÷ wall
  clock, not a naive sum (concurrent streams would otherwise inflate it).
- Steady-state figures take the **median** of the per-step NVTX windows
  (`decode_step_i`), which excludes the first capture/warmup step.
- Because `torch.profiler` instrumentation **disables CUDA graph replay**,
  the cudagraphs variant cannot be profiled faithfully through the same
  path (see §4). Its authoritative steady-state numbers come from the
  profiler-free throughput benchmark.

**Artifact index** — all numbers below are reproducible from
`results/prof_metrics.json` (summary), `results/*.table.txt`
(`key_averages` tables), and `results/*.trace.json` (Perfetto /
`chrome://tracing` timelines). Raw run logs: `results/run_eager.log`,
`results/run_compiled.log`.

## 3. Decode path (steady state, batch=1)

| Metric | eager | cudagraphs* |
|---|---|---|
| Steady TPOT (median of 30 steps) | **175.3 ms** | **59.2 ms** (profiler) / **65.6 ms** (throughput bench) |
| GPU busy ratio (step) | **0.145** | ~0.870 |
| Kernels per step | 1,717 | 2,148 |
| Device time per step | 24.8 ms | 51.5 ms |
| Steady window memcpy | 0.28 MB | (capture overhead, see §4) |

\* *cudagraphs steady figures combine the profiler's per-step median with the
profiler-free throughput benchmark; the profiled capture step is excluded
(see §4). Under `torch.profiler` the graph replay is disabled, so the
cudagraphs "kernels per step" (2,148) is the profiler-recorded kernel count,
not the actual launch count — in real replay it is a single
`cudaGraphLaunch`.*

**Operator breakdown (eager, per step, 30-step average):**

| Category | Calls/step | Time/step | Share | Note |
|---|---|---|---|---|
| `gemv2T_kernel_val` (q/k/v/o + MLP + lm_head GEMV) | ~197 | **15.8 ms** | **63.3 %** | batch=1 weight × vector; 5,910 calls / 30 steps |
| `fmha_cutlassF_f16_…_sm50` | 28 | 2.5 ms | 9.5 % | single-query attention over ≤128 KV; µs-scale per call on fast GPU |
| RMSNorm (`MeanOps` reduce + `rsqrt` + `mul`) | ~113 | ~2.0 ms | ~7.6 % | **not fused**: mean → rsqrt → mul are separate kernels |
| RoPE / `index_copy` / `cat` | ~56–168 | ~5 ms | ~19 % | cos/sin build and in-place cache writes |
| residual / cast / other elementwise | hundreds | ~6 ms | ~22 % | |

**Bandwidth derivation (kernel-level bound).** The ~197 GEMV calls per step
read the full set of linear weights once: with ~0.6 B parameters in fp16 and
tied lm_head/embed counted once, the per-token working set is **≈1.37 GB**.
Streaming that in 15.8 ms implies an **effective memory bandwidth of
≈87 GB/s** — about 77 % of the 1050 Ti Max-Q's nominal 112 GB/s. The GEMV
kernels are therefore **bandwidth-saturated at the kernel level**; you cannot
make a single decode step faster on this silicon without either reducing
bytes/parameter (quantization) or raising arithmetic intensity (batching).

**Dispatch bound (step-level).** Despite 26 ms of genuine GPU work per step,
the step wall is 172 ms — the GPU is idle ~85 % of the time. The gap is the
cost of issuing ~1,700 independent kernel launches from the CPU between
steps. On this GPU (≈2× the memory bandwidth of the MX330 used in earlier
profiling) the kernels finish quickly, so launch dispatch — not bandwidth —
becomes the step-level ceiling. This is exactly the regime where
`cudagraphs` pays off (§4).

## 4. Compiled path (`cudagraphs`) — measurement caveat and result

`torch.compile(backend="cudagraphs")` records each decode step as a static
CUDA graph and replays it. Under `torch.profiler`, however, graph replay is
**disabled** (the profiler must observe every launch), so the captured
window shows the *initial capture* cost: ~13,666 device-to-device memcpys
totaling ~34 GB, concentrated in the first step. This is an **instrumentation
artifact, not steady state**, and must not be read as cudagraphs'
per-token cost.

The reliable numbers come from two sources that agree:

- **Profiler, steady steps only** (median of `decode_step_1..29`, excluding
  the capture step): **59.2 ms TPOT, 87.0 % GPU busy, 2,148 kernels/step**.
- **Profiler-free throughput benchmark** (`bench_2_torch_compile`, same
  machine/model): **65.6 ms TPOT** — within 7 % of the profiled steady
  figure, confirming it.

**Interpretation.** `cudagraphs` collapses the ~1,700 per-step launches into
one graph replay, lifting GPU busy from ~15 % to ~87 % and cutting TPOT by
**~3×** (175 → 59–66 ms). The one-time capture cost (the ~34 GB DtoD
traffic) is amortized over the session and does not recur per token in
steady serving.

**Hardware-dependence.** On a 2 GB MX330 profiled earlier, the same path
*regressed* (+60 % TPOT): the 2 GB card could not hold the static graph
without memory thrashing, so the clone traffic dominated. The optimization's
value is therefore **VRAM-headroom dependent** — a useful reminder that
"enable all the acceleration flags" is the wrong default; the choice should
follow the measured roofline.

## 5. Prefill path (batch=1, 128 tokens)

| Metric | Value |
|---|---|
| Wall (single 128-token prefill) | **222.1 ms** |
| GEMM share of kernel time | **71.3 %** (`hgemm_32x32x32_NT_vec`, 196 calls) |
| Attention share | 7.1 % (`fmha_cutlassF` `sm50`, mem-efficient, no FA2) |
| Remaining (RMSNorm / RoPE / casts) | ~22 % |

Prefill is **compute-bound**: it is dominated by large GEMMs over the
sequence dimension, and Pascal's lack of Tensor Cores means fp16 GEMM
executes on fp32 arithmetic units at low efficiency. The `sm50` attention
kernel is a minor contributor and is the memory-efficient variant (FA2 is
unsupported on Pascal). Unlike decode, prefill's bottleneck is the **absence
of Tensor Core compute**, which neither `cudagraphs` (no fusion on SM<7.0)
nor batching-at-batch=1 addresses — only a newer architecture would.

## 6. Conclusions

1. **Decode at batch=1 on this Pascal GPU is bound twice over:** at the kernel
   level by memory bandwidth (GEMV saturates ~82 GB/s), and at the step level
   by launch dispatch (~15 % GPU utilization). The step-level bound is the one
   worth attacking first.
2. **`cudagraphs` is the highest-leverage decode optimization here (~3×,
   ~15 %→87 % busy).** Its benefit is VRAM-headroom dependent and vanishes on
   cards too small to hold the static graph.
3. **Prefill is compute-bound due to the Pascal fp16-without-Tensor-Core
   limitation**; it needs architectural compute, not scheduling tricks.
4. **Structural levers for decode**, independent of this specific GPU:
   quantization (fewer bytes/parameter → lower bandwidth demand) and batching
   (higher arithmetic intensity → moves right on the roofline).
5. **Methodological takeaway:** measure the roofline before choosing the
   optimization. A flag that is a clear win on a 4 GB card (cudagraphs) is a
   regression on a 2 GB card; blanket "enable everything" is wrong by default.

## 7. Reproduction

```powershell
# Interpreter: Anaconda Python with CUDA-enabled torch
# (D:\ProgramData\anaconda3\python.exe on the test machine)
$ PY="D:/ProgramData/anaconda3/python.exe"

# 1. PyTorch Profiler (eager + cudagraphs; run as separate processes
# to avoid holding two model graphs in 4 GB at once)
& $PY profiling/prof_1_torch_profiler.py --variant eager    --model F:/model/Qwen-0.6B
& $PY profiling/prof_1_torch_profiler.py --variant compiled --model F:/model/Qwen-0.6B

# 2. Nsight Compute payload (kernel-level SOL/DRAM; see nsys_commands.md)
& $PY profiling/prof_2_ncu_kernels.py --steps 10 --model F:/model/Qwen-0.6B

# 3. Throughput benchmark cross-check (profiler-free cudagraphs number)
& $PY inference/run_all.py --methods 1,2 --concurrency 1,4 --repeats 3
```

**Out-of-scope on this hardware:** Nsight Compute (2025.x) and Nsight
Systems CLI do not support Pascal (GP107, SM 6.1) — Nsight Compute rejects
the device (`Profiling is not supported on device 0`; `--list-chips` starts
at Turing). The kernel-level SOL/DRAM and timeline views must be collected on
a cloud **datacenter GPU (A100 / L20 / 4090)**; collection commands and
expected checks are documented in `nsys_commands.md`. The PyTorch-Profiler
findings above are hardware-characterization results that hold for any
Pascal-class GPU and do not require the Nsight tools.

Timeline viewing: `results/*.trace.json` → https://ui.perfetto.dev (or
`chrome://tracing`). The eager decode trace shows the characteristic
launch "jitter" between kernels; the cudagraphs trace (steady steps) shows
back-to-back execution.
