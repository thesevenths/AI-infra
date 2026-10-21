# 更新日志 / Changelog

## 2026-10-21 — 换机迁移：MX330 2GB → GTX 1050 Ti Max-Q 4GB

### 背景
环境对比（非纯硬件对照，模型也同步升级）：

| 项 | 旧机（基线） | 新机（本次） |
|---|---|---|
| GPU | NVIDIA MX330 2GB（Pascal SM6.1，64-bit 显存位宽） | NVIDIA GTX 1050 Ti Max-Q 4GB（Pascal SM6.1，128-bit 显存位宽） |
| 模型 | Qwen1.5-0.5B（24 层，词表 151851） | Qwen-0.6B（Qwen3 架构，28 层 / hidden 1024 / 8 KV heads / 词表 151936，权重 fp16 ≈ 1145 MiB） |
| Python / torch / transformers | 3.13 / torch 2.14 / transformers 4.57（仓库目标栈） | Anaconda base 3.12.3 / torch 2.8.0.dev+cu128 / transformers 4.51.0 |
| 运行解释器 | — | `D:\ProgramData\anaconda3\python.exe`（系统自带 CUDA 版 torch） |


### 运行方式
```bash
# 用本机 Anaconda 的 CUDA 版 torch（不要用 workbuddy 隔离 venv 里那个 CPU-only torch）
D:/ProgramData/anaconda3/python.exe run_all.py --methods 1,2 --concurrency 1,4 --repeats 3
```

### 本次结果（1050 Ti 4GB / Qwen-0.6B）
| Workload | batch | 方法 | TTFT p50 | TPOT p50 | tok/s | VRAM (smi) |
|---|---|---|---|---|---|---|
| A (in128/out128) | 1 | 1_eager | 219.1 ms | 166.7 ms | 5.9 | 1864 MiB |
| A | 1 | 2_torch_compile (cudagraphs) | 201.0 ms | 65.6 ms | 15.0 | 3505 MiB |
| A | 4 | 1_eager | 419.2 ms | 227.6 ms | 17.5 | 1810 MiB |
| A | 4 | 2_torch_compile (cudagraphs) | 437.1 ms | 105.3 ms | 37.0 | 3509 MiB |
| B (in512/out32) | 1 | 1_eager | 456.8 ms | 148.7 ms | 6.3 | 1810 MiB |
| B | 1 | 2_torch_compile (cudagraphs) | 445.4 ms | 67.8 ms | 12.6 | 3512 MiB |
| B | 4 | 1_eager | 1788.6 ms | 228.4 ms | 14.4 | 2046 MiB |
| B | 4 | 2_torch_compile (cudagraphs) | 1859.4 ms | 131.5 ms | 21.8 | 3588 MiB |

### 与旧基线对比（旧 MX330 2GB / Qwen1.5-0.5B）
- 层 1（eager）：tok/s 全面 ↑ 34%–380%（TTFT 普遍 ↓ 83%–92%）。
- 层 2（torch.compile/cudagraphs）：tok/s 全面 ↑ 409%–2248%，**逆转了旧机上层 2 慢于层 1 的现象**。
- 根因：4GB 显存让 cudagraphs 的静态图捕获（≈3500 MiB）不再 OOM/交换；1050 Ti 的 128-bit 位宽（MX330 64-bit）约 2× 带宽进一步拉高 decode 吞吐。
- 注意：模型也换了（0.5B→0.6B，28 vs 24 层），故非纯硬件对照；但层内排序两次一致。
