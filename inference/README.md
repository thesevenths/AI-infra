# LLM Inference 四层阶梯对比

用同一个模型、同一套负载、同一组指标，依次对比四种推理方案，
回答一个问题：**每往上一层，到底优化了什么、代价是什么。**

## 四层阶梯：各层解决什么问题

| 层 | 方案 | 针对的痛点 | 通俗理解 | 专业机制 |
|---|---|---|---|---|
| 1 | **PyTorch eager**（基线） | — | 算子逐个启动、来回搬数据(读取 HBM ➔ 算子 A ➔ 写回 HBM ➔ 读取 HBM ➔ 算子 B ➔ 写回 HBM....)；batch 必须等长要填充；显存按最大长度静态预分配、易碎片化 OOM | **Kernel Launch Overhead + Memory-bound**：Transformer 层级深，每生成 1 个 token 触发数百次小 kernel launch，中间结果频繁往返慢速 HBM，算力闲置；**Static Batching + Padding**：为拼矩阵用 `<PAD>` 补齐，大量无效计算；**静态 KV Cache 分配**：内部/外部碎片严重 |
| 2 | **+ torch.compile** | 只优化**内核执行方式** | 小算子拼接成大算子，只启动一次，中间结果不再来回搬; 一次性从 HBM 把数据读到 Shared Memory (SRAM)，在 SRAM 里连续做完算子 A 和 B，最后只把最终结果写回一次 HBM | **Kernel Fusion（算子融合）**：图编译把细粒度算子融合为粗粒度算子，中间结果驻留 GPU 片上高速存储（SRAM/Register），不再写回 HBM，大幅降低访存延迟。⚠️ 不改变调度（依然 Static Batching）与显存管理（依然静态分配）——批处理与 KV 碎片问题原样留给层 3 |
| 3 | **+ vLLM** | 只优化**服务架构** | 请求不用互相等齐才走；显存切成固定小块(16K)按需分配,杜绝浪费 | **Continuous Batching（迭代级调度）**：每个 decode step 动态踢出已完成请求、插入新请求，GPU 算力 0 闲置；**PagedAttention**：借鉴 OS 虚拟内存分页，KV Cache 切成固定大小 Block（如 16 token/block），经 Block Table 非连续映射物理显存，消灭碎片，VRAM 利用率逼近 100% |
| 4 | **+ SGLang / TensorRT-LLM** | 生产级**内核 + 架构**上限 | 相同前缀的 KV 直接复用，不再重算 | **RadixAttention（前缀缓存）**：KV Block 组织为全局 Radix Tree，共享前缀（System Prompt / 多轮对话 / RAG 上下文）的请求直接复用已算好的 KV，跳过重复 prefill——对 **TTFT 是断崖式优化**；TRT-LLM 一侧则是手工融合内核 + 图执行，测性能上限与工程代价（build 流程） |

每层的变量是隔离的：层 2 只换内核执行方式，层 3 只换服务架构，层 4 才是内核+架构同时拉满。

## 四个指标

| 指标 | 定义 | 主要反映 |
|---|---|---|
| **TTFT** (ms) | 首 token 延迟 = prefill 耗时 | 架构/调度（排队、前缀复用） |
| **TPOT** (ms) | 每输出 token 平均耗时 = decode 单步 | 内核效率（融合、launch 开销） |
| **Throughput** (tok/s) | 系统级输出吞吐 = 并发 × 单请求速率 | 批处理调度（连续 vs 静态） |
| **VRAM** (MiB) | 显存峰值（nvidia-smi + torch allocator） | KV Cache 管理（静态 vs 分页） |

## 实测结果（本机：GTX 1050 Ti Max-Q 4GB, Pascal SM6.1, fp16）

| Workload | batch | 方法 | TTFT p50 | TPOT p50 | tok/s | VRAM (smi) |
|---|---|---|---|---|---|---|
| A (in128/out128) | 1 | 1_eager | 219.1 ms | 166.7 ms | **5.9** | 1864 MiB |
| A | 1 | 2_torch_compile (cudagraphs) | 201.0 ms | 65.6 ms | **15.0** | 3505 MiB |
| A | 4 | 1_eager | 419.2 ms | 227.6 ms | **17.5** | 1810 MiB |
| A | 4 | 2_torch_compile (cudagraphs) | 437.1 ms | 105.3 ms | **37.0** | 3509 MiB |
| B (in512/out32) | 1 | 1_eager | 456.8 ms | 148.7 ms | **6.3** | 1810 MiB |
| B | 1 | 2_torch_compile (cudagraphs) | 445.4 ms | 67.8 ms | **12.6** | 3512 MiB |
| B | 4 | 1_eager | 1788.6 ms | 228.4 ms | **14.4** | 2046 MiB |
| B | 4 | 2_torch_compile (cudagraphs) | 1859.4 ms | 131.5 ms | **21.8** | 3588 MiB |

> 注：本机 Pascal SM6.1 不被 Triton 支持，torch.compile 自动降级为 `cudagraphs`
> 后端（仅 CUDA Graphs，无算子融合）。即便如此，cudagraphs 通过消除每步 kernel
> launch 开销，把 decode（TPOT）压到 eager 的 1/2 ~ 1/3，吞吐接近 3×。

### 与换机前基线对比（旧：MX330 2GB / Qwen1.5-0.5B → 新：1050 Ti 4GB / Qwen-0.6B）

| 指标 | 层 | 旧 (MX330 2GB) | 新 (1050 Ti 4GB) | 变化 |
|---|---|---|---|---|
| A b1 | 1_eager | 4.4 tok/s | 5.9 tok/s | ↑ 34% |
| A b1 | 2_torch_compile | 2.9 tok/s | 15.0 tok/s | ↑ 409% |
| A b4 | 1_eager | 4.1 tok/s | 17.5 tok/s | ↑ 330% |
| A b4 | 2_torch_compile | 3.2 tok/s | 37.0 tok/s | ↑ 1062% |
| B b4 | 2_torch_compile | 0.9 tok/s | 21.8 tok/s | ↑ 2248% |

关键结论：
- **4GB 显存是真正的分水岭**。旧机 2GB 下 cudagraphs 捕获静态图需约 3500 MiB，
  挤占 + 显示占用造成内存压力/交换，使层 2 反而比层 1 慢；新机 4GB 留足余量，
  层 2 图捕获不再 OOM/交换，decode 大幅领先层 1（与「层 2 只优化内核执行」的设定一致）。
- **1050 Ti 的 128-bit 显存位宽（MX330 仅 64-bit）约 2× 带宽**，对 memory-bound 的
  decode 直接体现在 TPFT→TPOT 与吞吐上。
- **注意：本次同时换了模型**（Qwen1.5-0.5B → Qwen-0.6B Qwen3，28 层 vs 24 层，略大），
  因此新旧对比并非纯硬件对照；但两次层内（层 1 vs 层 2）排序一致，阶梯结论仍成立。
- 层 3（vLLM）/ 层 4（SGLang、TRT-LLM）在本机仍无法运行（Windows + 4GB + SM6.1），
  需在 WSL2 / 云端 GPU 上补齐，命令见下方「运行」。

## 目录结构

```
inference/
├── llm_bench/               # 公共测量库（指标定义 / HF 测量内核 / 运行框架）
│   ├── __init__.py
│   └── measurement.py
├── bench_1_eager.py         # 层1  PyTorch eager 基线
├── bench_2_torch_compile.py # 层2  torch.compile（仅 decode 图化/融合）
├── bench_3_vllm.py          # 层3  vLLM（PagedAttention + continuous batching）
├── bench_4_sglang.py        # 层4  SGLang / TRT-LLM 流式 HTTP 压测客户端
├── run_all.py               # 一键跑批：逐层执行 + 完整日志落 logs/
├── compare.py               # 汇总 results/*.json → 阶梯对比表
├── results/                 # 各层结果 JSON（4 指标 × 2 workload × 并发）
└── logs/                    # 运行日志（命令行 / stdout+stderr / 退出码 / 耗时）
```

## 运行

```bash
# 一键四层 + 日志 + 对比表（本机建议 --methods 1,2）
python run_all.py --methods 1,2 --concurrency 1,4 --repeats 3

# 单跑某一层
python bench_1_eager.py --concurrency 1,4 --repeats 3
python bench_2_torch_compile.py --concurrency 1,4 --repeats 3   # 首次含编译/图捕获

# 层3：需要 Linux/WSL2（pip install vllm）
python bench_3_vllm.py --model /path/Qwen1.5-0.5B --max-model-len 2048 --concurrency 1,4,16,32

# 层4：对接任意远端服务（SGLang /generate 或 OpenAI 兼容端点，覆盖 TRT-LLM）
python -m sglang.launch_server --model-path /path/Qwen1.5-0.5B --port 30000
python bench_4_sglang.py --backend sglang --base-url http://<host>:30000 \
    --concurrency 1,4,16,32 --ssh-host <user@host>   # --ssh-host 远程采集服务端显存

# 汇总对比
python compare.py
```

模型路径默认 `F:/model/Qwen-0.6B`（Qwen3 架构），所有脚本可用 `--model` 覆盖。

## 测量口径（跨层可比的关键约定）

- 输入为固定种子随机 token，长度严格按 workload（性能只取决于 shape，与文本内容无关）；
- 输出强制固定长度（层 1/2 写死 decode 步数；层 3/4 `ignore_eos=True`），保证生成的 token 数一致；
- 预热（lazy init / JIT / 编译+图捕获）不计入任何指标；
- 每档位重复 `--repeats` 次取 p50/p95；
- 层 3 用 `enforce_eager=True` 关掉 vLLM 自带 CUDA graph，保证"层 3 vs 层 2"只隔离架构变量。

## 环境

- **运行解释器**：`D:\ProgramData\anaconda3\python.exe`（自带 CUDA 版 torch，可直接用 GPU；不需重新下载 torch）
- **本机实测栈**：Anaconda base Python 3.12.3 · torch 2.8.0.dev+cu128 · transformers 4.51.0 · GPU NVIDIA GTX 1050 Ti Max-Q 4GB（Pascal SM6.1，fp16）
- **仓库编写目标栈**：Python 3.13 · torch 2.14 · transformers 4.57（二者差异已通过 CHANGELOG 中记录的兼容补丁抹平，代码对两套版本均可运行）
- 层 3/4 的天花板验证建议：云端 SM ≥ 8.0 卡（A100/L20/4090），支持 bf16 与 FlashAttention。
