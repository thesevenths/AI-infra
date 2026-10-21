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

## 实测结果（本机：MX330 2GB, Pascal SM6.1, fp16）

| Workload | batch | 方法 | TTFT p50 | TPOT p50 | tok/s | VRAM (smi) |
|---|---|---|---|---|---|---|
| A (in128/out128) | 1 | 1_eager | 1301.9 ms | 218.1 ms | **4.4** | **1075 MiB** |
| A | 1 | 2_torch_compile | 1325.2 ms | 332.2 ms | 2.9 | 1963 MiB |
| A | 4 | 1_eager | **5036.7 ms** | **952.3 ms** | **4.1** | **1115 MiB** |
| A | 4 | 2_torch_compile | 9772.2 ms | 1187.0 ms | 3.2 | 1963 MiB |
| B (in512/out32) | 1 | 1_eager | **4841.6 ms** | **259.2 ms** | **2.5** | **1115 MiB** |
| B | 1 | 2_torch_compile | 6287.8 ms | 415.5 ms | 1.6 | 1963 MiB |
| B | 4 | 1_eager | **21301.9 ms** | **722.4 ms** | **3.0** | **1379 MiB** |
| B | 4 | 2_torch_compile | 91310.5 ms | 1506.5 ms | 0.9 | 1963 MiB |

层 3（vLLM）/ 层 4（SGLang、TRT-LLM）在本机无法运行（Windows + 2GB + SM6.1），
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

模型路径默认 `E:/model/Qwen1.5-0.5B`，所有脚本可用 `--model` 覆盖。

## 测量口径（跨层可比的关键约定）

- 输入为固定种子随机 token，长度严格按 workload（性能只取决于 shape，与文本内容无关）；
- 输出强制固定长度（层 1/2 写死 decode 步数；层 3/4 `ignore_eos=True`），保证生成的 token 数一致；
- 预热（lazy init / JIT / 编译+图捕获）不计入任何指标；
- 每档位重复 `--repeats` 次取 p50/p95；
- 层 3 用 `enforce_eager=True` 关掉 vLLM 自带 CUDA graph，保证"层 3 vs 层 2"只隔离架构变量。

## 环境

- Python 3.13 · torch 2.14 · transformers 4.57 · GPU NVIDIA MX330 2GB（Pascal SM6.1，fp16）
- 层 3/4 的天花板验证建议：云端 SM ≥ 8.0 卡（A100/L20/4090），支持 bf16 与 FlashAttention。
