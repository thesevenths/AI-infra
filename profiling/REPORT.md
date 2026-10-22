# Experiment 4 报告：为什么这个 Transformer 慢？

> 结论先行（本机实测，MX330 2GB + Qwen1.5-0.5B fp16，batch=1，输入 128 token）：
>
> 1. **Decode（稳态 TPOT 198.2 ms）的主因不是 launch 空隙，而是"带宽 + 低效 kernel"**：
>    GPU 忙时率 93.7%，1159 个 kernel/步里 GEMV 占 70% 时长；用 lm_head 单发 GEMV
>    实测本卡有效显存带宽只有 **≈8.4 GB/s**（规格标称 40 GB/s 的 1/5，笔记本功耗墙）。
>    batch=1 解码算术强度 ≈1 FLOP/byte，远低于本机 roofline 拐点（≈8–12 FLOP/B）——
>    **memory-bound 是原罪**，纯权重流送的物理下限 ≈101 ms，实测 198 ms 的剩余差距
>    在低效小 kernel（未融合的 RMSNorm/RoPE、sm50 回退的 attention）。
> 2. **Prefill（TTFT 1016 ms）是纯算力受限**：GPU 忙时率 98.8%，没有 launch 空间；
>    GEMM 占 80% 时长，Pascal 无 Tensor Core，cuBLAS 只能选 fp32 计算的
>    `hgemm/maxwell_fp16` 回退 kernel，实测只拿到 ≈69 GFLOP/s。
> 3. **cudagraphs 在这张 2GB 老卡上反而更慢（TPOT 316 ms，+60%）**：稳态每步多出
>    ≈452 次 DtoD memcpy / ≈900 MB 克隆流量，把 launch 收益（本机本来就只值 6%）
>    整个吃掉——这为 Experiment 2 "老卡 torch.compile 回归" 提供了机制级证据。

---

## 0. 实验环境与口径

| 项 | 值 |
|---|---|
| 机器 | 办公笔记本，NVIDIA **GeForce MX330 2GB**（Pascal GP108 / SM 6.1，256 核，64-bit GDDR5，标称带宽 40.06 GB/s，标称 fp32 ≈0.5 TFLOPS，**无 Tensor Core**） |
| 软件 | torch 2.14.1+cu126，transformers 4.57.3，`D:\python313\python.exe` |
| 模型 | `E:\model\Qwen1.5-0.5B`：24 层，hidden 1024，GQA kv_heads=2，vocab 151936，tie_word_embeddings |
| 负载 | 与 Experiment 2 完全一致：StaticCache、batch=1、输入 128 token（固定种子随机 id）、decode 30 步、`logits_to_keep=1`、fp16 + sdpa |
| 变体 | `1_eager` vs `2_torch_compile`（SM<7.0 无 Triton，自动只挂 `cudagraphs` backend） |

环境注记（都影响数据可信度声明）：
- 本机 16GB RAM 被办公软件占满时 `from_pretrained` 会触发 Windows 提交内存不足
  （os error 1455），脚本自动回退 `stream_load.py`（纯顺序读、不 mmap）。该加载只用于
  **性能测量**：权重逐 tensor 原样上卡，与标准加载的时序口径一致，但不做权重完整性复核。
- 指标计算不用 `prof.events()` 聚合（CPU 算子会携带聚合 device 时间，重复计 2–3 倍），
  而是解析导出的 chrome trace JSON，只取 `kernel / gpu_memcpy / gpu_memset` 三类事件；
  "GPU 忙时率" = 设备事件区间的**并集** / 墙钟，不是简单求和（并发流会虚高）。
- 稳态指标取 `decode_step_i` NVTX 标注的中位数（先对 30 步各自求窗口内 busy/时长，再取
  median），排除 CUDA graph capture 首步和预热噪声。

产物索引：本节之后所有数字可在 `results/prof_metrics.json`（汇总）、
`results/*.table.txt`（key_averages 表）、`results/*.trace.json`（Perfetto/
chrome://tracing 时间轴）中复核；原始日志 `results/runA_*.log`。

---

## 1. 层A：PyTorch Profiler 实测（本机，全部完成）

### 1.1 eager decode——稳态每步 198 ms 里装了什么

| 指标 | 实测值 |
|---|---|
| 稳态 TPOT（30 步中位） | **198.19 ms** |
| GPU 忙时率（稳态） | **0.937** → 每步 launch 空隙 ≈ **12.4 ms（6.3%）** |
| 每步设备事件数 | **1159**（kernel 为主） |
| 单 kernel 平均时长 | **158.2 µs**（34081 kernel / 30 步 / 36 个唯一名字） |
| 每步 kernel 时长合计 | 186.46 ms |
| 窗口内 memcpy/memset | 751 次 / **0.217 MB**（StaticCache 生效，稳态几乎零搬运） |

按 kernel 名聚合的时长分解（每步，30 步平均）：

| 类别 | 次数/步 | 时长/步 | 占比 | 说明 |
|---|---|---|---|---|
| `gemv2T_kernel_val` ×2 变体 | 97 + 72 | **130.9 ms** | **70.2%** | batch=1 权重乘：qkv/o/mlp/lm_head 全是 GEMV |
| `fmha_cutlassF_f16_..._sm50` | 24 | **30.7 ms** | 16.5% | 单发平均 **1.28 ms**——decode 只看 ≤160 个 KV，本应 µs 级 |
| RMSNorm 链（`MeanOps` reduce + `rsqrt` + `mul`） | ≈150 | 4.7 ms | 2.5% | **未融合**：一次归约要 mean→rsqrt→mul 3+ 个 kernel，每次全量往返显存 |
| RoPE 链（`mul`/`neg`/`cat`/`index_copy`） | ≈240 | 5.4 ms | 2.9% | cos/sin 拼接与 cache 原位写 |
| 其余 elementwise/copy | 数百 | ≈12 ms | 6% | 残差加、类型转换等 |

**关键换算（本机带宽的两次独立实测）**：
- prefill 的 lm_head 单发 GEMV：读 151936×1024×2 B = **311.2 MB**，用时 **37.1 ms**
  → 有效带宽 **≈8.4 GB/s**（标称 40 GB/s 的 21%；笔记本 MX330 受功耗/温度限制）。
- decode 全部 GEMV：每步流送权重 270.5M(主体) + 155.6M(lm_head) 参数 ×2 B ≈ **852 MB**，
  用时 130.9 ms → **≈6.5 GB/s**。
- 由此得纯权重流送下限：852 MB ÷ 8.4 GB/s ≈ **101 ms/token**。实测 198 ms 中多出的
  ~97 ms 就是上面 30.7 ms attention + 小 kernel 群 + 空隙——**"慢"的剩余部分不是玄学，
  是可点名的 kernel**。

Roofline 判定：batch=1 GEMV 每读 2 字节权重只做 2 FLOP，算术强度 **≈1 FLOP/byte**；
本机拐点 = 算力/带宽 ≈ 0.5 TF ÷ 40 GB/s = 12.5（用实测 8.4 GB/s 与 69 GFLOP/s 算也
≈8.2）。1 ≪ 8~12.5 → decode 深居 memory-bound 区，**换更快的 launch 方案救不了它，
能救的是量化权重（bytes/参数↓）与 batching（强度随 batch 线性↑）**。

### 1.2 eager prefill——TTFT 1016 ms 是另一种慢

| 指标 | 实测值 |
|---|---|
| 墙钟（1 次 128-token prefill） | **1016.0 ms** |
| GPU 忙时率 | **0.988**（几乎没有 launch 空隙可言） |
| kernel 数 / 平均时长 | 1161 个 / **864.7 µs** |
| GEMM 三大头 | `hgemm_128x128` 337.6 ms(48发) + `hgemm_32x32` 249.8 ms(48发) + `maxwell_fp16_sgemm` 219.5 ms(72发) = **807 ms，占 80%** |
| attention | `fmha_cutlassF` sm50 版，50.4 ms(24发)——非 FlashAttention（**Pascal 不支持 FA2**），走 cutlass mem-efficient 回退 |

算力利用率：prefill ≈ 2×270.5M×128 ≈ **69.2 GFLOP** ÷ 1.004 s kernel 时长 ≈
**69 GFLOP/s**，仅为标称 fp32 峰值的 ~14%。原因是 Pascal 的 fp16 GEMM 实为
fp32 单元计算（fp16 只省显存不省算力）+ 老 cublas 在小显存卡上的 tile 选择保守。
结论：**prefill 慢 = 纯算力受限 + 无 Tensor Core 的架构代差**，与 launch/带宽无关。

### 1.3 cudagraphs 对照——"消除 launch 空隙"在这张卡上是个负收益

同一负载下 `torch.compile(backend=cudagraphs)` 稳态实测：

| 指标 | eager | cudagraphs | Δ |
|---|---|---|---|
| 稳态 TPOT | 198.2 ms | **316.2 ms** | **+60%** |
| GPU 忙时率 | 0.937 | 0.972 | +3.5 pp |
| 每步设备事件 | 1159 | **1597** | **+438** |
| 每步 kernel+memcpy 时长 | 186.5 ms | **307.5 ms** | +121 ms |
| 窗口 memcpy | 751 次 / 0.2 MB | **13578 次 / 26983 MB** | **≈900 MB/步 DtoD** |

机制：SM<7.0 上 Inductor 无 Triton 融合，`cudagraphs` 只建图不省 kernel 数；
torch 的 cudagraph trees 机制每步要把输入/中间 buffer **克隆进静态图槽位**
（+452 次 DtoD/步，≈900 MB——在 8.4 GB/s 的卡上这就是 ~110 ms），而 eager 的
launch 空隙本来就只值 12 ms/步。**省不下的加上躲不掉的拷贝 = 净亏 118 ms/步。**
这解释了 Experiment 2 在老卡上看到的 torch.compile 回归；同一技术在 A100（融合
+ 8 TB/s HBM）上是正收益——工具没有万能药，只有本机 roofline。

> 复核提示：`results/compiled_decode.trace.json` 在 Perfetto 里放大任一步，能肉眼看到
> eager 的"锯齿"（µs 级间隙）对比 cudagraphs 步内密集的 DtoD 拷贝带。
> 注意：同进程先跑 eager 再 compile 会因提交内存耗尽 MemoryError，两个变体须分进程跑
> （`prof_1_torch_profiler.py --variant eager|compiled` 各一次，json 自动合并）。

---

## 2. 层B：Nsight Compute——本机实测不可行（已按预案转云端）

本机两步排查均有实证记录（`results/ncu_run.log`）：

1. 普通用户运行报 `ERR_NVGPUCTRPERM`（GPU 计数器权限）。用户属 Administrators，
   经 UAC 提权后**权限问题确认解决**（ncu 成功接管进程）。
2. 提权后 ncu 2025.1.0 直接拒绝：`Profiling is not supported on device 0`。
   `ncu --list-chips` 支持列表自 **Turing (tu102)** 起——**本机 MX330 是 Pascal
   (GP108)，该版本已放弃支持**。办公机不具备安装旧版 ncu 的条件，本机层B 到此为止。

补采任务与判定标准已写入 `nsys_commands.md` §4（云端 `ncu --set full`，数据中心卡
支持 Pascal 之后的所有架构且计数器全开）。预期验证三件事：
- `gemv2T`：Memory Throughput ≈ 饱和、SM Throughput 个位数 → memory-bound 直测；
- `fmha_cutlassF`：两者都低、单发极短 → 老架构 attention kernel 的低效直测；
- RMSNorm `MeanOps`：DRAM bytes ≈ 张体积 ×往返次数 → "未融合 = 多次全量往返"直测。

## 3. 层C：Nsight Systems——云端执行

nsys CLI 仅有 Linux 版（本机 Windows 无法采集，WSL2 未装发行版）。完整采集命令、
看什么（GPU 空隙率 / cudaLaunchKernel CPU 成本 / 稳态是否出现 cudaMalloc-memcpy）、
产物回收方式见 **`nsys_commands.md`**。跑完后本报告 §1.3 的结论可用时间轴目测复核：
A100 上 eager 的"锯齿"会远比本机明显（快卡 launch-bound 占比升高），cudagraphs
则应呈现首尾相接的实心块——与本机形成正反差对照。

---

## 4. 最终回答：为什么这个 Transformer 慢？

对"这个 Transformer"（batch=1 自回归解码，0.5B 模型，办公本 MX330）：

1. **结构性原因（换任何卡都存在）**：解码每生成 1 token 就要把全部权重从显存流送
   一遍（fp16 0.5B ≈ 852 MB），算术强度 ≈1 FLOP/byte，天生住在 roofline 的
   memory-bound 区。快卡靠高带宽硬抗（HBM 3 TB/s → 下限 0.3 ms），本卡 8.4 GB/s
   → **下限就是 101 ms/token**，实测 198 ms。
2. **本机加速器（让下限变成上限的三件事）**：
   - 笔记本 GPU 有效带宽只有标称 21%（lm_head 单发实测 8.4 GB/s）；
   - **未融合的 norm/RoPE 链**：每层 5–6 个全量往返显存的小 kernel，24 层堆出
     每步 ~440 个 µs–ms 级事件（占 decode 的 ~11%）；
   - **Pascal 架构代差**：无 Tensor Core（prefill 只拿到 69 GFLOP/s）、不支持
     FA2（fmha 走 sm50 回退，单发 1.28 ms）、fp16 只是存储格式不是算力。
3. **反直觉但已闭环验证的一条**：本机 launch 空隙只占 6.3%，所以 cudagraphs 这类
   "治 launch"的药在老卡上是**负收益**（+60% TPOT，每步 900 MB 克隆流量）；它治的
   是快卡上的病。这比"我会用工具"更有价值的地方就在这：**先量 roofline，再选药**。

一句话版本：**decode 是被显存带宽和未融合小 kernel 拖死的，prefill 是被无 Tensor
Core 的旧架构算力卡住的，而一切"高级加速开关"在 2GB Pascal 上都要先过 roofline 这一关。**

## 5. 待办（云端补数后更新本报告）

- [ ] 云端 `nsys profile` 两份 .nsys-rep（eager vs compiled），回填 §3 空隙率数字
- [ ] 云端 `ncu --set full` 三类代表 kernel 的 SOL/DRAM bytes，回填 §2
- [ ] （可选）同负载在 A100/L20 重跑 `prof_1_torch_profiler.py`，对照本机的
      busy 率与 TPOT，验证"快卡上 launch-bound 占比升高"的预期

## 附：复现命令

```powershell
# 层A（本机，两变体分进程跑；--model 必传，默认值是家里机器的路径）
D:\python313\python.exe e:\AI-infra\profiling\prof_1_torch_profiler.py --variant eager   --model E:/model/Qwen1.5-0.5B
D:\python313\python.exe e:\AI-infra\profiling\prof_1_torch_profiler.py --variant compiled --model E:/model/Qwen1.5-0.5B

# 层B（本机不可行留档；提权封装见 run_ncu_local.ps1 头注）
powershell -File e:\AI-infra\profiling\_launch_ncu_elevated.ps1 -Smoke

# 层C/层B 云端命令见 nsys_commands.md
```

时间轴查看：`results/*.trace.json` → https://ui.perfetto.dev （116 MB 的
eager_decode 那份最直观；chrome://tracing 亦可）。
