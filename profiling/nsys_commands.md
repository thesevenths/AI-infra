# Nsight Systems (nsys) 采集指南 —— 云端 Linux / WSL2 执行

> nsys CLI **只有 Linux 版**（本机 Windows 无法采集；WSL2 尚未安装发行版）。
> 采集在 Linux 上做，生成的 `.nsys-rep` 可以拷回 Windows 用 **Nsight Systems
> GUI** 打开——查看不需要 Linux。

## 0. 前置条件

- Linux GPU 机器（云端 A100/L20/4090 均可，驱动 ≥ 对应 CUDA 版本）
- 仓库代码 + 模型权重同步到该机；`pip install torch transformers`
- 安装 nsys（随 CUDA Toolkit 附带，或单独下载）：

```bash
# 确认已有
nsys --version || sudo apt install -y nsight-systems-cli
```

## 1. 采集命令（两种变体各一份）

在仓库的 `inference/` 目录下执行：

```bash
cd inference

# 层1 eager：只跑 1 个 shape，避免时间轴过长
nsys profile --trace=cuda,nvtx,osrt --stats=true \
    -o ../profiling/results/nsys_eager \
    --force-overwrite=true \
    python bench_1_eager.py --concurrency 1 --workloads A --repeats 1

# 层2 torch.compile（cudagraphs/inductor 视云端卡而定）
nsys profile --trace=cuda,nvtx,osrt --stats=true \
    -o ../profiling/results/nsys_compiled \
    --force-overwrite=true \
    python bench_2_torch_compile.py --concurrency 1 --workloads A --repeats 1
```

要点：
- `--repeats 1` + 单一 workload/并发：负载与 Experiment 2 口径一致，
  且 warmup 已把编译/图捕获排除在指标外；但 **进程启动后的编译期也在
  时间轴上**，在 GUI 里看测量段即可（bench 会打印每档结果，对照时间点）。
- `--trace=cuda,nvtx,osrt`：cuda 给出 kernel/memcpy/API 轨道；osrt 给出
  操作系统线程调度（观察 CPU 侧 launch 线程是否被抢占）。
- 若只想快速拿数字不采 GUI 文件：`nsys stats --report cuda_gpu_kern_sum
  --report cuda_gpu_time_util nsys_eager.nsys-rep`（`gpu-metrics` 需
  云端物理卡，vGPU/部分云实例不支持时间线利用率采样）。

## 2. 看什么（层C 的三个问题）

| 问题 | GUI 操作 | eager 预期 | cudagraphs 预期 |
|---|---|---|---|
| GPU 空隙率（launch-bound 的直接证据） | CUDA HW 轨道放大到单个 decode step，目测 kernel 之间的间隙；或 `nsys stats --report gpu_time_util` | 大量 µs 级间隙，利用率明显 <100% | kernel 首尾相接成"实心块" |
| cudaLaunchKernel 的 CPU 成本 | API 轨道对齐 HW 轨道：每个 HW kernel 前面挂一个极短的 launch 调用 | launch 数量 = kernel 数量（数百/步） | 一步只有一次 cudaGraphLaunch |
| 显存操作是否出现在稳态 | 搜 `cudaMalloc` / `cudaMemcpy` / `cudaFree` | 只应在 warmup/首步出现（本项目用 StaticCache 规避了 DynamicCache 的每步 cat/分配——如果稳态仍有 memcpy，说明有回归） | 同左 |

## 3. 采集产物回收

```bash
# 从云端拉回本地（示例）
scp <user>@<host>:~/AI-infra/profiling/results/nsys_*.nsys-rep profiling/results/
```

Windows 安装 [Nsight Systems GUI](https://developer.nvidia.com/nsight-systems)
直接打开 `.nsys-rep`。

## 4. 附：层B (ncu) 本机实测不可行，到云端补采

> 本机实测结论（2026-10-22，办公本 MX330/Pascal GP108）：
> 1. 普通用户跑 ncu 报 `ERR_NVGPUCTRPERM`（需管理员/UAC 提权，已验证可解决）；
> 2. 提权后 ncu 2025.1.0 报 `Profiling is not supported on device 0`，
>    `--list-chips` 支持列表从 Turing (tu102) 起 —— **该版本已放弃 Pascal**，
>    本机任何姿势都采不了，层B 必须去云端。

云端用数据中心卡（A100/L20）可直接 kernel-replay，无需 application 模式：

```bash
ncu --set full --launch-skip 70 --launch-count 40 \
    -o ../profiling/results/ncu_decode_b1_cloud \
    -- python profiling/prof_2_ncu_kernels.py --steps 10
```

云端数据中心卡（A100/L20）支持 `--set full` 全部计数器，能拿到精确的
DRAM bytes 与 roofline 点位，比本机 GeForce 的 `basic` 集更可信。
