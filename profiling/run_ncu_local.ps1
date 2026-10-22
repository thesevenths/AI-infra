# Experiment 4 / 层B：本机 Nsight Compute 采集封装
#
# 【本机结论 2026-10-22】提权（UAC）后权限问题解决（不再 ERR_NVGPUCTRPERM），
# 但 ncu 2025.1.0 报 "Profiling is not supported on device 0"：
# --list-chips 支持列表从 Turing (tu102) 起，本机 MX330 是 Pascal (GP108)，
# 该版本已放弃 Pascal 支持 —— 本机层B 不可行，按预案到云端补采
# （见 nsys_commands.md 第 4 节）。本脚本保留，供云端之外的兼容机器使用。
#
# 用法（PowerShell）：
#   1) 普通用户直接跑会 ERR_NVGPUCTRPERM（计数器无权限）。本脚本需提权：
#      右键"以管理员身份运行 PowerShell"后执行本文件；或由外层用
#      Start-Process -Verb RunAs 拉起（弹一次 UAC 点"是"）。
#   2) 先 smoke 再正式：
#      run_ncu_local.ps1 -Smoke   # 只采 2 个 kernel，验证计数器权限
#      run_ncu_local.ps1          # 正式采集
#   3) 产物：results\ncu_run.log         （ncu 进度日志）
#            results\ncu_decode_b1.ncu-rep（Nsight Compute GUI 打开）
#            results\ncu_decode_b1.csv    （文本指标表，由 --import 生成）
#
# 参数说明：
#  - --replay-mode application：2GB 卡放不下 kernel-replay 的内存快照
#    （replay 每 pass 要复制 ~1.2GB 设备状态），application 模式改为
#    整进程重放，显存峰值不变。
#  - --kernel-name 正则：只采 decode 的代表性 kernel——
#      gemv2T_kernel_val        decode 权重 GEMV（memory-bound 主角）
#      fmha_cutlassF            SDPA memory-efficient attention kernel
#      MeanOps                  RMSNorm 的 mean/sqrt（未融合，多次 HBM 往返）
#      CatArrayBatchedCopy      attention mask/KV cat
#      index_copy               StaticCache 原位写入
#  - --launch-skip 200：跳过 prefill（匹配 kernel ~200 个）后从 decode
#    取材；--launch-count 控制 replay 总时长（application 模式每个 pass
#    都要完整重跑一遍进程）。
#  - GeForce 部分计数器受限：--set basic 可用（SOL/带宽/占用率足够定性）。
#  - 输出走 ncu 自带 --log-file：Windows PowerShell 5.1 下 native 命令的
#    stderr 经 2>&1 管道会被转成 ErrorRecord，配合 ErrorActionPreference
#    Stop 会直接掐死脚本（上次失败原因），不再用管道接日志。

param([switch]$Smoke)

$ErrorActionPreference = "Continue"   # 关键：不能让 stderr 触发终止
$results = Join-Path $PSScriptRoot "results"
$python  = "D:\python313\python.exe"   # 本机 CUDA 版 torch 2.14.1+cu126 解释器
$ncu     = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe"

if (-not (Test-Path $ncu)) {
    # 兜底：PATH 上的 ncu.bat（开始菜单安装的默认位置）
    $ncu = "ncu"
}
New-Item -ItemType Directory -Force -Path $results | Out-Null

$export  = Join-Path $results "ncu_decode_b1"
$logfile = Join-Path $results "ncu_run.log"
if ($Smoke) { $count = 2; $steps = 4 } else { $count = 30; $steps = 10 }

& $ncu `
    --target-processes all `
    --replay-mode application `
    --set basic `
    --launch-skip 200 `
    --launch-count $count `
    --kernel-name "regex:gemv2T_kernel_val|fmha_cutlassF|MeanOps|CatArrayBatchedCopy|index_copy" `
    -f --export $export `
    --log-file $logfile `
    -- `
    $python (Join-Path $PSScriptRoot "prof_2_ncu_kernels.py") `
    --model "E:/model/Qwen1.5-0.5B" --steps $steps

# 把 .ncu-rep 导出成 CSV 文本表（单独一次 import，不采集）
& $ncu --import "$export.ncu-rep" --csv 2>&1 |
    Out-File -Encoding utf8 "$export.csv"

Write-Host ""
Write-Host "报告产物:"
Write-Host "  $logfile"
Write-Host "  $export.ncu-rep  (用 Nsight Compute GUI 打开)"
Write-Host "  $export.csv      (文本指标表)"
