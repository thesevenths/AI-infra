# 外层（非提权）调用：以管理员身份运行 run_ncu_local.ps1 并等待结束
# 用法: powershell -File profiling\_launch_ncu_elevated.ps1 [-Smoke]
param([switch]$Smoke)
$argList = '-NoProfile','-ExecutionPolicy','Bypass','-File','e:\AI-infra\profiling\run_ncu_local.ps1'
if ($Smoke) { $argList += '-Smoke' }
$p = Start-Process powershell -Verb RunAs -ArgumentList $argList -Wait -PassThru
Write-Host ("elevated run exit=" + $p.ExitCode)
