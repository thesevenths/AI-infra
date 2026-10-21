"""
run_all.py -- 依次运行 4 种 inference 方式，保留完整日志，最后输出对比表。

日志：inference/logs/<方法>.log（含命令行、全部 stdout/stderr、退出码、耗时）
汇总：inference/logs/comparison.log + 控制台

用法：
  python run_all.py                          # 四层全跑（层3/4 在不满足环境的机器上会留下失败日志）
  python run_all.py --methods 1,2            # 只跑部分层
  python run_all.py --concurrency 1,4 --repeats 3
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"

# (层号, 方法名, 脚本, 该层附加参数)
BENCHES = [
    ("1", "1_eager", "bench_1_eager.py", []),
    ("2", "2_torch_compile", "bench_2_torch_compile.py", []),
    ("3", "3_vllm", "bench_3_vllm.py",
     ["--max-model-len", "384", "--gpu-memory-utilization", "0.5"]),
    ("4", "4_sglang", "bench_4_sglang.py", []),
]


def run_one(name, script, extra, args):
    log_path = LOGS / f"{name}.log"
    cmd = [sys.executable, str(HERE / script),
           "--concurrency", args.concurrency, "--repeats", str(args.repeats)] + extra
    # 子进程统一按 UTF-8 读写管道，规避 Windows 控制台 cp936 编码不一致
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print(f"\n>>> [{name}] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(cmd)}\n" + "=" * 60 + "\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            sys.stdout.write(line)   # 控制台同步可见
            sys.stdout.flush()
            log.write(line)
        code = p.wait()
        dur = time.time() - t0
        log.write("=" * 60 + f"\nexit_code={code} duration={dur:.1f}s\n")
    print(f"<<< [{name}] exit={code} {dur:.1f}s -> {log_path}", flush=True)
    return code


def main():
    # 控制台编码（如 GBK）无法表示的字符降级为 '?'，避免回显崩溃
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="1,2,3,4", help="逗号分隔层号")
    ap.add_argument("--concurrency", default="1,4")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    want = set(args.methods.split(","))
    for layer, name, script, extra in BENCHES:
        if layer in want:
            run_one(name, script, extra, args)

    # 汇总对比表同时写入 logs/comparison.log
    p = subprocess.run([sys.executable, str(HERE / "compare.py")],
                       cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out = p.stdout + p.stderr
    cmp_log = LOGS / "comparison.log"
    cmp_log.write_text(out, encoding="utf-8")
    print(out)
    print(f"comparison saved -> {cmp_log}")


if __name__ == "__main__":
    main()
