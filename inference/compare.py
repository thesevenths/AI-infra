"""
compare.py -- 汇总对比：读取 results/*.json（各层 bench 产出），打印阶梯表。

用法：
  python compare.py                          # 对比 results 下所有已存在的方法
  python compare.py --methods 1_eager,2_torch_compile
"""
import argparse
import json
from pathlib import Path

from llm_bench import RESULTS_DIR

METHOD_ORDER = ["1_eager", "2_torch_compile", "3_vllm", "4_sglang", "4_openai"]


def load(methods):
    rows = {}
    for m in methods:
        p = RESULTS_DIR / f"{m}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            rows.setdefault((r["workload"], r["concurrency"]), {})[m] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows = load(methods)
    if not rows:
        print(f"results/ 下没有任何结果，先运行 bench_*.py（目录: {RESULTS_DIR}）")
        return

    present = [m for m in methods if any(m in v for v in rows.values())]
    hdr = (f"{'workload':<22}{'batch':>5}{'method':<16}"
           f"{'TTFT p50':>10}{'TPOT p50':>10}{'tok/s':>8}{'VRAM MiB':>10}")
    print("\n============ Experiment 2 / 四层阶梯对比 ============")
    print(hdr)
    print("-" * len(hdr))
    for (wl, b) in sorted(rows):
        for m in present:
            r = rows[(wl, b)].get(m)
            if r is None:
                continue
            print(f"{wl:<22}{b:>5}{m:<16}"
                  f"{r['ttft_ms']:>9.1f}ms{r['tpot_ms']:>9.1f}ms"
                  f"{r['throughput_tps']:>8.1f}{r['vram_smi_peak_mib']:>10.0f}")
        print()
    missing = [m for m in methods if m not in present]
    if missing:
        print(f"（缺少: {', '.join(missing)} —— 层3/层4 需要 Linux/云端环境，"
              f"见对应 bench_*.py 头部说明）")


if __name__ == "__main__":
    main()
