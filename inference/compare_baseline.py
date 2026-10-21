"""
compare_baseline.py -- 对比「旧基线」与「本次运行」的四层阶梯结果。

旧基线：baseline_old/results/*.json（换电脑前的 MX330 2GB + Qwen1.5-0.5B 实测）
本次运行：results/*.json（当前机器实测，例如 1050 Ti Max-Q 4GB + Qwen-0.6B）

用法：
  python compare_baseline.py
  python compare_baseline.py --baseline baseline_old/results --current results

说明：本脚本只做只读对比，不修改任何结果文件。
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD_ORDER = ["1_eager", "2_torch_compile", "3_vllm", "4_sglang", "4_openai"]


def load_dir(d):
    rows = {}
    if not d.exists():
        return rows
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in data:
            rows.setdefault((r["workload"], r["concurrency"], r["method"]), r)
    return rows


def pct(new, old):
    if old in (0, None):
        return float("nan")
    return (new - old) / old * 100.0


def fmt_delta(new, old, unit="ms", lower_better=True):
    d = pct(new, old)
    if d != d:  # nan
        return "  n/a"
    arrow = "↓" if (new < old) == lower_better else "↑"
    return f"{arrow}{abs(d):5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(HERE / "baseline_old" / "results"))
    ap.add_argument("--current", default=str(HERE / "results"))
    args = ap.parse_args()

    old = load_dir(Path(args.baseline))
    new = load_dir(Path(args.current))

    keys = sorted(set(old) | set(new))
    if not keys:
        print("没有可对比的数据（baseline 或 current 为空）。")
        return

    methods = [m for m in METHOD_ORDER if any(k[2] == m for k in keys)]
    hdr = (f"{'workload':<22}{'batch':>5}{'method':<16}"
           f"{'TTFT旧':>10}{'TTFT新':>10}{'Δ':>8}"
           f"{'TPOT旧':>10}{'TPOT新':>10}{'Δ':>8}"
           f"{'tok/s旧':>9}{'tok/s新':>9}{'Δ':>8}"
           f"{'VRAM旧':>9}{'VRAM新':>9}{'Δ':>8}")
    print("\n===== 基线(MX330 2GB / Qwen1.5-0.5B) vs 本次(当前机) =====")
    print(hdr)
    print("-" * len(hdr))
    for (wl, b, m) in keys:
        o = old.get((wl, b, m))
        n = new.get((wl, b, m))
        if o is None or n is None:
            tag = " (缺基线)" if o is None else " (缺本次)"
            print(f"{wl:<22}{b:>5}{m:<16}  {tag}")
            continue
        print(f"{wl:<22}{b:>5}{m:<16}"
              f"{o['ttft_ms']:>9.1f}ms{n['ttft_ms']:>9.1f}ms"
              f"{fmt_delta(n['ttft_ms'], o['ttft_ms']):>8}"
              f"{o['tpot_ms']:>9.1f}ms{n['tpot_ms']:>9.1f}ms"
              f"{fmt_delta(n['tpot_ms'], o['tpot_ms']):>8}"
              f"{o['throughput_tps']:>9.1f}{n['throughput_tps']:>9.1f}"
              f"{fmt_delta(n['throughput_tps'], o['throughput_tps'], lower_better=False):>8}"
              f"{o['vram_smi_peak_mib']:>8.0f}{n['vram_smi_peak_mib']:>8.0f}"
              f"{fmt_delta(n['vram_smi_peak_mib'], o['vram_smi_peak_mib']):>8}")
    print("(Δ: ↓=改善, ↑=退化；tok/s 越大越好，其余越小越好)")


if __name__ == "__main__":
    main()
