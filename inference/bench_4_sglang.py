"""
层4：SGLang / TensorRT-LLM —— 生产级内核+架构，测"上限与工程代价"

本脚本是一个流式 HTTP 压测客户端（纯 Python，Windows 本机也能跑），
对接两种服务端：
  --backend sglang   : SGLang 的 /generate 接口（stream=true, ndjson 增量）
  --backend openai   : 任意 OpenAI 兼容 /v1/completions 流式接口
                       （覆盖 TensorRT-LLM 的 trtllmserve / triton 端口）

服务端必须在 Linux 起（SGLang/TRT-LLM 不支持 Windows），例如同命令模板：
  # SGLang:
  python -m sglang.launch_server --model-path /path/Qwen1.5-0.5B --port 30000
  # TensorRT-LLM（需先 trtllm-build，这本身就是本层要记录的"工程代价"之一）:
  trtllmserve --model=...  （起好后用 --backend openai --base-url 指过去）

测量口径与层1/2/3 一致：TTFT=首个 token 到达；TPOT=token 间隔均值；
Throughput=总输出 token/墙钟；VRAM=nvidia-smi 峰值（--ssh-host 可远程采集，
默认采本机）。

用法示例：
  python bench_4_sglang.py --backend sglang --base-url http://127.0.0.1:30000 \
      --concurrency 1,4,16,32
"""

import argparse
import json
import subprocess
import threading
import time

import requests

from llm_bench import DEFAULT_MODEL, WORKLOADS, RunResult, make_input, save_and_print_table


def make_prompt(tokenizer, input_len, seed=0):
    """与层1 同源的随机 token 输入，转成文本喂给服务端。"""
    ids = make_input(tokenizer, "cpu", 1, input_len, seed=seed)[0].tolist()
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def stream_request(url, headers, payload, backend):
    """发一个流式请求，返回 (ttft_ms, tpot_ms, n_tokens)。"""
    t0 = time.perf_counter()
    token_times = []
    try:
        with requests.post(url, headers=headers, json=payload,
                           stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if backend == "sglang":               # SGLang ndjson
                    data = json.loads(line)
                    text = data.get("text", "")
                else:                                 # OpenAI SSE
                    s = line.decode("utf-8")
                    if not s.startswith("data: "):
                        continue
                    s = s[6:]
                    if s.strip() == "[DONE]":
                        break
                    data = json.loads(s)
                    text = data["choices"][0].get("text", "")
                # 用增量长度估算收到的 token 数（近似，够算 p50）
                if text:
                    token_times.append(time.perf_counter())
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return None
    if not token_times:
        return None
    ttft = (token_times[0] - t0) * 1e3
    tpot = ((token_times[-1] - token_times[0]) * 1e3
            / max(1, len(token_times) - 1))
    return ttft, tpot, len(token_times)


def nvidia_smi_remote(ssh_host=None):
    """采集 nvidia-smi 显存读数；给了 --ssh-host 就在服务端机器上采。"""
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if ssh_host:
        cmd = ["ssh", ssh_host] + cmd
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
        return max(int(x) for x in r.stdout.split() if x.strip().isdigit())
    except Exception:
        return -1


def main():
    p = argparse.ArgumentParser(description="Experiment 2 / layer-4 SGLang/TRT-LLM bench")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--backend", choices=["sglang", "openai"], default="sglang")
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--concurrency", default="1,4,16,32")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--workloads", default="A,B")
    p.add_argument("--ssh-host", default=None,
                   help="远程 GPU 机器(如 user@host)，用于采集服务端 nvidia-smi")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.backend == "sglang":
        url = f"{args.base_url}/generate"
        headers = {"Content-Type": "application/json"}
    else:
        url = f"{args.base_url}/v1/completions"
        headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer EMPTY"}

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    selected = []
    if "A" in args.workloads:
        selected.append(("A_short_in_long_out", WORKLOADS["A_short_in_long_out"]))
    if "B" in args.workloads:
        selected.append(("B_long_in_short_out", WORKLOADS["B_long_in_short_out"]))

    method = f"4_{args.backend}"
    results = []
    for wname, wl in selected:
        prompt = make_prompt(tokenizer, wl["input_len"])
        for b in concurrencies:
            res = RunResult(method=method, workload=wname, concurrency=b,
                            repeats=args.repeats)
            for r in range(args.repeats):
                if args.backend == "sglang":
                    payload_tpl = {"text": prompt,
                                   "sampling_params": {
                                       "temperature": 0, "max_new_tokens": wl["output_len"],
                                       "ignore_eos": True},
                                   "stream": True}
                else:
                    payload_tpl = {"model": "model", "prompt": prompt,
                                   "max_tokens": wl["output_len"],
                                   "temperature": 0, "stream": True}

                outs = [None] * b

                def worker(i, tpl=payload_tpl):
                    outs[i] = stream_request(url, headers, tpl, args.backend)

                # 显存后台采样
                stop = threading.Event()
                peak = [0]

                def sampler():
                    while not stop.is_set():
                        peak[0] = max(peak[0], nvidia_smi_remote(args.ssh_host))
                        time.sleep(0.2)

                th = threading.Thread(target=sampler, daemon=True)
                th.start()

                threads = [threading.Thread(target=worker) for _ in range(b)]
                t0 = time.perf_counter()
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = time.perf_counter() - t0
                stop.set()
                th.join(timeout=1)

                got = [o for o in outs if o]
                if not got:
                    print(f"  {wname} batch={b}: 全部请求失败，服务端起了吗？")
                    continue
                total_tokens = sum(g[2] for g in got)
                for ttft, tpot, _ in got:
                    res.add_sample(ttft, tpot, wall, 1, 1)
                # 系统吞吐用真实 token 数重算（add_sample 的近似不用）
                res._thpts = [total_tokens / wall]
                res.vram_smi_peak_mib = float(peak[0])
            if getattr(res, "_ttfts", None):
                results.append(res.finalize())
                print(f"  {wname} batch={b}: TTFT {res.ttft_ms:.1f}ms | "
                      f"TPOT {res.tpot_ms:.1f}ms | "
                      f"{res.throughput_tps:.1f} tok/s | "
                      f"VRAM {res.vram_smi_peak_mib:.0f}MiB", flush=True)

    save_and_print_table(method, results)


if __name__ == "__main__":
    main()
