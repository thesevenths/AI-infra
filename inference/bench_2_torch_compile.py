"""
层2：torch.compile —— 只换内核执行方式，运行时架构不变

与层1共用同一套 HF 模型和测量循环，唯一区别是 decode 步的 model 被
torch.compile 包装（prefill 保持 eager，理由见 llm_bench.measurement）。
预期现象：
  - TPOT 下降幅度取决于 GPU：编译收益来自算子融合与消除 launch 开销，
    在 decode 本身已 GPU-bound 的入门卡上收益有限
  - TTFT 不变（prefill 不编译——这也是 serving 框架的通行做法）
  - Throughput 在并发档位几乎不动（批处理架构没变——这是本层的实验结论）

注意：
  - 首次调用触发编译/图捕获，全部放进 warmup，不计入任何指标。
  - 测量内核用 StaticCache：decode 每步 shape 恒定，才能被 cudagraph
    trees 单一图复用；层1 用同一内核重跑，保证可比。
  - Inductor 的 GPU 代码生成依赖 Triton，要求 SM>=7.0；不满足时
    --backend auto 自动降级为 cudagraphs（只保留 CUDA Graphs 效果，
    无算子融合）。

用法：
  python bench_2_torch_compile.py --concurrency 1,4 --repeats 3
"""
import torch

from llm_bench import (WORKLOADS, apply_inductor_workaround, build_parser,
                       run_suite, warmup)


def main():
    p = build_parser("2_torch_compile")
    p.add_argument("--mode", default="default",
                   help="torch.compile mode: default | max-autotune-no-cudagraphs")
    p.add_argument("--backend", default="auto",
                   help="inductor | cudagraphs | eager | auto(按显卡能力选)")
    args = p.parse_args()

    print(f"[inductor] {apply_inductor_workaround()}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    device = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # DynamicCache 式的"上一步输出被下一步读取"与 cudagraph trees 的
    # generation 复写机制冲突；该配置让图输出跨 generation 自动 clone。
    import torch._inductor.config as inductor_config
    inductor_config.triton.cudagraph_trees_generation_cloning = "user_visible"

    # 本层唯一的"改动"：选后端 + torch.compile 包装
    backend = args.backend
    if backend == "auto":
        cap = torch.cuda.get_device_capability(0) if device == "cuda" else (0, 0)
        if device == "cuda" and cap < (7, 0):
            backend = "cudagraphs"
            print(f"[auto] GPU SM{cap[0]}.{cap[1]} 不被 Triton 支持，"
                  f"降级为 cudagraphs 后端（仅 CUDA Graphs，无算子融合）")
        else:
            backend = "inductor"
    compile_kwargs = ({"backend": backend} if backend != "inductor"
                      else {"mode": args.mode})
    # dynamic=False：禁止跨 shape 动态泛化，确保每个 (batch, total_len)
    # 拥有独立的编译条目与 cudagraph trees 节点，避免静态输入尺寸错配。
    compile_kwargs["dynamic"] = False
    if backend in ("cudagraphs", "inductor"):
        # 预导入 cudagraph_trees：测量内核据此决定 decode 每步是否
        # cudagraph_mark_step_begin（约束图内存生命周期）。
        # 用 from-import：函数内 `import torch.x` 会把 torch 变成局部名。
        from torch._inductor import cudagraph_trees  # noqa: F401
    print(f"[2_torch_compile] torch.compile({compile_kwargs}) —— 仅 decode")

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    selected = []
    if "A" in args.workloads:
        selected.append(WORKLOADS["A_short_in_long_out"])
    if "B" in args.workloads:
        selected.append(WORKLOADS["B_long_in_short_out"])
    # 编译预热必须覆盖真实测量的每个 shape，否则编译时间混进 TTFT
    shapes = [(b, wl["input_len"], wl["output_len"])
              for b in concurrencies for wl in selected]

    # 每个 (batch, total_len) 一套：独立 compile wrapper（配合 dynamic=False
    # 隔离 shape）+ 独立 StaticCache（warmup 与测量共用，buffer 地址恒定，
    # 不因新地址反复重录）。
    from transformers import StaticCache
    decode_models = {(b, i + o): torch.compile(model, **compile_kwargs)
                     for (b, i, o) in shapes}
    caches = {(b, i + o): StaticCache(config=model.config, max_cache_len=i + o)
              for (b, i, o) in shapes}

    print(f"[2_torch_compile] 首次调用会触发后端编译/图捕获（每个 shape 一次，"
          f"可能持续数分钟）...", flush=True)
    warmup(model, tokenizer, device, shapes, decode_models=decode_models,
           caches=caches)

    run_suite("2_torch_compile", model, tokenizer, device, args,
              decode_models=decode_models, caches=caches)


if __name__ == "__main__":
    main()
