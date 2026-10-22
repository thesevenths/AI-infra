"""
stream_load.py -- 低提交内存环境下的顺序流式模型加载

适用场景：在提交内存（committed memory）紧张的 Windows 主机上，
transformers 的 from_pretrained 会先对 model.safetensors 做整体
view-mapping（metadata 探测），在提交内存不足时抛
OSError "页面文件太小 (os error 1455)"，且重试通常无法稳定通过。

方案：完全绕开 safetensors 的 mmap——纯文件顺序读，逐个 tensor 经固定
大小的 CPU 跳板直接上 GPU，配合 accelerate.init_empty_weights 的 meta
模型骨架：
  - CPU 侧峰值 = 单个最大 tensor（Qwen-0.6B 的 embed_tokens 约
    151936 × 1024 × 2B ≈ 300 MiB）；
  - GPU 侧 = 权重本体（fp16 下约 1.2 GiB）；
  - RoPE inv_freq 等非持久 buffer 在权重赋值后按构造公式重算。

与 from_pretrained 的区别：不做权重复检 / tie 推断，仅用于**性能测量**
（本 profiling 只关心时序，输入为固定种子的随机 token id，与 throughput
benchmark 口径一致）。在提交内存充足时仍建议优先使用标准 from_pretrained。

用法：
    from stream_load import load_hf_model
    model = load_hf_model("F:/model/Qwen-0.6B", "cuda")
"""
import json
import os
import struct

import torch

_DTYPE_MAP = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}


def iter_safetensors(path, device=None, chunk_mb=16):
    """顺序读 safetensors：yield (name, tensor)，全程不 mmap。

    device 非空时用固定 chunk_mb 大小的 CPU 跳板分批 H2D，避免一次性
    分配大 buffer（办公机上 CUDA context 建立后，剩余提交内存不够装
    单个 296MiB 的 embedding）。
    """
    staging = None
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            nbytes = e - s
            dtype = _DTYPE_MAP[info["dtype"]]
            shape = info["shape"]
            f.seek(base + s)
            if device is None:
                buf = bytearray(f.read(nbytes))
                t = torch.frombuffer(buf, dtype=torch.uint8).view(dtype).reshape(shape)
                del buf
            else:
                t = torch.empty(shape, dtype=dtype, device=device)
                if nbytes <= chunk_mb * 2**20:
                    buf = bytearray(f.read(nbytes))
                    src = torch.frombuffer(buf, dtype=torch.uint8)
                    t.view(torch.uint8).reshape(-1)[:] = src.to(device)
                    del buf, src
                else:
                    if staging is None:
                        staging = torch.empty(chunk_mb * 2**20, dtype=torch.uint8,
                                              pin_memory=False)
                    dst_flat = t.view(torch.uint8).reshape(-1)
                    off = 0
                    while off < nbytes:
                        k = min(chunk_mb * 2**20, nbytes - off)
                        buf = f.read(k)
                        staging[:k].copy_(torch.frombuffer(bytearray(buf), dtype=torch.uint8))
                        dst_flat[off:off + k].copy_(staging[:k].to(device))
                        off += k
                    del buf
            yield name, t


@torch.no_grad()
def load_hf_model(model_path, device="cuda"):
    """meta 骨架 + 逐参数流式上卡。返回 eval 模式的 fp16 模型。"""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    ckpts = sorted(fn for fn in os.listdir(model_path)
                   if fn.endswith(".safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"{model_path} 下没有 .safetensors 权重")

    tgt = device if str(device).startswith("cuda") and torch.cuda.is_available() else None
    loaded = set()
    for fn in ckpts:
        for name, t in iter_safetensors(os.path.join(model_path, fn), tgt):
            parent, _, leaf = name.rpartition(".")
            try:
                sub = model.get_submodule(parent) if parent else model
            except AttributeError:
                continue                      # 不在模型里的键（少见）忽略
            old = getattr(sub, leaf, None)
            setattr(sub, leaf, torch.nn.Parameter(t, requires_grad=False))
            # lm_head/embed 共享权重：另一侧的 meta 占位直接 alias
            if getattr(config, "tie_word_embeddings", False) and \
                    name == "model.embed_tokens.weight" and hasattr(model, "lm_head"):
                if getattr(model.lm_head, "weight", None) is not None and \
                        model.lm_head.weight.is_meta:
                    model.lm_head.weight = getattr(sub, leaf)
            loaded.add(name)
            del old

    # RoPE inv_freq 等 meta buffer：按构造公式在目标设备上重算
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    for m in model.modules():
        cls = type(m).__name__
        if cls.endswith("RotaryEmbedding") and hasattr(m, "inv_freq"):
            theta = getattr(m, "rope_theta", getattr(config, "rope_theta", 10000.0))
            inv = 1.0 / (theta ** (torch.arange(
                0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
            m.register_buffer("inv_freq", inv, persistent=False)
            scaling = getattr(m, "attention_scaling", None)
            if scaling is not None and torch.is_tensor(scaling) and scaling.is_meta:
                m.register_buffer("attention_scaling",
                                  torch.tensor(1.0, device=device), persistent=False)

    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise RuntimeError(f"流式加载后仍有 meta 参数: {meta_left[:5]}")
    return model.eval().requires_grad_(False)
