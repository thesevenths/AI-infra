# kernels — CUDA operator library

A from-scratch CUDA operator runtime that backs the transformer forward pass. It exists to give the `profiling/` stage a **controlled, inspectable substrate**: when attribution shows that GEMV, RMSNorm, or attention dominate decode time, the optimized replacement is built and micro-benchmarked here, against the same `Tensor` abstraction the rest of the stack uses.

This is intentionally a reference library — every operator ships with a CPU/naive baseline so that kernel correctness is checked against a ground-truth implementation before any performance claim is made.

## Layout

```
kernels/
├── operators/
│   ├── matrix.cu              # tiled GEMM: float4 vectorized loads, shared-memory tiling
│   ├── flash_attn/
│   │   ├── flash_attn.cu       # FlashAttention-2 reference (online-softmax attention)
│   │   └── self_attn.cu        # Q/K/V orchestration
│   ├── rms_norm.cu            # RMS LayerNorm (warp-shuffle + block reduce)
│   ├── softmax.cu             # numerically-stable row softmax
│   ├── reduce.cu              # reduction primitive
│   └── transpose.cu           # naive device-wide transpose
├── include/
│   ├── base/                  # memory / buffer / device-type abstractions
│   │   ├── alloc.h
│   │   ├── buffer.h
│   │   └── base.h             # DeviceType, DataType, ModelBufferType
│   └── tensor/
│       └── tensor.h           # Tensor descriptor
└── tensor/
    └── tensor.cpp             # Tensor construction + device allocation
```

## Operators

### GEMM — `operators/matrix.cu`
Shared-memory tiled matrix multiplication parameterized by compile-time `BM / BN / BK` and row strides. Global memory is loaded through `float4` vectorized transactions (`load_from_gmem`) and staged into shared memory before the compute phase, reducing global-memory traffic and improving coalescing. This is the kernel class that dominates decode (the M=1 "GEMV" case) and prefill (full GEMM) in `profiling/`.

### FlashAttention-2 — `operators/flash_attn/`
A reference FlashAttention-2 implementation: `flash_attention_v2_kernel` computes attention with online softmax (running max/sum), decomposed into `naive_nrow_gemm` (QKᵀ), `row_softmax`, and `naive_pv` (PV). `self_attn.cu` wires Q/K/V into the kernel. The reference is the correctness oracle against which fused vendor attention (SDPA / cuDNN) is compared.

### RMSNorm — `operators/rms_norm.cu`
Row-wise RMS normalization with an `eps` floor. The square-sum uses a warp-level `__shfl_down_sync` reduction followed by a block-level reduction in shared memory, then a single normalization pass with the learned weight. This is one of the per-step operators the profiler attributes in the decode hot path.

### Softmax — `operators/softmax.cu`
Numerically-stable row softmax (`max` subtraction before `exp`) with a CPU ground-truth `softmax_forward_cpu` used for validation.

### Reduce / Transpose — `operators/reduce.cu`, `operators/transpose.cu`
Reusable primitives (reduction, device-wide transpose) that compose the higher-level operators above.

## Memory model

`include/base/` provides the device/allocation abstractions (`DeviceType`, `DataType`, `ModelBufferType`, `alloc.h`, `buffer.h`) and `tensor/` provides the `Tensor` descriptor plus device allocation (`tensor.cpp`). Operators take `Tensor`-typed inputs/outputs, keeping the library decoupled from any specific model framework.

## Building & validating

Each operator ships a self-contained host driver (`main`) that:
1. allocates device buffers through the `Tensor`/`base` abstractions,
2. launches the CUDA kernel,
3. copies results back and checks them against the CPU baseline.

A typical build/run (requires an NVCC toolchain matching the target CUDA):

```bash
nvcc -O3 -arch=sm_61 operators/matrix.cu -o matrix_test && ./matrix_test
nvcc -O3 -arch=sm_61 operators/flash_attn/self_attn.cu -o fa2_test && ./fa2_test
# repeat per operator
```

Micro-benchmarks here are the bottom of the stack: a kernel that wins against its CPU baseline and against cuBLAS on the target SM is the candidate that `inference/` and `profiling/` then validate at serving scale.
