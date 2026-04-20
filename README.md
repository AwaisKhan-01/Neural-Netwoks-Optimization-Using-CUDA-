# Neural Networks Optimization Using CUDA

This project is a performance-focused implementation of MNIST digit classification that compares a baseline CPU training loop with progressively optimized CUDA implementations. It is designed as an AI engineering portfolio project to show practical model training, profiling, and systems-level optimization work in one place.

The core AI/ML workload is a 2-layer neural network (`784 -> 128 -> 10`) trained on MNIST. CUDA acceleration is applied to forward/backward pass kernels, memory transfer strategy, stream usage, and (in V4) Tensor Core-backed matrix operations through cuBLAS.

## Why this matters

For AI engineering roles, shipping models is not only about accuracy—it is also about training/inference efficiency, GPU utilization, and reproducible performance experiments. This repository highlights those trade-offs by comparing multiple GPU optimization strategies side-by-side.

## Implementations in this repo

- **V1 (`v1.c`)**: Sequential CPU baseline
- **V2 (`v2.cu`)**: Naive CUDA implementation
- **V3 (`v3.cu`)**: Optimized CUDA (kernel launch tuning, streams, memory strategy)
- **V4 (`v4.cu`)**: Advanced CUDA with **cuBLAS/Tensor Cores (TF32)**

## Reproducible setup

### Requirements

- Linux (Ubuntu tested)
- GCC (for V1)
- NVIDIA CUDA Toolkit 11.x+ (for V2/V3/V4)
- cuBLAS (for V4)
- NVIDIA GPU (V2/V3 on most CUDA-capable GPUs, **V4 expects Ampere+ for TF32 Tensor Core path**)

### Verify environment

```bash
gcc --version
nvcc --version
nvidia-smi
```

### Dataset layout (required by current code)

The code expects MNIST files in this exact folder structure:

```text
data/
  train-images-idx3-ubyte/train-images-idx3-ubyte
  train-labels-idx1-ubyte/train-labels-idx1-ubyte
  t10k-images-idx3-ubyte/t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte
```

MNIST source: <https://www.kaggle.com/datasets/hojjatk/mnist-dataset>

## Run training, evaluation, and benchmarks

> Each executable runs end-to-end training for 3 epochs and then evaluation on the MNIST test set.

### V1 — CPU baseline

```bash
gcc -Wall -O2 -o v1_nn v1.c -lm
./v1_nn
```

### V2 — Naive CUDA

```bash
nvcc -O2 -o v2_nn v2.cu
./v2_nn
```

### V3 — Optimized CUDA

```bash
nvcc -O2 -o v3_nn v3.cu
./v3_nn
```

### V4 — Tensor Core/cuBLAS path

```bash
nvcc -O2 -arch=sm_80 -lcublas -o v4_nn v4.cu
./v4_nn
```

### Optional: existing profiling flow for V1

```bash
make -f makefile run
```

## Results (portfolio view)

### Benchmark summary

| Version | Device Path | Relative Speedup vs V1 | Test Accuracy |
|---|---|---:|---:|
| V1 | CPU | 1.00x | ~96.9% |
| V2 | Naive CUDA | _Fill after run_ | _Fill after run_ |
| V3 | Optimized CUDA | ~3.82x (reported) | ~96.2% |
| V4 | Tensor Core + cuBLAS | ~4.51x (reported) | ~91.9% (TF32 trade-off) |

### Figures (placeholders)

- Training time comparison chart: `docs/figures/training-time-placeholder.png`
- Speedup chart: `docs/figures/speedup-placeholder.png`
- Accuracy vs speed trade-off chart: `docs/figures/accuracy-speed-placeholder.png`

## Demo

- Demo GIF placeholder: `docs/figures/demo-placeholder.gif`
- Demo video placeholder: `https://example.com/demo-video`

## Directory structure

```text
.
├── v1.c
├── v2.cu
├── v3.cu
├── v4.cu
├── makefile
├── data/
├── docs/
│   └── figures/
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## What I learned (AI engineering)

- Profiling first is essential: identify where GPU acceleration actually helps before kernel tuning.
- Memory bandwidth and transfer patterns are often the bottleneck, not only raw FLOPs.
- Kernel launch configuration and stream strategy can materially impact throughput.
- Tensor Core/TF32 acceleration can improve speed but may require accuracy trade-off analysis.
- Reproducibility (fixed setup + clear commands + benchmark tables) is critical for fair model-system comparisons.
