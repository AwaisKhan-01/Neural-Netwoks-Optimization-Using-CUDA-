# Contributing Guide

Thank you for your interest in improving this CUDA optimization project.

## Local Setup

1. Clone the repository and move to project root.
2. Ensure MNIST files are present under `data/` (see `README.md`).
3. Install required tools:
   - GCC + GNU Make
   - NVIDIA CUDA Toolkit (`nvcc`)
   - cuBLAS (for `v4.cu`)
   - Optional: Graphviz (`dot`) for call graph image generation

## Build & Run

### CPU baseline (V1)

```bash
make -f makefile clean
make -f makefile run
```

### CUDA variants

```bash
nvcc -O2 -o v2.out v2.cu && ./v2.out
nvcc -O2 -o v3.out v3.cu && ./v3.out
nvcc -arch=sm_80 -O2 -lcublas -o v4.out v4.cu && ./v4.out
```

## Benchmarks / Validation

This repository currently uses benchmark-style validation instead of unit tests.

- Capture run output and include key metrics in your PR:
  - `CPU Total Time`
  - `GPU Total Time`
  - `Test Accuracy`
- Run each benchmark multiple times and report average behavior when possible.

## Style Expectations

- Keep changes focused and minimal.
- Follow existing C/CUDA code style in touched files.
- Preserve compile flags and warning behavior unless change is justified.
- Do not commit generated binaries, profiling outputs, or temporary logs.

## Pull Request Expectations

When opening a PR, include:

1. What changed and why.
2. Which variant(s) were affected (V1/V2/V3/V4).
3. Build/run commands used for validation.
4. Before/after benchmark metrics (if performance-related).
