# Contributing

Thanks for your interest in this portfolio project.

## Quick start

1. Install dependencies: GCC, CUDA Toolkit (11.x+), and cuBLAS.
2. Place MNIST files in the expected `data/` subdirectories shown in `README.md`.
3. Run one of the versions:

```bash
gcc -Wall -O2 -o v1_nn v1.c -lm && ./v1_nn
nvcc -O2 -o v2_nn v2.cu && ./v2_nn
nvcc -O2 -o v3_nn v3.cu && ./v3_nn
nvcc -O2 -arch=sm_80 -lcublas -o v4_nn v4.cu && ./v4_nn
```

## Reproducing benchmark results

- Run V1, V2, V3, and V4 from the same machine/environment.
- Record:
  - total training time
  - train/test accuracy
  - computed speedup vs V1
- Update the results table in `README.md` with your measured values.

## Contribution scope

- Keep changes focused and minimal.
- Prefer reproducibility/documentation improvements over broad refactors.
- For algorithmic changes, include before/after timing and accuracy notes.
