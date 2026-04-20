# Architecture Notes

This project compares four implementations of the same MNIST classification workflow:

1. **V1 (`v1.c`)**: CPU-only baseline
2. **V2 (`v2.cu`)**: naive CUDA kernels
3. **V3 (`v3.cu`)**: optimized CUDA kernels + launch tuning + pinned memory usage
4. **V4 (`v4.cu`)**: cuBLAS/Tensor-Core-oriented GEMM acceleration

## Data/Execution Flow

```mermaid
flowchart TD
    D[data/MNIST binaries] --> L[Load + normalize]
    L --> T[Train epochs]
    T --> E[Test evaluation]
    E --> M[Metrics collection]

    subgraph Implementations
      V1[V1: CPU]
      V2[V2: CUDA naive]
      V3[V3: CUDA optimized]
      V4[V4: cuBLAS/Tensor path]
    end

    L --> V1 --> T
    L --> V2 --> T
    L --> V3 --> T
    L --> V4 --> T
```

## Performance Engineering Themes

- Kernel parallelism for matrix math and activation/gradient operations
- Memory transfer overhead management (including pinned host memory in optimized paths)
- Runtime comparison of CPU vs. GPU using explicit timing instrumentation
