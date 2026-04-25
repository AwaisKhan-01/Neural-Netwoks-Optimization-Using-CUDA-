# Neural Network Optimization Using CUDA (MNIST)

Accelerate a simple neural network for **MNIST digit classification** with progressively optimized implementations—from a baseline **CPU** version to **Tensor Core** acceleration on modern NVIDIA GPUs.

## Highlights

- **Four implementations (V1–V4)** showing an optimization journey
- CUDA kernels, GPU timing, and performance/accuracy comparisons
- Optional **Tensor Core (TF32) GEMM** via **cuBLAS** (Ampere+)

## Project Structure / Versions

| Version | Target | Description |
|---|---|---|
| **V1** | CPU | Sequential baseline implementation |
| **V2** | GPU | Naive CUDA implementation |
| **V3** | GPU | Optimized CUDA (dynamic configuration, streams) |
| **V4** | GPU | Tensor Cores via **cuBLAS** (TF32) |

## Requirements

### Hardware

- **V1:** Any CPU
- **V2–V3:** NVIDIA GPU with CUDA support
- **V4:** **NVIDIA Ampere (SM80) or newer** for Tensor Core TF32 path

### Software

- Linux (Ubuntu tested)
- `gcc` / `g++` (for V1)
- **CUDA Toolkit 11.x+**
- `nvcc`
- **cuBLAS** (for V4)

## Dataset (MNIST)

Download MNIST from:
- http://yann.lecun.com/exdb/mnist/

Place the following files under `data/`:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

## Build & Run

> Commands below assume you are in the repository root.

### V1 — Sequential CPU

```bash
make clean && make src/V1
./src/nn.exe
```

**Expected output:** epoch-wise loss/accuracy
- Training time: ~22.38s
- Test accuracy: ~96.78%

### V2 — Naive CUDA GPU

```bash
nvcc -O2 -o src/n src/v2.cu
./src/n
```

**Expected output:** CPU/GPU metrics and comparison
- GPU time (reported): ~183.16s

### V3 — Optimized CUDA GPU

```bash
nvcc -O2 -o src/V3/n src/v3.cu
./src/n
```

**Expected output:** optimization notes + metrics
- GPU time (reported): ~6.78s
- Speedup: ~3.82×
- Test accuracy: ~96.20%

### V4 — Tensor Cores (cuBLAS, TF32)

```bash
nvcc -arch=sm_80 -O2 -lcublas -o src/n src/v4.cu
./src/n
```

**Expected output:** Tensor Core details + metrics
- GPU time (reported): ~5.82s
- Speedup: ~4.51×
- Test accuracy: ~91.93%

## Notes

- **Accuracy vs speed:** V4 may show lower accuracy due to **TF32** behavior; V3 typically offers a better speed/accuracy trade-off.
- **Performance:** V3 and V4 outperform the CPU baseline; V2 is primarily educational and may be slower due to naive design.

## Troubleshooting

- Confirm CUDA and GPU availability:
  ```bash
  nvidia-smi
  nvcc --version
  ```
- Ensure the MNIST files are in `data/` with the exact names listed above.
- For V4, verify your GPU supports **SM80+** and that cuBLAS links correctly.

---
