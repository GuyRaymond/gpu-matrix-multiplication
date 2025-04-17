The following performance comparison, between CPU and GPU implementations, was conducted for the multiplication of two square matrices of the same size, ranging from 1x1 to 1024x1024.

## Prerequisites

- [Docker](https://docs.docker.com/desktop) installed on your system.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed to enable GPU support.

## Running the Solution

### 1. Clone this Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/guyraymond/gpu-maxtrix-multiplication.git
```

```bash
cd gpu-maxtrix-multiplication
```

Place your image in images/input.jpg.

### 2. Start Docker
```bash
docker desktop start
```

### 3. Build the Docker image

```bash
docker build -t maxtrix-multiplication .
```

### 4. Run the Docker container

```bash
docker run --gpus all gpu-matrix-multiplication
```


## Output

### Performance Comparison (Set 1)
| Matrix Size | Geometric Mean in second (CPU, GPU) | Fastest Geometric Mean |
|-------------|-------------------------------------|------------------------|
| 1x1         | 0.000000 (CPU), 0.002233 (GPU)      | CPU infx               |
| 2x2         | 0.000000 (CPU), 0.001778 (GPU)      | CPU infx               |
| 4x4         | 0.000000 (CPU), 0.002163 (GPU)      | CPU infx               |
| 8x8         | 0.000000 (CPU), 0.002134 (GPU)      | CPU infx               |
| 16x16       | 0.000000 (CPU), 0.002422 (GPU)      | CPU infx               |
| 32x32       | 0.000000 (CPU), 0.002184 (GPU)      | CPU infx               |
| 64x64       | 0.000000 (CPU), 0.001520 (GPU)      | CPU infx               |
| 128x128     | 0.003516 (CPU), 0.001490 (GPU)      | GPU 2.360692x          |
| 256x256     | 0.031331 (CPU), 0.002685 (GPU)      | GPU 11.668093x         |
| 512x512     | 0.399059 (CPU), 0.009497 (GPU)      | GPU 42.021511x         |
| 1024x1024   | 7.221411 (CPU), 0.025972 (GPU)      | GPU 278.048759x        |

### Performance Comparison (Set 2)
| Matrix Size | Geometric Mean in second (CPU, GPU) | Fastest Geometric Mean |
|-------------|-------------------------------------|------------------------|
| 5x5         | 0.000000 (CPU), 0.001262 (GPU)      | CPU infx               |
| 10x10       | 0.000000 (CPU), 0.001023 (GPU)      | CPU infx               |
| 25x25       | 0.000000 (CPU), 0.001124 (GPU)      | CPU infx               |
| 50x50       | 0.000000 (CPU), 0.001130 (GPU)      | CPU infx               |
| 100x100     | 0.001076 (CPU), 0.001319 (GPU)      | CPU 1.226486x          |
| 125x125     | 0.002334 (CPU), 0.001341 (GPU)      | GPU 1.740465x          |
| 150x150     | 0.004452 (CPU), 0.001400 (GPU)      | GPU 3.179260x          |
| 200x200     | 0.010681 (CPU), 0.002914 (GPU)      | GPU 3.665074x          |
| 250x250     | 0.022710 (CPU), 0.004050 (GPU)      | GPU 5.606940x          |
| 500x500     | 0.206672 (CPU), 0.008438 (GPU)      | GPU 24.492740x         |
| 1000x1000   | 1.725369 (CPU), 0.026168 (GPU)      | GPU 65.933495x         |

## Conclusion
The smaller the matrix size, the faster the computation.
For matrix sizes less than or equal to 100x100, the CPU implementation is the fastest.
For matrix sizes greater than or equal to 125x125, the GPU implementation is the fastest.
Having a power of 2 as the size of the rows and columns does not seem to provide any significant benefits.
