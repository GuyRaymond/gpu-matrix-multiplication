#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm> // For std::max

#define BLOCK_SIZE 16

using namespace std;

// CUDA kernel for matrix multiplication (using double)
__global__ void matrixMulKernel(double* C, const double* A, const double* B, int N, int NN) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double value = 0.0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void matrixMulGPU(double* C, const double* A, const double* B, int N, int NN) {
    double *d_A, *d_B, *d_C;
    size_t size = NN * sizeof(double);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, N, NN);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrixMulCPU(double* C, const double* A, const double* B, int N, int NN) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            double value = 0.0;
            for (int k = 0; k < N; ++k) {
                value += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }
}

// Benchmark function
template <typename T, typename... Parameters>
double benchmark(int runs, T&& func, Parameters&&... parameters) {
    vector<double> times;

    for (int i = 0; i < runs; ++i) {
        auto start = chrono::high_resolution_clock::now();
        forward<T>(func)(forward<Parameters>(parameters)...);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    // Calculate geometric mean
    double product = 1.0;
    for (double tps : times) {
        product *= tps;
    }
    double geometricMean = pow(product, 1.0 / times.size());

    return geometricMean;
}

// Function to benchmark a specific matrix size
void benchmarkMatrixSize(int N, int runs, vector<string>& col_1, vector<string>& col_2, vector<string>& col_3) {
    int NN = N * N; // Define NN = N * N
    size_t size = NN * sizeof(double);

    // Allocate host memory
    double *h_A = (double*)malloc(size);
    double *h_B = (double*)malloc(size);
    double *h_C_CPU = (double*)malloc(size);
    double *h_C_GPU = (double*)malloc(size);

    // Initialize matrices with small values to avoid overflow
    for (int i = 0; i < NN; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX; // Random value between 0 and 1
        h_B[i] = static_cast<double>(rand()) / RAND_MAX; // Random value between 0 and 1
    }

    // Benchmark CPU
    double cpuGeometricMean = benchmark(runs, matrixMulCPU, h_C_CPU, h_A, h_B, N, NN);

    // Benchmark GPU
    double gpuGeometricMean = benchmark(runs, matrixMulGPU, h_C_GPU, h_A, h_B, N, NN);

    // Determine the fastest implementation
    string fastestGeometric = (cpuGeometricMean < gpuGeometricMean) ? "CPU " + to_string(gpuGeometricMean / cpuGeometricMean) + "x" : "GPU " + to_string(cpuGeometricMean / gpuGeometricMean) + "x";

    // Append results to table columns
    col_1.push_back(to_string(N) + "x" + to_string(N));
    col_2.push_back(to_string(cpuGeometricMean) + " (CPU), " + to_string(gpuGeometricMean) + " (GPU)");
    col_3.push_back(fastestGeometric);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
}

// Function to build and print the Markdown table
void printMarkdownTable(const vector<string>& col_1, const vector<string>& col_2, const vector<string>& col_3) {
    // Find the maximum length of each column
    size_t col_1_max = 0, col_2_max = 0, col_3_max = 0;
    for (size_t i = 0; i < col_1.size(); ++i) {
        col_1_max = max(col_1_max, col_1[i].size());
        col_2_max = max(col_2_max, col_2[i].size());
        col_3_max = max(col_3_max, col_3[i].size());
    }

    // Build the separator
    string sep = "|" + string(col_1_max + 2, '-') + "|" + string(col_2_max + 2, '-') + "|" + string(col_3_max + 2, '-') + "|\n";

    // Print the table header
    cout << sep;
    cout << "| " << col_1[0] << string(col_1_max - col_1[0].size(), ' ') << " | "
         << col_2[0] << string(col_2_max - col_2[0].size(), ' ') << " | "
         << col_3[0] << string(col_3_max - col_3[0].size(), ' ') << " |\n";
    cout << sep;

    // Print the table rows
    for (size_t i = 1; i < col_1.size(); ++i) {
        cout << "| " << col_1[i] << string(col_1_max - col_1[i].size(), ' ') << " | "
             << col_2[i] << string(col_2_max - col_2[i].size(), ' ') << " | "
             << col_3[i] << string(col_3_max - col_3[i].size(), ' ') << " |\n";
    }
    cout << sep;
}

int main() {
    const vector<int> matrixSizes1 = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}; // First set of matrix sizes
    const vector<int> matrixSizes2 = {5, 10, 25, 50, 100, 125, 150, 200, 250, 500, 1000}; // Second set of matrix sizes
    const int runs = 100; // Number of runs for benchmarking

    // Initialize table columns
    vector<string> col_1 = {"Matrix Size"};
    vector<string> col_2 = {"Geometric Mean in second (CPU, GPU)"};
    vector<string> col_3 = {"Fastest Geometric Mean"};

    // Benchmark for the first set of matrix sizes
    for (int N : matrixSizes1) {
        benchmarkMatrixSize(N, runs, col_1, col_2, col_3);
    }

    // Print the first Markdown table
    cout << "### Performance Comparison (Set 1)\n";
    printMarkdownTable(col_1, col_2, col_3);

    // Benchmark for the second set of matrix sizes
    vector<string> col_1_2 = {"Matrix Size"};
    vector<string> col_2_2 = {"Geometric Mean in second (CPU, GPU)"};
    vector<string> col_3_2 = {"Fastest Geometric Mean"};

    for (int N : matrixSizes2) {
        benchmarkMatrixSize(N, runs, col_1_2, col_2_2, col_3_2);
    }

    // Print the second Markdown table
    cout << "\n### Performance Comparison (Set 2)\n";
    printMarkdownTable(col_1_2, col_2_2, col_3_2);

    return 0;
}
