#include "ops.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vf {

// ── CUDA error helpers ────────────────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": " +
                                 cudaGetErrorString(err));
}

static void cublas_check(cublasStatus_t st, const char* ctx) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(ctx) +
                                 ": cuBLAS error " + std::to_string(st));
}

// ── cuBLAS handle (lazy singleton) ────────────────────────────────────────────

static cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        cublas_check(cublasCreate(&h), "cublasCreate");
        return h;
    }();
    return handle;
}

// ── Element-wise kernels (T06) ────────────────────────────────────────────────

__global__ void add_kernel(const float* a, const float* b, float* out,
                            int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* out,
                            int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void relu_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

// ── Gradient-support kernels (T10) ────────────────────────────────────────────

__global__ void step_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0f ? 1.0f : 0.0f;
}

__global__ void transpose_kernel(const float* A, float* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        B[col * M + row] = A[row * N + col];
}

// ── T13: new element-wise kernels ─────────────────────────────────────────────

__global__ void sub_kernel(const float* a, const float* b, float* out,
                            int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void sigmoid_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}

__global__ void tanh_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(a[i]);
}

// Softmax kernel: one thread per column (sample), sequential over C classes.
// Input layout: row-major [C, N] — element (i, j) at index i*N + j.
// Normalises each column j independently (axis 0), numerically stable.
__global__ void softmax_kernel(const float* in, float* out, int C, int N) {
    int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= N) return;

    // Find column max
    float max_val = in[j];
    for (int i = 1; i < C; ++i) {
        float v = in[i * N + j];
        if (v > max_val) max_val = v;
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        out[i * N + j] = expf(in[i * N + j] - max_val);
        sum            += out[i * N + j];
    }

    // Normalise
    for (int i = 0; i < C; ++i)
        out[i * N + j] /= sum;
}

// ── CUDA namespace implementations ────────────────────────────────────────────

namespace cuda {

// ── Helper: launch element-wise 1-D kernel ────────────────────────────────────

static Tensor elementwise_binary(const float* da, const float* db,
                                  const std::vector<int64_t>& shape,
                                  int64_t n,
                                  void(*kernel)(const float*, const float*,
                                                float*, int64_t)) {
    Tensor out(shape, Device::CUDA);
    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    kernel<<<blocks, threads>>>(da, db, out.cuda_data(), n);
    cuda_check(cudaDeviceSynchronize(), "elementwise_binary sync");
    return out;
}

static Tensor elementwise_unary(const float* da,
                                 const std::vector<int64_t>& shape,
                                 int64_t n,
                                 void(*kernel)(const float*, float*, int64_t)) {
    Tensor out(shape, Device::CUDA);
    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    kernel<<<blocks, threads>>>(da, out.cuda_data(), n);
    cuda_check(cudaDeviceSynchronize(), "elementwise_unary sync");
    return out;
}

// ── T06: element-wise ops ─────────────────────────────────────────────────────

Tensor add(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a.cuda_data(), b.cuda_data(),
                               a.shape(), a.numel(), add_kernel);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a.cuda_data(), b.cuda_data(),
                               a.shape(), a.numel(), mul_kernel);
}

Tensor relu(const Tensor& a) {
    return elementwise_unary(a.cuda_data(), a.shape(), a.numel(), relu_kernel);
}

// ── T13: sub, sigmoid, tanh, softmax ─────────────────────────────────────────

Tensor sub(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a.cuda_data(), b.cuda_data(),
                               a.shape(), a.numel(), sub_kernel);
}

Tensor sigmoid(const Tensor& a) {
    return elementwise_unary(a.cuda_data(), a.shape(), a.numel(), sigmoid_kernel);
}

Tensor tanh(const Tensor& a) {
    return elementwise_unary(a.cuda_data(), a.shape(), a.numel(), tanh_kernel);
}

Tensor softmax(const Tensor& a) {
    // Treat 1-D [C] as [C, 1] so the column-based kernel handles it uniformly.
    int C, N;
    if (a.ndim() == 1) {
        C = static_cast<int>(a.shape()[0]);
        N = 1;
    } else if (a.ndim() == 2) {
        C = static_cast<int>(a.shape()[0]);
        N = static_cast<int>(a.shape()[1]);
    } else {
        throw std::runtime_error(
            "softmax: expected 1-D or 2-D tensor (got ndim=" +
            std::to_string(a.ndim()) + ")");
    }

    Tensor out(a.shape(), Device::CUDA);
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(a.cuda_data(), out.cuda_data(), C, N);
    cuda_check(cudaDeviceSynchronize(), "softmax sync");
    return out;
}

// ── T07: cuBLAS matmul ────────────────────────────────────────────────────────

Tensor matmul(const Tensor& a, const Tensor& b) {
    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);

    Tensor out({static_cast<int64_t>(M), static_cast<int64_t>(N)},
               Device::CUDA);

    const float alpha = 1.0f, beta = 0.0f;

    cublas_check(
        cublasSgemm(cublas_handle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    b.cuda_data(), N,
                    a.cuda_data(), K,
                    &beta,
                    out.cuda_data(), N),
        "cublasSgemm");

    return out;
}

// ── T10: step and transpose ───────────────────────────────────────────────────

Tensor step(const Tensor& a) {
    return elementwise_unary(a.cuda_data(), a.shape(), a.numel(), step_kernel);
}

Tensor transpose(const Tensor& a) {
    const int M = static_cast<int>(a.shape()[0]);
    const int N = static_cast<int>(a.shape()[1]);

    Tensor out({static_cast<int64_t>(N), static_cast<int64_t>(M)},
               Device::CUDA);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    transpose_kernel<<<blocks, threads>>>(a.cuda_data(), out.cuda_data(), M, N);
    cuda_check(cudaDeviceSynchronize(), "transpose sync");

    return out;
}

}  // namespace cuda
}  // namespace vf
