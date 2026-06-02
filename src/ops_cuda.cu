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

// Heaviside step: (x > 0) ? 1 : 0  — subgradient convention at x = 0 is 0.
__global__ void step_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0f ? 1.0f : 0.0f;
}

// Row-major 2-D transpose: A[M, N] → B[N, M]
__global__ void transpose_kernel(const float* A, float* B,
                                  int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        B[col * M + row] = A[row * N + col];
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

// ── T07: cuBLAS matmul ────────────────────────────────────────────────────────
//
// Row-major trick: treating a row-major A[M,K] as a column-major matrix gives
// A^T[K,M].  cuBLAS computes (in col-major notation):
//   C_col = B_col * A_col   ≡   C_row = A_row @ B_row
// Call signature (N, M, K) with B leading A in the argument list.

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
                    b.cuda_data(), N,   // B (row-major) → treated as B^T col-major
                    a.cuda_data(), K,   // A (row-major) → treated as A^T col-major
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
