#include "ops.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vf {
namespace cuda {

// ── Error helpers ─────────────────────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
}

static void cublas_check(cublasStatus_t status, const char* ctx) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(ctx) +
                                 ": cuBLAS error " + std::to_string(status));
}

// ── cuBLAS handle — lazy singleton, lives for the process lifetime ─────────────

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublas_check(cublasCreate(&handle), "cublasCreate");
    }
    return handle;
}

// ── Grid helpers ──────────────────────────────────────────────────────────────

static constexpr int kBlockSize = 256;

static int grid(int64_t n) {
    return static_cast<int>((n + kBlockSize - 1) / kBlockSize);
}

// ── Kernels ───────────────────────────────────────────────────────────────────

__global__ void add_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ c,
                            int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void mul_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ c,
                            int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void relu_kernel(const float* __restrict__ a,
                             float*       __restrict__ c,
                             int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
}

// ── Host launchers ────────────────────────────────────────────────────────────

Tensor add(const Tensor& a, const Tensor& b) {
    const int64_t n = a.numel();
    Tensor out(a.shape(), Device::CUDA);

    add_kernel<<<grid(n), kBlockSize>>>(
        a.cuda_data(), b.cuda_data(), out.cuda_data(), n);
    cuda_check(cudaGetLastError(), "add_kernel launch");

    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    const int64_t n = a.numel();
    Tensor out(a.shape(), Device::CUDA);

    mul_kernel<<<grid(n), kBlockSize>>>(
        a.cuda_data(), b.cuda_data(), out.cuda_data(), n);
    cuda_check(cudaGetLastError(), "mul_kernel launch");

    return out;
}

Tensor relu(const Tensor& a) {
    const int64_t n = a.numel();
    Tensor out(a.shape(), Device::CUDA);

    relu_kernel<<<grid(n), kBlockSize>>>(
        a.cuda_data(), out.cuda_data(), n);
    cuda_check(cudaGetLastError(), "relu_kernel launch");

    return out;
}

// ── matmul via cuBLAS ─────────────────────────────────────────────────────────
//
// Our tensors are row-major.  cuBLAS expects column-major.
//
// Row-major A[M×K] is identical in memory to column-major A^T[K×M].
// So:  C_row = A_row * B_row
//   ↔  C^T_col[N×M] = B^T_col[N×K] * A^T_col[K×M]
//
// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
//             &alpha, b_data, N,   ← B^T in col-major, lda = N
//                     a_data, K,   ← A^T in col-major, ldb = K
//             &beta,  c_data, N)   ← C^T in col-major, ldc = N

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Shape validation is done by the dispatch layer before we're called.
    const int64_t M = a.shape()[0];
    const int64_t K = a.shape()[1];
    const int64_t N = b.shape()[1];

    Tensor out({M, N}, Device::CUDA);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublas_check(
        cublasSgemm(get_cublas_handle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N),   // m
                    static_cast<int>(M),   // n
                    static_cast<int>(K),   // k
                    &alpha,
                    b.cuda_data(), static_cast<int>(N),   // A (= B^T), lda
                    a.cuda_data(), static_cast<int>(K),   // B (= A^T), ldb
                    &beta,
                    out.cuda_data(), static_cast<int>(N)),  // C (= C^T), ldc
        "cublasSgemm");

    return out;
}

}  // namespace cuda
}  // namespace vf
