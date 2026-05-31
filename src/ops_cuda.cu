#include "ops.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vf {
namespace cuda {

// ── Error helper ──────────────────────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
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
//
// Shape / device validation is already done by the dispatch layer in
// ops_cpu.cpp before these functions are called.

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

}  // namespace cuda
}  // namespace vf
