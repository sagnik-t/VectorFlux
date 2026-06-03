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

__global__ void neg_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

__global__ void sigmoid_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}

__global__ void tanh_kernel(const float* a, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(a[i]);
}

// Softmax: one thread per column, sequential over C classes.
// Layout: row-major [C, N] — element (i, j) at index i*N + j.
__global__ void softmax_kernel(const float* in, float* out, int C, int N) {
    int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= N) return;

    float max_val = in[j];
    for (int i = 1; i < C; ++i) {
        float v = in[i * N + j];
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        out[i * N + j] = expf(in[i * N + j] - max_val);
        sum            += out[i * N + j];
    }

    for (int i = 0; i < C; ++i)
        out[i * N + j] /= sum;
}

// ── T14: reduction kernels ────────────────────────────────────────────────────

// Parallel reduction using shared memory + atomicAdd.
// Requires dynamic shared memory: launch as kernel<<<B, T, T*sizeof(float)>>>
__global__ void reduce_sum_kernel(const float* in, float* out, int64_t n) {
    extern __shared__ float sdata[];
    int64_t tid = threadIdx.x;
    int64_t i   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    sdata[tid]  = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

// Fill kernel: sets every element of `out` to `val`.
__global__ void fill_kernel(float* out, float val, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = val;
}

// ── T14: per-sample CE loss kernel ────────────────────────────────────────────
// One thread per sample (column j).  Stores per-sample CE into out_per_sample[j].
// L_j = log_sum_exp(logits[:,j]) - dot(labels[:,j], logits[:,j])
__global__ void softmax_ce_forward_kernel(const float* logits,
                                           const float* labels,
                                           float* out_per_sample,
                                           int C, int N) {
    int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= N) return;

    // log-sum-exp for column j
    float max_val = logits[j];
    for (int i = 1; i < C; ++i)
        max_val = fmaxf(max_val, logits[i * N + j]);

    float sum_exp = 0.0f;
    for (int i = 0; i < C; ++i)
        sum_exp += expf(logits[i * N + j] - max_val);
    float log_sum_exp = logf(sum_exp) + max_val;

    // dot(labels[:,j], logits[:,j])
    float dot = 0.0f;
    for (int i = 0; i < C; ++i)
        dot += labels[i * N + j] * logits[i * N + j];

    out_per_sample[j] = log_sum_exp - dot;
}

// ── T14: fused softmax-CE gradient kernel ─────────────────────────────────────
// One thread per element (C*N total).
// grad_i = (softmax(logits)_i - labels_i) * scale
// Recomputes softmax per column — O(C^2 * N) but fine for small C.
__global__ void softmax_ce_grad_kernel(const float* logits,
                                        const float* labels,
                                        float* out,
                                        float scale,
                                        int C, int N) {
    int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(C) * N;
    if (idx >= total) return;

    int col = static_cast<int>(idx % N);   // which sample

    // Recompute softmax for this column
    float max_val = logits[col];
    for (int r = 1; r < C; ++r)
        max_val = fmaxf(max_val, logits[static_cast<int64_t>(r) * N + col]);

    float sum_exp = 0.0f;
    for (int r = 0; r < C; ++r)
        sum_exp += expf(logits[static_cast<int64_t>(r) * N + col] - max_val);

    float sm_val = expf(logits[idx] - max_val) / sum_exp;
    out[idx] = (sm_val - labels[idx]) * scale;
}

// ── CUDA namespace implementations ────────────────────────────────────────────

namespace cuda {

// ── Helpers ───────────────────────────────────────────────────────────────────

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

Tensor neg(const Tensor& a) {
    return elementwise_unary(a.cuda_data(), a.shape(), a.numel(), neg_kernel);
}

Tensor sigmoid(const Tensor& a) {
    return elementwise_unary(a.cuda_data(), a.shape(), a.numel(), sigmoid_kernel);
}

Tensor tanh(const Tensor& a) {
    return elementwise_unary(a.cuda_data(), a.shape(), a.numel(), tanh_kernel);
}

Tensor softmax(const Tensor& a) {
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

// ── T14: reduce_sum ───────────────────────────────────────────────────────────
// Parallel shared-memory reduction. Output is a [1] CUDA tensor.
Tensor reduce_sum(const Tensor& a) {
    const int64_t n = a.numel();

    Tensor out({1}, Device::CUDA);
    // Zero the accumulator before atomic adds
    cuda_check(cudaMemset(out.cuda_data(), 0, sizeof(float)), "reduce_sum memset");

    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        a.cuda_data(), out.cuda_data(), n);
    cuda_check(cudaDeviceSynchronize(), "reduce_sum sync");
    return out;
}

// ── T14: reduce_mean ─────────────────────────────────────────────────────────
// reduce_sum then scale in-place with a fill_kernel on a [1] tensor.
Tensor reduce_mean(const Tensor& a) {
    const int64_t n   = a.numel();
    Tensor        out = cuda::reduce_sum(a);      // [1] CUDA tensor with sum
    const float   inv = 1.0f / static_cast<float>(n);
    // Multiply the single element by 1/N
    fill_kernel<<<1, 1>>>(out.cuda_data(), 0.0f, 1);  // dummy to reuse fill
    // Actually: we need scale, not fill. Use a multiply-in-place approach:
    // read scalar, multiply, write back — simplest with a tiny host round-trip.
    float s;
    cuda_check(cudaMemcpy(&s, out.cuda_data(), sizeof(float),
                           cudaMemcpyDeviceToHost), "reduce_mean copy D2H");
    s *= inv;
    cuda_check(cudaMemcpy(out.cuda_data(), &s, sizeof(float),
                           cudaMemcpyHostToDevice), "reduce_mean copy H2D");
    return out;
}

// ── T14: softmax_cross_entropy_with_logits ────────────────────────────────────
// Launches per-sample CE kernel, then calls reduce_mean on the result.
Tensor softmax_cross_entropy_with_logits(const Tensor& logits,
                                          const Tensor& labels) {
    if (logits.ndim() != 2)
        throw std::runtime_error(
            "softmax_cross_entropy_with_logits: logits must be 2-D [C, N]");

    const int C = static_cast<int>(logits.shape()[0]);
    const int N = static_cast<int>(logits.shape()[1]);

    // Compute per-sample CE into a [N] buffer
    Tensor per_sample({static_cast<int64_t>(N)}, Device::CUDA);
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    softmax_ce_forward_kernel<<<blocks, threads>>>(
        logits.cuda_data(), labels.cuda_data(),
        per_sample.cuda_data(), C, N);
    cuda_check(cudaDeviceSynchronize(), "softmax_ce forward sync");

    // Mean over samples — reuses cuda::reduce_mean
    return cuda::reduce_mean(per_sample);
}

// ── T14: reduce_sum_grad ──────────────────────────────────────────────────────
// Fill input.shape with g_scalar[0].
Tensor reduce_sum_grad(const Tensor& input, const Tensor& g_scalar) {
    float g;
    cuda_check(cudaMemcpy(&g, g_scalar.cuda_data(), sizeof(float),
                           cudaMemcpyDeviceToHost), "reduce_sum_grad D2H");
    const int64_t n = input.numel();
    Tensor out(input.shape(), Device::CUDA);
    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    fill_kernel<<<blocks, threads>>>(out.cuda_data(), g, n);
    cuda_check(cudaDeviceSynchronize(), "reduce_sum_grad sync");
    return out;
}

// ── T14: reduce_mean_grad ─────────────────────────────────────────────────────
// Fill input.shape with g_scalar[0] / N.
Tensor reduce_mean_grad(const Tensor& input, const Tensor& g_scalar) {
    float g;
    cuda_check(cudaMemcpy(&g, g_scalar.cuda_data(), sizeof(float),
                           cudaMemcpyDeviceToHost), "reduce_mean_grad D2H");
    const int64_t n     = input.numel();
    const float   scale = g / static_cast<float>(n);
    Tensor out(input.shape(), Device::CUDA);
    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);
    fill_kernel<<<blocks, threads>>>(out.cuda_data(), scale, n);
    cuda_check(cudaDeviceSynchronize(), "reduce_mean_grad sync");
    return out;
}

// ── T14: softmax_ce_grad ──────────────────────────────────────────────────────
// (softmax(logits) - labels) * g / N
Tensor softmax_ce_grad(const Tensor& logits, const Tensor& labels,
                        const Tensor& g_scalar) {
    float g;
    cuda_check(cudaMemcpy(&g, g_scalar.cuda_data(), sizeof(float),
                           cudaMemcpyDeviceToHost), "softmax_ce_grad D2H");

    const int     C     = static_cast<int>(logits.shape()[0]);
    const int     N     = static_cast<int>(logits.shape()[1]);
    const float   scale = g / static_cast<float>(N);
    const int64_t total = static_cast<int64_t>(C) * N;

    Tensor out(logits.shape(), Device::CUDA);
    const int threads = 256;
    const int blocks  = static_cast<int>((total + threads - 1) / threads);
    softmax_ce_grad_kernel<<<blocks, threads>>>(
        logits.cuda_data(), labels.cuda_data(), out.cuda_data(), scale, C, N);
    cuda_check(cudaDeviceSynchronize(), "softmax_ce_grad sync");
    return out;
}

}  // namespace cuda
}  // namespace vf
