#include "ops.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vf {

// ── Internal validation helpers ───────────────────────────────────────────────

static void check_same_device(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::invalid_argument(
            "Both tensors must be on the same device "
            "(got " +
            std::string(a.device() == Device::CPU ? "cpu" : "cuda") + " and " +
            std::string(b.device() == Device::CPU ? "cpu" : "cuda") + ")");
    }
}

static void check_same_shape(const Tensor& a, const Tensor& b,
                              const char* op_name) {
    if (a.shape() != b.shape()) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            return s + "]";
        };
        throw std::invalid_argument(
            std::string(op_name) + ": shape mismatch " +
            shape_str(a) + " vs " + shape_str(b));
    }
}

// ── CPU implementations ───────────────────────────────────────────────────────

namespace cpu {

Tensor add(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b, "add");

    const int64_t  n  = a.numel();
    Tensor         out(a.shape());
    const float*   pa = a.data();
    const float*   pb = b.data();
    float*         pc = out.data();

    for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] + pb[i];
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b, "mul");

    const int64_t  n  = a.numel();
    Tensor         out(a.shape());
    const float*   pa = a.data();
    const float*   pb = b.data();
    float*         pc = out.data();

    for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] * pb[i];
    return out;
}

Tensor relu(const Tensor& a) {
    const int64_t  n  = a.numel();
    Tensor         out(a.shape());
    const float*   pa = a.data();
    float*         pc = out.data();

    for (int64_t i = 0; i < n; ++i)
        pc[i] = pa[i] > 0.0f ? pa[i] : 0.0f;
    return out;
}

// Naive O(M·K·N) triple-loop.
// a : [M, K]   b : [K, N]   →   out : [M, N]
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument(
            "matmul requires 2-D tensors (got ndim=" +
            std::to_string(a.ndim()) + " and " +
            std::to_string(b.ndim()) + ")");
    }

    const int64_t M  = a.shape()[0];
    const int64_t K  = a.shape()[1];
    const int64_t K2 = b.shape()[0];
    const int64_t N  = b.shape()[1];

    if (K != K2) {
        throw std::invalid_argument(
            "matmul: inner dimensions must match "
            "(a.shape[1]=" + std::to_string(K) +
            " != b.shape[0]=" + std::to_string(K2) + ")");
    }

    Tensor out({M, N});
    const float* pa = a.data();
    const float* pb = b.data();
    float*       pc = out.data();

    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += pa[i * K + k] * pb[k * N + j];
            pc[i * N + j] = acc;
        }
    return out;
}

}  // namespace cpu

// ── Dispatch layer ────────────────────────────────────────────────────────────
//
// Shape / device validation lives here; the cpu:: and cuda:: implementations
// receive pre-validated inputs.

Tensor add(const Tensor& a, const Tensor& b) {
    check_same_device(a, b);
    check_same_shape(a, b, "add");
    if (a.device() == Device::CPU) return cpu::add(a, b);
    return cuda::add(a, b);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    check_same_device(a, b);
    check_same_shape(a, b, "mul");
    if (a.device() == Device::CPU) return cpu::mul(a, b);
    return cuda::mul(a, b);
}

Tensor relu(const Tensor& a) {
    if (a.device() == Device::CPU) return cpu::relu(a);
    return cuda::relu(a);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    check_same_device(a, b);
    if (a.device() == Device::CPU) return cpu::matmul(a, b);
    throw std::runtime_error("vf::matmul: CUDA not yet implemented (T07)");
}

}  // namespace vf
