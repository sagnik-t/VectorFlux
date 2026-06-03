#include "ops.h"

#include <algorithm>
#include <cmath>
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

// ── T13: element-wise subtraction ────────────────────────────────────────────
Tensor sub(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b, "sub");
    const int64_t  n  = a.numel();
    Tensor         out(a.shape());
    const float*   pa = a.data();
    const float*   pb = b.data();
    float*         pc = out.data();
    for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] - pb[i];
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

// step(x) = (x > 0) ? 1.0 : 0.0  — subgradient convention: step(0) = 0
Tensor step(const Tensor& a) {
    const int64_t  n  = a.numel();
    Tensor         out(a.shape());
    const float*   pa = a.data();
    float*         pc = out.data();
    for (int64_t i = 0; i < n; ++i)
        pc[i] = pa[i] > 0.0f ? 1.0f : 0.0f;
    return out;
}

// ── T13: sigmoid ─────────────────────────────────────────────────────────────
// sigma(x) = 1 / (1 + exp(-x))
Tensor sigmoid(const Tensor& a) {
    const int64_t n  = a.numel();
    Tensor        out(a.shape());
    const float*  pa = a.data();
    float*        pc = out.data();
    for (int64_t i = 0; i < n; ++i)
        pc[i] = 1.0f / (1.0f + std::exp(-pa[i]));
    return out;
}

// ── T13: tanh ─────────────────────────────────────────────────────────────────
Tensor tanh(const Tensor& a) {
    const int64_t n  = a.numel();
    Tensor        out(a.shape());
    const float*  pa = a.data();
    float*        pc = out.data();
    for (int64_t i = 0; i < n; ++i)
        pc[i] = std::tanh(pa[i]);
    return out;
}

// ── T13: softmax ─────────────────────────────────────────────────────────────
// For 1-D [C]:     normalise the whole vector.
// For 2-D [C, N]:  normalise each column independently (axis 0).
// Numerically stable: subtract column max before exp.
Tensor softmax(const Tensor& a) {
    if (a.ndim() == 1) {
        const int64_t C  = a.shape()[0];
        Tensor        out(a.shape());
        const float*  pa = a.data();
        float*        pc = out.data();

        float max_val = *std::max_element(pa, pa + C);
        float sum = 0.0f;
        for (int64_t i = 0; i < C; ++i) {
            pc[i] = std::exp(pa[i] - max_val);
            sum  += pc[i];
        }
        for (int64_t i = 0; i < C; ++i) pc[i] /= sum;
        return out;

    } else if (a.ndim() == 2) {
        const int64_t C  = a.shape()[0];
        const int64_t N  = a.shape()[1];
        Tensor        out(a.shape());
        const float*  pa = a.data();
        float*        pc = out.data();

        for (int64_t j = 0; j < N; ++j) {
            // Find column max
            float max_val = pa[j];                          // row 0, col j
            for (int64_t i = 1; i < C; ++i)
                max_val = std::max(max_val, pa[i * N + j]);

            // Exp and sum
            float sum = 0.0f;
            for (int64_t i = 0; i < C; ++i) {
                pc[i * N + j] = std::exp(pa[i * N + j] - max_val);
                sum           += pc[i * N + j];
            }
            // Normalise
            for (int64_t i = 0; i < C; ++i)
                pc[i * N + j] /= sum;
        }
        return out;

    } else {
        throw std::invalid_argument(
            "softmax: expected 1-D or 2-D tensor (got ndim=" +
            std::to_string(a.ndim()) + ")");
    }
}

// Naive O(M·K·N) triple-loop.
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

// Row-major transpose: [M, N] → [N, M]
Tensor transpose(const Tensor& a) {
    if (a.ndim() != 2) {
        throw std::invalid_argument(
            "transpose requires a 2-D tensor (got ndim=" +
            std::to_string(a.ndim()) + ")");
    }
    const int64_t M  = a.shape()[0];
    const int64_t N  = a.shape()[1];
    Tensor        out({N, M});
    const float*  pa = a.data();
    float*        pc = out.data();
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
            pc[j * M + i] = pa[i * N + j];
    return out;
}

}  // namespace cpu

// ── Dispatch layer ────────────────────────────────────────────────────────────

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

Tensor sub(const Tensor& a, const Tensor& b) {
    check_same_device(a, b);
    check_same_shape(a, b, "sub");
    if (a.device() == Device::CPU) return cpu::sub(a, b);
    return cuda::sub(a, b);
}

Tensor relu(const Tensor& a) {
    if (a.device() == Device::CPU) return cpu::relu(a);
    return cuda::relu(a);
}

Tensor sigmoid(const Tensor& a) {
    if (a.device() == Device::CPU) return cpu::sigmoid(a);
    return cuda::sigmoid(a);
}

Tensor tanh(const Tensor& a) {
    if (a.device() == Device::CPU) return cpu::tanh(a);
    return cuda::tanh(a);
}

Tensor softmax(const Tensor& a) {
    if (a.device() == Device::CPU) return cpu::softmax(a);
    return cuda::softmax(a);
}

Tensor step(const Tensor& a) {
    if (a.device() == Device::CPU) return cpu::step(a);
    return cuda::step(a);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    check_same_device(a, b);
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument(
            "matmul requires 2-D tensors (got ndim=" +
            std::to_string(a.ndim()) + " and " +
            std::to_string(b.ndim()) + ")");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument(
            "matmul: inner dimensions must match "
            "(a.shape[1]=" + std::to_string(a.shape()[1]) +
            " != b.shape[0]=" + std::to_string(b.shape()[0]) + ")");
    }
    if (a.device() == Device::CPU) return cpu::matmul(a, b);
    return cuda::matmul(a, b);
}

Tensor transpose(const Tensor& a) {
    if (a.ndim() != 2) {
        throw std::invalid_argument(
            "transpose requires a 2-D tensor (got ndim=" +
            std::to_string(a.ndim()) + ")");
    }
    if (a.device() == Device::CPU) return cpu::transpose(a);
    return cuda::transpose(a);
}

}  // namespace vf
