#pragma once

#include "tensor.h"

namespace vf {

// ── Public API — auto-dispatches on tensor.device() ───────────────────────────
//
// Usage:
//   Tensor c = vf::add(a, b);
//   Tensor y = vf::relu(vf::matmul(W, x));
//
// Shapes
//   add / mul / sub : a.shape == b.shape        (element-wise, no broadcasting)
//   relu / step     : any shape
//   sigmoid / tanh  : any shape
//   softmax         : 1-D [C] or 2-D [C, N] — normalises along axis 0
//   matmul          : a=[M,K], b=[K,N] → [M,N]  (2-D only)
//   transpose       : a=[M,N]          → [N,M]  (2-D only)
//
// Device rules
//   Binary ops: both inputs must be on the same device.

Tensor add      (const Tensor& a, const Tensor& b);
Tensor mul      (const Tensor& a, const Tensor& b);
Tensor sub      (const Tensor& a, const Tensor& b);   // T13: a - b element-wise
Tensor relu     (const Tensor& a);
Tensor sigmoid  (const Tensor& a);                    // T13: 1 / (1 + exp(-x))
Tensor tanh     (const Tensor& a);                    // T13: tanh(x)
Tensor softmax  (const Tensor& a);                    // T13: exp(x)/sum(exp(x)) along axis 0
Tensor matmul   (const Tensor& a, const Tensor& b);
Tensor step     (const Tensor& a);   // Heaviside: (x > 0) ? 1 : 0
Tensor transpose(const Tensor& a);   // 2-D matrix transpose

// ── cpu:: — host implementations ──────────────────────────────────────────────

namespace cpu {
    Tensor add      (const Tensor& a, const Tensor& b);
    Tensor mul      (const Tensor& a, const Tensor& b);
    Tensor sub      (const Tensor& a, const Tensor& b);
    Tensor relu     (const Tensor& a);
    Tensor sigmoid  (const Tensor& a);
    Tensor tanh     (const Tensor& a);
    Tensor softmax  (const Tensor& a);
    Tensor matmul   (const Tensor& a, const Tensor& b);
    Tensor step     (const Tensor& a);
    Tensor transpose(const Tensor& a);
}  // namespace cpu

// ── cuda:: — device implementations ──────────────────────────────────────────

namespace cuda {
    Tensor add      (const Tensor& a, const Tensor& b);
    Tensor mul      (const Tensor& a, const Tensor& b);
    Tensor sub      (const Tensor& a, const Tensor& b);
    Tensor relu     (const Tensor& a);
    Tensor sigmoid  (const Tensor& a);
    Tensor tanh     (const Tensor& a);
    Tensor softmax  (const Tensor& a);
    Tensor matmul   (const Tensor& a, const Tensor& b);
    Tensor step     (const Tensor& a);
    Tensor transpose(const Tensor& a);
}  // namespace cuda

}  // namespace vf
