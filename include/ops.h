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
//   reduce_sum      : any shape         → [1]    (sum all elements)
//   reduce_mean     : any shape         → [1]    (mean all elements)
//   softmax_cross_entropy_with_logits:
//                     logits=[C,N], labels=[C,N] → [1]  (fused, numerically stable)
//
// Gradient-support ops (only appear in backward graphs, no public gradient):
//   reduce_sum_grad  : [input, g_scalar]           → same shape as input
//   reduce_mean_grad : [input, g_scalar]           → same shape as input
//   softmax_ce_grad  : [logits, labels, g_scalar]  → same shape as logits
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

// ── T14: reduction ops ────────────────────────────────────────────────────────
Tensor reduce_sum (const Tensor& a);                  // sum all elements → [1]
Tensor reduce_mean(const Tensor& a);                  // mean all elements → [1]

// ── T14: fused loss op ────────────────────────────────────────────────────────
// logits: [C, N]  (C classes, N samples)
// labels: [C, N]  (one-hot or soft labels, sum-to-1 per column)
// Returns scalar mean cross-entropy over N samples.
Tensor softmax_cross_entropy_with_logits(const Tensor& logits,
                                          const Tensor& labels);

// ── T14: gradient support ops (terminal — no higher-order gradient) ───────────
// These only appear in backward graphs; not intended for direct use.
Tensor reduce_sum_grad (const Tensor& input, const Tensor& g_scalar);
Tensor reduce_mean_grad(const Tensor& input, const Tensor& g_scalar);
Tensor softmax_ce_grad (const Tensor& logits, const Tensor& labels,
                        const Tensor& g_scalar);
Tensor neg(const Tensor& a);   // negate all elements — used in SubOp::gradient

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
    // T14
    Tensor reduce_sum (const Tensor& a);
    Tensor reduce_mean(const Tensor& a);
    Tensor softmax_cross_entropy_with_logits(const Tensor& logits,
                                              const Tensor& labels);
    Tensor reduce_sum_grad (const Tensor& input, const Tensor& g_scalar);
    Tensor reduce_mean_grad(const Tensor& input, const Tensor& g_scalar);
    Tensor softmax_ce_grad (const Tensor& logits, const Tensor& labels,
                            const Tensor& g_scalar);
    Tensor neg(const Tensor& a);
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
    // T14
    Tensor reduce_sum (const Tensor& a);
    Tensor reduce_mean(const Tensor& a);
    Tensor softmax_cross_entropy_with_logits(const Tensor& logits,
                                              const Tensor& labels);
    Tensor reduce_sum_grad (const Tensor& input, const Tensor& g_scalar);
    Tensor reduce_mean_grad(const Tensor& input, const Tensor& g_scalar);
    Tensor softmax_ce_grad (const Tensor& logits, const Tensor& labels,
                            const Tensor& g_scalar);
    Tensor neg(const Tensor& a);
}  // namespace cuda

}  // namespace vf
