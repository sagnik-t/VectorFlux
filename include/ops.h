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
//   add / mul   : a.shape == b.shape          (element-wise, no broadcasting)
//   relu        : any shape
//   matmul      : a=[M,K], b=[K,N] → [M,N]   (2-D only for now)
//
// Device rules
//   Both inputs must be on the same device.
//   CUDA path throws "not yet implemented" until T06/T07 add it.

Tensor add   (const Tensor& a, const Tensor& b);
Tensor mul   (const Tensor& a, const Tensor& b);
Tensor relu  (const Tensor& a);
Tensor matmul(const Tensor& a, const Tensor& b);

// ── cpu:: — device-specific implementations (used by the dispatch layer) ──────
//
// T06 will add a parallel  vf::cuda::  namespace in ops_cuda.cu.
// External callers should prefer the top-level vf:: functions; these are
// exposed mainly so tests can target the CPU path directly if needed.

namespace cpu {
    Tensor add   (const Tensor& a, const Tensor& b);
    Tensor mul   (const Tensor& a, const Tensor& b);
    Tensor relu  (const Tensor& a);
    Tensor matmul(const Tensor& a, const Tensor& b);
}  // namespace cpu

}  // namespace vf
