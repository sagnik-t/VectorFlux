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
//   matmul CUDA path throws "not yet implemented" until T07 adds cuBLAS.

Tensor add   (const Tensor& a, const Tensor& b);
Tensor mul   (const Tensor& a, const Tensor& b);
Tensor relu  (const Tensor& a);
Tensor matmul(const Tensor& a, const Tensor& b);

// ── cpu:: — host implementations (used by the dispatch layer) ─────────────────
//
// External callers should prefer the top-level vf:: functions.
// These are exposed mainly so tests can target the CPU path directly.

namespace cpu {
    Tensor add   (const Tensor& a, const Tensor& b);
    Tensor mul   (const Tensor& a, const Tensor& b);
    Tensor relu  (const Tensor& a);
    Tensor matmul(const Tensor& a, const Tensor& b);
}  // namespace cpu

// ── cuda:: — device implementations (T06: add/mul/relu; T07: matmul) ─────────
//
// Defined in src/ops_cuda.cu, compiled by nvcc.
// The dispatch layer in ops_cpu.cpp calls these after verifying device.

namespace cuda {
    Tensor add (const Tensor& a, const Tensor& b);
    Tensor mul (const Tensor& a, const Tensor& b);
    Tensor relu(const Tensor& a);
    // matmul added in T07 (cuBLAS)
}  // namespace cuda

}  // namespace vf
