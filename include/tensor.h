#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vf {

// ── Device ────────────────────────────────────────────────────────────────────
// A device enum so that every header that includes tensor.h can use it.
enum class Device { CPU, CUDA };

// ── Tensor ────────────────────────────────────────────────────────────────────
// A CPU float32 tensor with row-major (C-order) strides.
//
// Shape  [2, 3, 4]  →  strides [12, 4, 1]  (units: elements)
//
class Tensor {
public:
    // ── Constructors ──────────────────────────────────────────────────────────

    // Zero-initialised tensor.
    explicit Tensor(std::vector<int64_t> shape, Device device = Device::CPU);

    // Tensor whose values are copied from `data`.
    // Throws if data.size() != numel().
    Tensor(std::vector<int64_t> shape,
           std::vector<float>   data,
           Device               device = Device::CPU);

    Tensor(const Tensor&)            = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&)                 = default;
    Tensor& operator=(Tensor&&)      = default;
    // ── Metadata ──────────────────────────────────────────────────────────────

    int64_t ndim()  const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const;

    const std::vector<int64_t>& shape()   const { return shape_;   }
    const std::vector<int64_t>& strides() const { return strides_; }
    Device                      device()  const { return device_;  }

    // ── Raw data access (CPU buffer) ──────────────────────────────────────────

    float*       data()       { return data_.data(); }
    const float* data() const { return data_.data(); }

    // ── Element access ────────────────────────────────────────────────────────
    // t.at({i, j, k})  — checks rank and per-dimension bounds.

    float& at(const std::vector<int64_t>& indices);
    float  at(const std::vector<int64_t>& indices) const;

    // ── Utilities ─────────────────────────────────────────────────────────────

    // Compact repr: "Tensor(shape=[2, 3], device=cpu, data=[0, 1, 2, ...])"
    std::string to_string() const;

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;  // row-major, in elements
    std::vector<float>   data_;
    Device               device_;

    void    compute_strides();
    int64_t flat_index(const std::vector<int64_t>& indices) const;
};

}  // namespace vf
