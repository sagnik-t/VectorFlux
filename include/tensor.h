#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vf {

// ── Device ────────────────────────────────────────────────────────────────────
enum class Device { CPU, CUDA };

// ── Tensor ────────────────────────────────────────────────────────────────────
// A float32 tensor with row-major (C-order) strides.
// Lives on CPU (data_) or CUDA (cuda_data_); never both at once.
//
// Shape  [2, 3, 4]  →  strides [12, 4, 1]  (units: elements)
//
class Tensor {
public:
    // ── Constructors ──────────────────────────────────────────────────────────

    // Zero-initialised tensor on `device`.
    explicit Tensor(std::vector<int64_t> shape, Device device = Device::CPU);

    // Tensor initialised from host data.
    // `data` is always host memory; if device == CUDA it is uploaded immediately.
    // Throws if data.size() != numel().
    Tensor(std::vector<int64_t> shape,
           std::vector<float>   data,
           Device               device = Device::CPU);

    // ── Rule-of-five ──────────────────────────────────────────────────────────
    // cuda_data_ is a raw pointer — we must manage it explicitly.

    ~Tensor();

    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);

    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // ── Metadata ──────────────────────────────────────────────────────────────

    int64_t ndim()  const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const;

    const std::vector<int64_t>& shape()   const { return shape_;   }
    const std::vector<int64_t>& strides() const { return strides_; }
    Device                      device()  const { return device_;  }

    // ── Raw data access ───────────────────────────────────────────────────────
    // data()       — host pointer; valid only when device() == CPU.
    // cuda_data()  — device pointer; valid only when device() == CUDA.

    float*       data()            { return data_.data(); }
    const float* data()      const { return data_.data(); }

    float*       cuda_data()       { return cuda_data_; }
    const float* cuda_data() const { return cuda_data_; }

    // ── Element access (CPU only) ─────────────────────────────────────────────
    // t.at({i, j, k})  — checks rank and per-dimension bounds.
    // Throws if called on a CUDA tensor.

    float& at(const std::vector<int64_t>& indices);
    float  at(const std::vector<int64_t>& indices) const;

    // ── Device transfer ───────────────────────────────────────────────────────
    // Returns a new tensor on the requested device.
    // Accepts "cpu" or "cuda".  No-op (copy) if already on that device.

    Tensor to(const std::string& device) const;

    // ── Utilities ─────────────────────────────────────────────────────────────

    // "Tensor(shape=[2, 3], device=cpu, data=[0, 1, 2, ...])"
    // CUDA tensors print "<on cuda>" instead of values.
    std::string to_string() const;

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;  // row-major, in elements
    std::vector<float>   data_;     // host buffer  (non-empty iff device_ == CPU)
    float*               cuda_data_ = nullptr;  // device buffer (non-null iff device_ == CUDA)
    Device               device_;

    void    compute_strides();
    int64_t flat_index(const std::vector<int64_t>& indices) const;
};

}  // namespace vf
