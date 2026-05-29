#include "tensor.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vf {

// ── Private helpers ───────────────────────────────────────────────────────────

// Build row-major strides from shape_.
// e.g. shape [2,3,4] → strides [12,4,1]
// Scalar (shape []) → strides [] — numel() still returns 1.
void Tensor::compute_strides() {
    const size_t n = shape_.size();
    strides_.resize(n);
    if (n == 0) return;          // scalar: no strides needed

    strides_[n - 1] = 1;
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// Convert a multi-dimensional index to a flat buffer offset.
// Validates rank and per-dimension bounds; throws on mismatch / out-of-range.
int64_t Tensor::flat_index(const std::vector<int64_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument(
            "Index rank " + std::to_string(indices.size()) +
            " does not match tensor rank " + std::to_string(shape_.size()));
    }

    int64_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "Index " + std::to_string(indices[i]) +
                " is out of range for dim " + std::to_string(i) +
                " with size " + std::to_string(shape_[i]));
        }
        offset += indices[i] * strides_[i];
    }
    return offset;
}

// ── Constructors ──────────────────────────────────────────────────────────────

Tensor::Tensor(std::vector<int64_t> shape, Device device)
    : shape_(std::move(shape)), device_(device) {
    compute_strides();
    data_.assign(numel(), 0.0f);
}

Tensor::Tensor(std::vector<int64_t> shape,
               std::vector<float>   data,
               Device               device)
    : shape_(std::move(shape)), device_(device) {
    compute_strides();
    const int64_t expected = numel();
    if (static_cast<int64_t>(data.size()) != expected) {
        throw std::invalid_argument(
            "Data size " + std::to_string(data.size()) +
            " does not match tensor numel " + std::to_string(expected));
    }
    data_ = std::move(data);
}

// ── numel ─────────────────────────────────────────────────────────────────────

int64_t Tensor::numel() const {
    // Product of all dims.
    // Empty shape (scalar) → 1.  Any dim == 0 → 0.
    int64_t n = 1;
    for (int64_t s : shape_) n *= s;
    return n;
}

// ── Element access ────────────────────────────────────────────────────────────

float& Tensor::at(const std::vector<int64_t>& indices) {
    return data_[flat_index(indices)];
}

float Tensor::at(const std::vector<int64_t>& indices) const {
    return data_[flat_index(indices)];
}

// ── to_string ─────────────────────────────────────────────────────────────────

std::string Tensor::to_string() const {
    std::ostringstream os;

    // Shape
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) os << ", ";
        os << shape_[i];
    }
    os << "]";

    // Device
    os << ", device=" << (device_ == Device::CPU ? "cpu" : "cuda");

    // First few values (cap at 8 so scalars / tiny tensors print in full)
    os << ", data=[";
    const int64_t total = numel();
    const int64_t show  = std::min(total, static_cast<int64_t>(8));
    for (int64_t i = 0; i < show; ++i) {
        if (i) os << ", ";
        os << data_[static_cast<size_t>(i)];
    }
    if (total > show) os << ", ...";
    os << "])";

    return os.str();
}

}  // namespace vf
