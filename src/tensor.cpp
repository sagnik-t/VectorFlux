#include "tensor.h"

#include <cuda_runtime.h>   // cudaMalloc / cudaFree / cudaMemcpy / cudaMemset

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vf {

// ── CUDA error helper ─────────────────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(err));
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

void Tensor::compute_strides() {
    const size_t n = shape_.size();
    strides_.resize(n);
    if (n == 0) return;

    strides_[n - 1] = 1;
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

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
    const int64_t n = numel();

    if (device_ == Device::CPU) {
        data_.assign(n, 0.0f);
    } else {
        cuda_check(cudaMalloc(&cuda_data_, n * sizeof(float)), "cudaMalloc");
        cuda_check(cudaMemset(cuda_data_, 0, n * sizeof(float)), "cudaMemset");
    }
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

    if (device_ == Device::CPU) {
        data_ = std::move(data);
    } else {
        cuda_check(cudaMalloc(&cuda_data_, expected * sizeof(float)), "cudaMalloc");
        cuda_check(cudaMemcpy(cuda_data_, data.data(),
                              expected * sizeof(float),
                              cudaMemcpyHostToDevice), "cudaMemcpy H→D");
    }
}

// ── Rule-of-five ──────────────────────────────────────────────────────────────

Tensor::~Tensor() {
    if (cuda_data_) cudaFree(cuda_data_);
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      strides_(other.strides_),
      device_(other.device_),
      data_(other.data_) {
    if (other.cuda_data_) {
        const int64_t n = numel();
        cuda_check(cudaMalloc(&cuda_data_, n * sizeof(float)), "cudaMalloc (copy ctor)");
        cuda_check(cudaMemcpy(cuda_data_, other.cuda_data_,
                              n * sizeof(float),
                              cudaMemcpyDeviceToDevice), "cudaMemcpy D→D (copy ctor)");
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    // Free any existing CUDA memory before overwriting.
    if (cuda_data_) { cudaFree(cuda_data_); cuda_data_ = nullptr; }

    shape_   = other.shape_;
    strides_ = other.strides_;
    device_  = other.device_;
    data_    = other.data_;

    if (other.cuda_data_) {
        const int64_t n = numel();
        cuda_check(cudaMalloc(&cuda_data_, n * sizeof(float)), "cudaMalloc (copy assign)");
        cuda_check(cudaMemcpy(cuda_data_, other.cuda_data_,
                              n * sizeof(float),
                              cudaMemcpyDeviceToDevice), "cudaMemcpy D→D (copy assign)");
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      device_(other.device_),
      data_(std::move(other.data_)),
      cuda_data_(other.cuda_data_) {
    other.cuda_data_ = nullptr;   // ownership transferred; source must not free
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    if (cuda_data_) { cudaFree(cuda_data_); cuda_data_ = nullptr; }

    shape_     = std::move(other.shape_);
    strides_   = std::move(other.strides_);
    device_    = other.device_;
    data_      = std::move(other.data_);
    cuda_data_ = other.cuda_data_;

    other.cuda_data_ = nullptr;
    return *this;
}

// ── numel ─────────────────────────────────────────────────────────────────────

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int64_t s : shape_) n *= s;
    return n;
}

// ── Element access (CPU only) ─────────────────────────────────────────────────

float& Tensor::at(const std::vector<int64_t>& indices) {
    if (device_ != Device::CPU)
        throw std::runtime_error("Tensor::at() is only valid on CPU tensors");
    return data_[flat_index(indices)];
}

float Tensor::at(const std::vector<int64_t>& indices) const {
    if (device_ != Device::CPU)
        throw std::runtime_error("Tensor::at() is only valid on CPU tensors");
    return data_[flat_index(indices)];
}

// ── Device transfer ───────────────────────────────────────────────────────────

Tensor Tensor::to(const std::string& device_str) const {
    if (device_str != "cpu" && device_str != "cuda") {
        throw std::invalid_argument(
            "Unknown device '" + device_str + "': expected 'cpu' or 'cuda'");
    }

    const Device target = (device_str == "cuda") ? Device::CUDA : Device::CPU;

    // Already on the requested device — return a copy.
    if (target == device_) return *this;

    const int64_t n = numel();

    if (target == Device::CUDA) {
        // CPU → CUDA
        Tensor out(shape_, Device::CUDA);   // allocates cuda_data_, zero-init
        cuda_check(cudaMemcpy(out.cuda_data_, data_.data(),
                              n * sizeof(float),
                              cudaMemcpyHostToDevice), "cudaMemcpy H→D (.to)");
        return out;
    } else {
        // CUDA → CPU
        Tensor out(shape_, Device::CPU);    // allocates data_, zero-init
        cuda_check(cudaMemcpy(out.data_.data(), cuda_data_,
                              n * sizeof(float),
                              cudaMemcpyDeviceToHost), "cudaMemcpy D→H (.to)");
        return out;
    }
}

// ── to_string ─────────────────────────────────────────────────────────────────

std::string Tensor::to_string() const {
    std::ostringstream os;

    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) os << ", ";
        os << shape_[i];
    }
    os << "]";

    os << ", device=" << (device_ == Device::CPU ? "cpu" : "cuda");

    os << ", data=";
    if (device_ == Device::CUDA) {
        os << "[<on cuda>]";
    } else {
        os << "[";
        const int64_t total = numel();
        const int64_t show  = std::min(total, static_cast<int64_t>(8));
        for (int64_t i = 0; i < show; ++i) {
            if (i) os << ", ";
            os << data_[static_cast<size_t>(i)];
        }
        if (total > show) os << ", ...";
        os << "]";
    }

    os << ")";
    return os.str();
}

}  // namespace vf
