#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // py::array_t
#include <pybind11/stl.h>     // std::vector ↔ Python list auto-conversion

#include <cstring>
#include "tensor.h"
#include "ops.h"              // T04: add / mul / relu / matmul

namespace py = pybind11;

// ── Forward declarations from other translation units ─────────────────────────
std::string hello_cuda();   // src/hello.cu

// ── Module ────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_core, m) {
    m.doc() = "VectorFlux core extension";

    // ── T01: hello-world CUDA kernel ──────────────────────────────────────────
    m.def("hello_cuda", &hello_cuda, "Run a hello-world CUDA kernel");

    // ── T02/T03: Tensor ───────────────────────────────────────────────────────
    py::class_<vf::Tensor>(m, "Tensor")

        // ── Constructors ──────────────────────────────────────────────────────

        // vf.Tensor([2, 3])  →  zero-initialised CPU tensor
        .def(py::init([](std::vector<int64_t> shape) {
            return vf::Tensor(std::move(shape));
        }), py::arg("shape"))

        // vf.Tensor(np_array)  →  copies data in; forcecast handles non-float32
        // c_style ensures C-contiguous layout even if the array is a view/transpose
        .def(py::init([](py::array_t<float,
                          py::array::c_style | py::array::forcecast> arr) {
            auto buf = arr.request();
            std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());
            const float* ptr = static_cast<const float*>(buf.ptr);
            return vf::Tensor(shape, std::vector<float>(ptr, ptr + buf.size));
        }), py::arg("array"))

        // ── Metadata ──────────────────────────────────────────────────────────

        // Returns a tuple, matching numpy convention
        .def_property_readonly("shape", [](const vf::Tensor& t) {
            return py::tuple(py::cast(t.shape()));
        })
        .def_property_readonly("strides", [](const vf::Tensor& t) {
            // Strides in elements (not bytes), matching our C++ convention
            return py::tuple(py::cast(t.strides()));
        })
        .def_property_readonly("ndim",   &vf::Tensor::ndim)
        .def_property_readonly("device", [](const vf::Tensor& t) -> std::string {
            return t.device() == vf::Device::CPU ? "cpu" : "cuda";
        })
        .def("numel", &vf::Tensor::numel, "Total number of elements")

        // ── Element access ────────────────────────────────────────────────────

        // t.at([i, j, k])  — read
        .def("at", [](const vf::Tensor& t, std::vector<int64_t> idx) {
            return t.at(idx);
        }, py::arg("indices"), "Read element at multi-index, e.g. t.at([0, 2])")

        // t.set([i, j, k], value)  — write
        // (Python floats don't have reference semantics, so write is a separate method)
        .def("set", [](vf::Tensor& t, std::vector<int64_t> idx, float val) {
            t.at(idx) = val;
        }, py::arg("indices"), py::arg("value"),
           "Write element at multi-index, e.g. t.set([0, 2], 3.14)")

        // ── NumPy interop ─────────────────────────────────────────────────────

        // Returns a fresh C-contiguous float32 numpy array (a copy)
        .def("to_numpy", [](const vf::Tensor& t) {
            std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());
            py::array_t<float> arr(shape);
            std::memcpy(arr.request().ptr, t.data(),
                        static_cast<size_t>(t.numel()) * sizeof(float));
            return arr;
        }, "Return a numpy copy of this tensor's data")

        // ── Repr ──────────────────────────────────────────────────────────────

        .def("__repr__", &vf::Tensor::to_string);

    // ── T04: Basic CPU ops ────────────────────────────────────────────────────
    //
    // All four ops auto-dispatch on tensor.device().
    // CUDA paths are wired up in T06 (element-wise) and T07 (matmul).

    m.def("add",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::add(a, b); },
        py::arg("a"), py::arg("b"),
        "Element-wise addition.  a and b must have identical shapes.");

    m.def("mul",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::mul(a, b); },
        py::arg("a"), py::arg("b"),
        "Element-wise multiplication.  a and b must have identical shapes.");

    m.def("relu",
        [](const vf::Tensor& a) { return vf::relu(a); },
        py::arg("a"),
        "Element-wise ReLU: max(x, 0).");

    m.def("matmul",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::matmul(a, b); },
        py::arg("a"), py::arg("b"),
        "2-D matrix multiply: [M,K] @ [K,N] -> [M,N].");
}
