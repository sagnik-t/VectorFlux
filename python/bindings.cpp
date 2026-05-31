#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // py::array_t
#include <pybind11/stl.h>     // std::vector ↔ Python list auto-conversion

#include <cstring>
#include "tensor.h"
#include "ops.h"

namespace py = pybind11;

std::string hello_cuda();   // src/hello.cu

PYBIND11_MODULE(_core, m) {
    m.doc() = "VectorFlux core extension";

    // ── T01 ───────────────────────────────────────────────────────────────────
    m.def("hello_cuda", &hello_cuda, "Run a hello-world CUDA kernel");

    // ── T02/T03/T05: Tensor ───────────────────────────────────────────────────
    py::class_<vf::Tensor>(m, "Tensor")

        // ── Constructors ──────────────────────────────────────────────────────

        .def(py::init([](std::vector<int64_t> shape) {
            return vf::Tensor(std::move(shape));
        }), py::arg("shape"))

        .def(py::init([](py::array_t<float,
                          py::array::c_style | py::array::forcecast> arr) {
            auto buf = arr.request();
            std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());
            const float* ptr = static_cast<const float*>(buf.ptr);
            return vf::Tensor(shape, std::vector<float>(ptr, ptr + buf.size));
        }), py::arg("array"))

        // ── Metadata ──────────────────────────────────────────────────────────

        .def_property_readonly("shape", [](const vf::Tensor& t) {
            return py::tuple(py::cast(t.shape()));
        })
        .def_property_readonly("strides", [](const vf::Tensor& t) {
            return py::tuple(py::cast(t.strides()));
        })
        .def_property_readonly("ndim",   &vf::Tensor::ndim)
        .def_property_readonly("device", [](const vf::Tensor& t) -> std::string {
            return t.device() == vf::Device::CPU ? "cpu" : "cuda";
        })
        .def("numel", &vf::Tensor::numel)

        // ── Element access ────────────────────────────────────────────────────

        .def("at", [](const vf::Tensor& t, std::vector<int64_t> idx) {
            return t.at(idx);
        }, py::arg("indices"))

        .def("set", [](vf::Tensor& t, std::vector<int64_t> idx, float val) {
            t.at(idx) = val;
        }, py::arg("indices"), py::arg("value"))

        // ── NumPy interop ─────────────────────────────────────────────────────

        .def("to_numpy", [](const vf::Tensor& t) {
            if (t.device() != vf::Device::CPU) {
                throw std::runtime_error(
                    "to_numpy() requires a CPU tensor; call .to('cpu') first");
            }
            std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());
            py::array_t<float> arr(shape);
            std::memcpy(arr.request().ptr, t.data(),
                        static_cast<size_t>(t.numel()) * sizeof(float));
            return arr;
        }, "Return a numpy copy of this tensor's data (CPU tensors only)")

        // ── T05: Device transfer ──────────────────────────────────────────────

        .def("to", [](const vf::Tensor& t, const std::string& device) {
            return t.to(device);
        }, py::arg("device"),
           "Move tensor to device.  device='cpu' or 'cuda'.")

        // ── Repr ──────────────────────────────────────────────────────────────

        .def("__repr__", &vf::Tensor::to_string);

    // ── T04: Basic CPU ops ────────────────────────────────────────────────────

    m.def("add",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::add(a, b); },
        py::arg("a"), py::arg("b"));

    m.def("mul",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::mul(a, b); },
        py::arg("a"), py::arg("b"));

    m.def("relu",
        [](const vf::Tensor& a) { return vf::relu(a); },
        py::arg("a"));

    m.def("matmul",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::matmul(a, b); },
        py::arg("a"), py::arg("b"));
}
