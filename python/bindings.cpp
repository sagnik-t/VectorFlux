#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // py::array_t
#include <pybind11/stl.h>     // std::vector ↔ Python list auto-conversion

#include <cstring>
#include "tensor.h"
#include "ops.h"
#include "graph.h"
#include "session.h"

namespace py = pybind11;

std::string hello_cuda();   // src/hello.cu

// Helper: convert a Python dict {Node: Tensor} → C++ unordered_map.
// Needed because pybind11's STL converters don't handle shared_ptr keys.
static std::unordered_map<vf::NodeRef, vf::Tensor>
feed_dict_from_py(const py::dict& d) {
    std::unordered_map<vf::NodeRef, vf::Tensor> fd;
    fd.reserve(d.size());
    for (auto& item : d) {
        fd.emplace(item.first.cast<vf::NodeRef>(), item.second.cast<vf::Tensor>());
    }
    return fd;
}

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

    // ── T08: Graph nodes ──────────────────────────────────────────────────────

    py::class_<vf::Node, vf::NodeRef>(m, "Node")
        .def_property_readonly("name",      &vf::Node::name)
        .def_property_readonly("type",      &vf::Node::type)
        .def_property_readonly("inputs",    &vf::Node::inputs)
        .def_property_readonly("evaluated", &vf::Node::evaluated)
        .def("__repr__", [](const vf::Node& n) {
            return "Node(type=" + n.type() +
                   ", name=" + n.name() +
                   ", inputs=" + std::to_string(n.inputs().size()) + ")";
        });

    m.def("make_const",
        [](const vf::Tensor& t, const std::string& name) {
            return vf::default_graph().make_const(t, name);
        },
        py::arg("value"), py::arg("name") = "",
        "Create a Const leaf node holding a fixed tensor.");

    m.def("make_add",
        [](vf::NodeRef a, vf::NodeRef b, const std::string& name) {
            return vf::default_graph().make_add(a, b, name);
        },
        py::arg("a"), py::arg("b"), py::arg("name") = "");

    m.def("make_mul",
        [](vf::NodeRef a, vf::NodeRef b, const std::string& name) {
            return vf::default_graph().make_mul(a, b, name);
        },
        py::arg("a"), py::arg("b"), py::arg("name") = "");

    m.def("make_relu",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_relu(a, name);
        },
        py::arg("a"), py::arg("name") = "");

    m.def("make_matmul",
        [](vf::NodeRef a, vf::NodeRef b, const std::string& name) {
            return vf::default_graph().make_matmul(a, b, name);
        },
        py::arg("a"), py::arg("b"), py::arg("name") = "");

    m.def("reset_default_graph",
        &vf::reset_default_graph,
        "Clear all nodes from the default graph (use between tests).");

    // ── T09: Session ──────────────────────────────────────────────────────────
    //
    // vf.Session() — executes a computation graph via topological traversal.
    //
    // sess.run(node)                      → Tensor
    // sess.run([node1, node2])            → list[Tensor]
    // sess.run(node,   feed_dict={n: t})  → Tensor
    // sess.run([...],  feed_dict={n: t})  → list[Tensor]
    //
    // feed_dict maps any Node to a Tensor, overriding that node's Op for this
    // run.  The dict is not retained between calls.

    py::class_<vf::Session>(m, "Session")
        .def(py::init<>(), "Create a Session operating on the default graph.")

        // Single fetch → Tensor
        .def("run",
            [](vf::Session& sess,
               vf::NodeRef   fetch,
               py::dict      feed_dict_py) -> vf::Tensor {
                return sess.run(fetch, feed_dict_from_py(feed_dict_py));
            },
            py::arg("fetch"),
            py::arg("feed_dict") = py::dict(),
            "Execute the graph and return the output of `fetch`.")

        // List of fetches → list[Tensor]
        .def("run",
            [](vf::Session&                sess,
               std::vector<vf::NodeRef>    fetches,
               py::dict                    feed_dict_py) -> std::vector<vf::Tensor> {
                return sess.run(fetches, feed_dict_from_py(feed_dict_py));
            },
            py::arg("fetches"),
            py::arg("feed_dict") = py::dict(),
            "Execute the graph and return the outputs of each node in `fetches`.");
}
