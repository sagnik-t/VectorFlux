#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include "tensor.h"
#include "ops.h"
#include "graph.h"
#include "session.h"
#include "autograd.h"

namespace py = pybind11;

std::string hello_cuda();

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

    m.def("hello_cuda", &hello_cuda);

    py::class_<vf::Tensor>(m, "Tensor")
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

        .def("at", [](const vf::Tensor& t, std::vector<int64_t> idx) {
            return t.at(idx);
        }, py::arg("indices"))

        .def("set", [](vf::Tensor& t, std::vector<int64_t> idx, float val) {
            t.at(idx) = val;
        }, py::arg("indices"), py::arg("value"))

        .def("to_numpy", [](const vf::Tensor& t) {
            if (t.device() != vf::Device::CPU)
                throw std::runtime_error(
                    "to_numpy() requires a CPU tensor; call .to('cpu') first");
            std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());
            py::array_t<float> arr(shape);
            std::memcpy(arr.request().ptr, t.data(),
                        static_cast<size_t>(t.numel()) * sizeof(float));
            return arr;
        })

        .def("to", [](const vf::Tensor& t, const std::string& device) {
            return t.to(device);
        }, py::arg("device"))

        .def("__repr__", &vf::Tensor::to_string);

    // ── Eager tensor ops ──────────────────────────────────────────────────────

    m.def("add",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::add(a, b); },
        py::arg("a"), py::arg("b"));
    m.def("mul",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::mul(a, b); },
        py::arg("a"), py::arg("b"));
    m.def("sub",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::sub(a, b); },
        py::arg("a"), py::arg("b"));
    m.def("relu",
        [](const vf::Tensor& a) { return vf::relu(a); },
        py::arg("a"));
    m.def("sigmoid",
        [](const vf::Tensor& a) { return vf::sigmoid(a); },
        py::arg("a"));
    m.def("tanh",
        [](const vf::Tensor& a) { return vf::tanh(a); },
        py::arg("a"));
    m.def("softmax",
        [](const vf::Tensor& a) { return vf::softmax(a); },
        py::arg("a"));
    m.def("matmul",
        [](const vf::Tensor& a, const vf::Tensor& b) { return vf::matmul(a, b); },
        py::arg("a"), py::arg("b"));
    m.def("step",
        [](const vf::Tensor& a) { return vf::step(a); },
        py::arg("a"));
    m.def("transpose",
        [](const vf::Tensor& a) { return vf::transpose(a); },
        py::arg("a"));

    // ── T14: eager reduction ops ──────────────────────────────────────────────

    m.def("reduce_sum",
        [](const vf::Tensor& a) { return vf::reduce_sum(a); },
        py::arg("a"));
    m.def("reduce_mean",
        [](const vf::Tensor& a) { return vf::reduce_mean(a); },
        py::arg("a"));

    // ── Node ──────────────────────────────────────────────────────────────────

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

    // ── Graph builders ────────────────────────────────────────────────────────

    m.def("make_const",
        [](const vf::Tensor& t, const std::string& name) {
            return vf::default_graph().make_const(t, name);
        }, py::arg("value"), py::arg("name") = "");
    m.def("make_add",
        [](vf::NodeRef a, vf::NodeRef b, const std::string& name) {
            return vf::default_graph().make_add(a, b, name);
        }, py::arg("a"), py::arg("b"), py::arg("name") = "");
    m.def("make_mul",
        [](vf::NodeRef a, vf::NodeRef b, const std::string& name) {
            return vf::default_graph().make_mul(a, b, name);
        }, py::arg("a"), py::arg("b"), py::arg("name") = "");
    m.def("make_sub",
        [](vf::NodeRef a, vf::NodeRef b, const std::string& name) {
            return vf::default_graph().make_sub(a, b, name);
        }, py::arg("a"), py::arg("b"), py::arg("name") = "");
    m.def("make_relu",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_relu(a, name);
        }, py::arg("a"), py::arg("name") = "");
    m.def("make_sigmoid",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_sigmoid(a, name);
        }, py::arg("a"), py::arg("name") = "");
    m.def("make_tanh",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_tanh(a, name);
        }, py::arg("a"), py::arg("name") = "");
    m.def("make_softmax",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_softmax(a, name);
        }, py::arg("a"), py::arg("name") = "");
    m.def("make_matmul",
        [](vf::NodeRef a, vf::NodeRef b, const std::string& name) {
            return vf::default_graph().make_matmul(a, b, name);
        }, py::arg("a"), py::arg("b"), py::arg("name") = "");
    m.def("make_step",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_step(a, name);
        }, py::arg("a"), py::arg("name") = "");
    m.def("make_transpose",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_transpose(a, name);
        }, py::arg("a"), py::arg("name") = "");
    m.def("make_oneslike",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_oneslike(a, name);
        }, py::arg("a"), py::arg("name") = "");

    // ── T11: Placeholder, Variable, global_variables_initializer ─────────────

    m.def("make_placeholder",
        [](std::vector<int64_t> shape, const std::string& name) {
            return vf::default_graph().make_placeholder(std::move(shape), name);
        }, py::arg("shape"), py::arg("name") = "");

    m.def("make_variable",
        [](const vf::Tensor& initial_value, const std::string& name) {
            return vf::default_graph().make_variable(initial_value, name);
        }, py::arg("initial_value"), py::arg("name") = "");

    m.def("variable_assign",
        [](vf::NodeRef variable, const vf::Tensor& value) {
            auto* var_op = dynamic_cast<vf::VariableOp*>(variable->op().get());
            if (!var_op)
                throw std::runtime_error(
                    "variable_assign: node '" + variable->name() +
                    "' is not a Variable");
            var_op->assign(value);
        }, py::arg("variable"), py::arg("value"));

    // ── T14: read current variable value (for optimizer updates) ─────────────
    m.def("variable_get_value",
        [](vf::NodeRef variable) -> vf::Tensor {
            auto* var_op = dynamic_cast<vf::VariableOp*>(variable->op().get());
            if (!var_op)
                throw std::runtime_error(
                    "variable_get_value: node '" + variable->name() +
                    "' is not a Variable");
            return var_op->value();
        }, py::arg("variable"));

    m.def("global_variables_initializer",
        [](const std::string& name) {
            return vf::default_graph().make_init_variables(name);
        }, py::arg("name") = "");

    m.def("reset_default_graph", &vf::reset_default_graph);

    m.def("gradients",
        [](vf::NodeRef ys, std::vector<vf::NodeRef> xs) {
            return vf::gradients(ys, xs);
        },
        py::arg("ys"), py::arg("xs"));

    // ── T14: graph builders for reduction and loss ops ────────────────────────

    m.def("make_reduce_sum",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_reduce_sum(a, name);
        }, py::arg("a"), py::arg("name") = "");

    m.def("make_reduce_mean",
        [](vf::NodeRef a, const std::string& name) {
            return vf::default_graph().make_reduce_mean(a, name);
        }, py::arg("a"), py::arg("name") = "");

    m.def("make_softmax_cross_entropy_with_logits",
        [](vf::NodeRef logits, vf::NodeRef labels, const std::string& name) {
            return vf::default_graph().make_softmax_cross_entropy_with_logits(
                logits, labels, name);
        }, py::arg("logits"), py::arg("labels"), py::arg("name") = "");

    py::class_<vf::Session>(m, "Session")
        .def(py::init<>())
        .def("run",
            [](vf::Session& sess, vf::NodeRef fetch,
               py::dict feed_dict_py) -> vf::Tensor {
                return sess.run(fetch, feed_dict_from_py(feed_dict_py));
            },
            py::arg("fetch"), py::arg("feed_dict") = py::dict())
        .def("run",
            [](vf::Session& sess, std::vector<vf::NodeRef> fetches,
               py::dict feed_dict_py) -> std::vector<vf::Tensor> {
                return sess.run(fetches, feed_dict_from_py(feed_dict_py));
            },
            py::arg("fetches"), py::arg("feed_dict") = py::dict());
}
