#include "graph.h"
#include "ops.h"

#include <stdexcept>

namespace vf {

// ── Concrete op forward() implementations ────────────────────────────────────
// Delegate directly to the existing eager dispatch layer in ops_cpu.cpp.

Tensor AddOp::forward(const std::vector<Tensor>& in) const {
    return vf::add(in[0], in[1]);
}

Tensor MulOp::forward(const std::vector<Tensor>& in) const {
    return vf::mul(in[0], in[1]);
}

Tensor ReluOp::forward(const std::vector<Tensor>& in) const {
    return vf::relu(in[0]);
}

Tensor MatMulOp::forward(const std::vector<Tensor>& in) const {
    return vf::matmul(in[0], in[1]);
}

// ── Node ──────────────────────────────────────────────────────────────────────

Node::Node(std::shared_ptr<Op>  op,
           std::vector<NodeRef> inputs,
           std::string          name)
    : op_(std::move(op)),
      inputs_(std::move(inputs)),
      name_(std::move(name)) {}

void Node::set_output(Tensor t) {
    output_ = std::move(t);
}

const Tensor& Node::output() const {
    if (!output_.has_value()) {
        throw std::runtime_error(
            "Node '" + name_ + "' has not been evaluated; call Session.run() first");
    }
    return *output_;
}

void Node::reset() {
    output_.reset();
}

// ── Graph ─────────────────────────────────────────────────────────────────────

NodeRef Graph::register_node(std::shared_ptr<Op>  op,
                              std::vector<NodeRef> inputs,
                              const std::string&   type,
                              const std::string&   user_name) {
    std::string name = user_name.empty()
        ? type + "_" + std::to_string(next_id_++)
        : user_name;
    auto node = std::make_shared<Node>(std::move(op), std::move(inputs), name);
    nodes_.push_back(node);
    return node;
}

NodeRef Graph::make_const(Tensor value, std::string name) {
    return register_node(
        std::make_shared<ConstOp>(std::move(value)), {}, "Const", name);
}

NodeRef Graph::make_add(NodeRef a, NodeRef b, std::string name) {
    return register_node(std::make_shared<AddOp>(), {a, b}, "Add", name);
}

NodeRef Graph::make_mul(NodeRef a, NodeRef b, std::string name) {
    return register_node(std::make_shared<MulOp>(), {a, b}, "Mul", name);
}

NodeRef Graph::make_relu(NodeRef a, std::string name) {
    return register_node(std::make_shared<ReluOp>(), {a}, "Relu", name);
}

NodeRef Graph::make_matmul(NodeRef a, NodeRef b, std::string name) {
    return register_node(std::make_shared<MatMulOp>(), {a, b}, "MatMul", name);
}

void Graph::reset_values() {
    for (auto& n : nodes_) n->reset();
}

void Graph::clear() {
    nodes_.clear();
    next_id_ = 0;
}

// ── Default graph ─────────────────────────────────────────────────────────────

Graph& default_graph() {
    static Graph g;
    return g;
}

void reset_default_graph() {
    default_graph().clear();
}

}  // namespace vf
