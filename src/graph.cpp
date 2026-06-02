#include "graph.h"
#include "ops.h"

#include <stdexcept>

namespace vf {

// ── Forward pass implementations ──────────────────────────────────────────────
// Delegate to the eager dispatch layer in ops_cpu.cpp / ops_cuda.cu.

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

Tensor StepOp::forward(const std::vector<Tensor>& in) const {
    return vf::step(in[0]);
}

Tensor TransposeOp::forward(const std::vector<Tensor>& in) const {
    return vf::transpose(in[0]);
}

Tensor OnesLikeOp::forward(const std::vector<Tensor>& in) const {
    const auto& src = in[0];
    const int64_t n = src.numel();
    std::vector<float> ones_data(static_cast<size_t>(n), 1.0f);
    // Tensor(shape, data, device) uploads to GPU if device == CUDA.
    return Tensor(src.shape(), std::move(ones_data), src.device());
}

// ── T11: InitVariablesOp forward ─────────────────────────────────────────────
//
// Iterates the variable NodeRefs captured at construction time and calls
// initialize() on each VariableOp.  Returns a dummy scalar so Session.run()
// has a valid Tensor to hand back to the caller.

Tensor InitVariablesOp::forward(const std::vector<Tensor>&) const {
    for (const auto& var_node : variables_) {
        auto* var_op = dynamic_cast<VariableOp*>(var_node->op().get());
        if (var_op) var_op->initialize();
    }
    return Tensor({1}, {0.0f});   // dummy; callers typically ignore this
}

// ── Backward pass (gradient graph construction) ───────────────────────────────
//
// Each gradient() call wires new nodes into the default graph.
// Session.run() then evaluates them together with the forward nodes.

// add: out = a + b
//   dL/da = dL/dout * 1  = grad_in
//   dL/db = dL/dout * 1  = grad_in
std::vector<NodeRef> AddOp::gradient(const NodeRef& /*node*/,
                                      const NodeRef& grad_in) const {
    return {grad_in, grad_in};
}

// mul: out = a * b
//   dL/da = dL/dout * b
//   dL/db = dL/dout * a
std::vector<NodeRef> MulOp::gradient(const NodeRef& node,
                                      const NodeRef& grad_in) const {
    const NodeRef& a = node->inputs()[0];
    const NodeRef& b = node->inputs()[1];
    return {
        default_graph().make_mul(grad_in, b),
        default_graph().make_mul(grad_in, a)
    };
}

// relu: out = max(0, a)
//   dL/da = dL/dout * step(a)   (step = Heaviside, 1 if x>0 else 0)
std::vector<NodeRef> ReluOp::gradient(const NodeRef& node,
                                       const NodeRef& grad_in) const {
    const NodeRef& a = node->inputs()[0];
    return {default_graph().make_mul(grad_in, default_graph().make_step(a))};
}

// matmul: out = A @ B,  A=[M,K]  B=[K,N]  out=[M,N]
//   dL/dA = dL/dout @ B^T   →  [M,N] @ [N,K] = [M,K]
//   dL/dB = A^T @ dL/dout   →  [K,M] @ [M,N] = [K,N]
std::vector<NodeRef> MatMulOp::gradient(const NodeRef& node,
                                         const NodeRef& grad_in) const {
    const NodeRef& a = node->inputs()[0];
    const NodeRef& b = node->inputs()[1];
    return {
        default_graph().make_matmul(grad_in,
                                    default_graph().make_transpose(b)),
        default_graph().make_matmul(default_graph().make_transpose(a),
                                    grad_in)
    };
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

NodeRef Graph::make_step(NodeRef a, std::string name) {
    return register_node(std::make_shared<StepOp>(), {a}, "Step", name);
}

NodeRef Graph::make_transpose(NodeRef a, std::string name) {
    return register_node(std::make_shared<TransposeOp>(), {a}, "Transpose", name);
}

NodeRef Graph::make_oneslike(NodeRef a, std::string name) {
    return register_node(std::make_shared<OnesLikeOp>(), {a}, "OnesLike", name);
}

// ── T11: new graph factory methods ───────────────────────────────────────────

NodeRef Graph::make_placeholder(std::vector<int64_t> shape, std::string name) {
    return register_node(
        std::make_shared<PlaceholderOp>(std::move(shape)), {}, "Placeholder", name);
}

NodeRef Graph::make_variable(Tensor initial_value, std::string name) {
    auto node = register_node(
        std::make_shared<VariableOp>(initial_value), {}, "Variable", name);
    variables_.push_back(node);
    return node;
}

NodeRef Graph::make_init_variables(std::string name) {
    // Snapshot the variable list at the time of this call.
    // Variables created afterwards are not included — consistent with TF1.
    return register_node(
        std::make_shared<InitVariablesOp>(variables_),
        {},
        "InitVariables",
        name.empty() ? "init_variables" : name);
}

// ── Graph lifecycle ───────────────────────────────────────────────────────────

void Graph::reset_values() {
    for (auto& n : nodes_) n->reset();
}

void Graph::clear() {
    nodes_.clear();
    variables_.clear();   // must be cleared alongside nodes_
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
