#include "graph.h"
#include "ops.h"

#include <stdexcept>

namespace vf {

// ── Forward pass implementations ──────────────────────────────────────────────

Tensor AddOp::forward(const std::vector<Tensor>& in) const {
    return vf::add(in[0], in[1]);
}

Tensor MulOp::forward(const std::vector<Tensor>& in) const {
    return vf::mul(in[0], in[1]);
}

Tensor SubOp::forward(const std::vector<Tensor>& in) const {
    return vf::sub(in[0], in[1]);
}

Tensor ReluOp::forward(const std::vector<Tensor>& in) const {
    return vf::relu(in[0]);
}

Tensor SigmoidOp::forward(const std::vector<Tensor>& in) const {
    return vf::sigmoid(in[0]);
}

Tensor TanhOp::forward(const std::vector<Tensor>& in) const {
    return vf::tanh(in[0]);
}

Tensor SoftmaxOp::forward(const std::vector<Tensor>& in) const {
    return vf::softmax(in[0]);
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
    return Tensor(src.shape(), std::move(ones_data), src.device());
}

Tensor InitVariablesOp::forward(const std::vector<Tensor>&) const {
    for (const auto& var_node : variables_) {
        auto* var_op = dynamic_cast<VariableOp*>(var_node->op().get());
        if (var_op) var_op->initialize();
    }
    return Tensor({1}, {0.0f});
}

// ── Backward pass ─────────────────────────────────────────────────────────────

// add: dL/da = dL/dout,  dL/db = dL/dout
std::vector<NodeRef> AddOp::gradient(const NodeRef& /*node*/,
                                      const NodeRef& grad_in) const {
    return {grad_in, grad_in};
}

// mul: dL/da = dL/dout * b,  dL/db = dL/dout * a
std::vector<NodeRef> MulOp::gradient(const NodeRef& node,
                                      const NodeRef& grad_in) const {
    const NodeRef& a = node->inputs()[0];
    const NodeRef& b = node->inputs()[1];
    return {
        default_graph().make_mul(grad_in, b),
        default_graph().make_mul(grad_in, a)
    };
}

// relu: dL/dx = dL/dout * step(x)
std::vector<NodeRef> ReluOp::gradient(const NodeRef& node,
                                       const NodeRef& grad_in) const {
    const NodeRef& a = node->inputs()[0];
    return {default_graph().make_mul(grad_in, default_graph().make_step(a))};
}

// sigmoid: dL/dx = dL/dout * sigma(x) * (1 - sigma(x))
//   `node` IS the sigmoid node whose evaluated output == sigma(x).
//   ones = oneslike(node),  one_minus_s = sub(ones, node)
//   deriv = mul(node, one_minus_s)  → sigma(x) * (1 - sigma(x))
std::vector<NodeRef> SigmoidOp::gradient(const NodeRef& node,
                                          const NodeRef& grad_in) const {
    auto& g = default_graph();
    auto ones      = g.make_oneslike(node);
    auto one_minus = g.make_sub(ones, node);         // 1 - sigma(x)
    auto deriv     = g.make_mul(node, one_minus);    // sigma(x) * (1 - sigma(x))
    return { g.make_mul(grad_in, deriv) };
}

// tanh: dL/dx = dL/dout * (1 - tanh(x)^2)
//   t_sq = mul(node, node)  → tanh(x)^2
//   ones = oneslike(node)
//   deriv = sub(ones, t_sq) → 1 - tanh(x)^2
std::vector<NodeRef> TanhOp::gradient(const NodeRef& node,
                                       const NodeRef& grad_in) const {
    auto& g = default_graph();
    auto t_sq  = g.make_mul(node, node);             // tanh(x)^2
    auto ones  = g.make_oneslike(node);
    auto deriv = g.make_sub(ones, t_sq);             // 1 - tanh(x)^2
    return { g.make_mul(grad_in, deriv) };
}

// matmul: A=[M,K], B=[K,N]
//   dL/dA = dL/dout @ B^T
//   dL/dB = A^T @ dL/dout
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

NodeRef Graph::make_sub(NodeRef a, NodeRef b, std::string name) {
    return register_node(std::make_shared<SubOp>(), {a, b}, "Sub", name);
}

NodeRef Graph::make_relu(NodeRef a, std::string name) {
    return register_node(std::make_shared<ReluOp>(), {a}, "Relu", name);
}

NodeRef Graph::make_sigmoid(NodeRef a, std::string name) {
    return register_node(std::make_shared<SigmoidOp>(), {a}, "Sigmoid", name);
}

NodeRef Graph::make_tanh(NodeRef a, std::string name) {
    return register_node(std::make_shared<TanhOp>(), {a}, "Tanh", name);
}

NodeRef Graph::make_softmax(NodeRef a, std::string name) {
    return register_node(std::make_shared<SoftmaxOp>(), {a}, "Softmax", name);
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
    return register_node(
        std::make_shared<InitVariablesOp>(variables_),
        {},
        "InitVariables",
        name.empty() ? "init_variables" : name);
}

void Graph::reset_values() {
    for (auto& n : nodes_) n->reset();
}

void Graph::clear() {
    nodes_.clear();
    variables_.clear();
    next_id_ = 0;
}

Graph& default_graph() {
    static Graph g;
    return g;
}

void reset_default_graph() {
    default_graph().clear();
}

}  // namespace vf
