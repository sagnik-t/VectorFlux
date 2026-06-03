#pragma once

#include "tensor.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vf {

class Node;
using NodeRef = std::shared_ptr<Node>;

// ── Op base class ─────────────────────────────────────────────────────────────

class Op {
public:
    virtual ~Op() = default;
    virtual std::string type_name() const = 0;
    virtual Tensor forward(const std::vector<Tensor>& inputs) const = 0;
    virtual std::vector<NodeRef> gradient(
            const NodeRef& /*node*/,
            const NodeRef& /*grad_in*/) const {
        return {};
    }
};

// ── Concrete ops ──────────────────────────────────────────────────────────────

class ConstOp : public Op {
public:
    explicit ConstOp(Tensor value) : value_(std::move(value)) {}
    std::string type_name() const override { return "Const"; }
    Tensor forward(const std::vector<Tensor>&) const override { return value_; }
private:
    Tensor value_;
};

class AddOp : public Op {
public:
    std::string type_name() const override { return "Add"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

class MulOp : public Op {
public:
    std::string type_name() const override { return "Mul"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── T13: SubOp ───────────────────────────────────────────────────────────────
// Element-wise subtraction: out = a - b.
// gradient: dL/da = g,  dL/db = -g  (via NegOp)
class SubOp : public Op {
public:
    std::string type_name() const override { return "Sub"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

class ReluOp : public Op {
public:
    std::string type_name() const override { return "Relu"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── T13: SigmoidOp ───────────────────────────────────────────────────────────
// forward:  sigma(x) = 1 / (1 + exp(-x))
// gradient: dL/dx = dL/dout * sigma(x) * (1 - sigma(x))
//           Built as: mul(grad_in, mul(node, sub(oneslike(node), node)))
class SigmoidOp : public Op {
public:
    std::string type_name() const override { return "Sigmoid"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── T13: TanhOp ──────────────────────────────────────────────────────────────
// forward:  tanh(x)
// gradient: dL/dx = dL/dout * (1 - tanh(x)^2)
//           Built as: mul(grad_in, sub(oneslike(node), mul(node, node)))
class TanhOp : public Op {
public:
    std::string type_name() const override { return "Tanh"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── T13: SoftmaxOp ───────────────────────────────────────────────────────────
// forward: exp(x) / sum(exp(x)) along axis 0 (numerically stable).
// gradient: fused into SoftmaxCrossEntropyWithLogitsOp (T14).
class SoftmaxOp : public Op {
public:
    std::string type_name() const override { return "Softmax"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

class MatMulOp : public Op {
public:
    std::string type_name() const override { return "MatMul"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── Gradient-support ops ──────────────────────────────────────────────────────

class StepOp : public Op {
public:
    std::string type_name() const override { return "Step"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

class TransposeOp : public Op {
public:
    std::string type_name() const override { return "Transpose"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

class OnesLikeOp : public Op {
public:
    std::string type_name() const override { return "OnesLike"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// ── T11: Placeholder and Variable ops ────────────────────────────────────────

class PlaceholderOp : public Op {
public:
    explicit PlaceholderOp(std::vector<int64_t> shape)
        : shape_(std::move(shape)) {}
    std::string type_name() const override { return "Placeholder"; }
    Tensor forward(const std::vector<Tensor>&) const override {
        throw std::runtime_error(
            "Placeholder has no value — provide one via feed_dict");
    }
    const std::vector<int64_t>& placeholder_shape() const { return shape_; }
private:
    std::vector<int64_t> shape_;
};

class VariableOp : public Op {
public:
    explicit VariableOp(Tensor initial_value)
        : initial_value_(initial_value),
          current_value_(std::move(initial_value)) {}

    std::string type_name() const override { return "Variable"; }
    Tensor forward(const std::vector<Tensor>&) const override {
        return current_value_;
    }
    void assign(Tensor new_value) { current_value_ = std::move(new_value); }
    void initialize() { current_value_ = initial_value_; }
    const Tensor& value() const { return current_value_; }

private:
    Tensor initial_value_;
    Tensor current_value_;
};

class InitVariablesOp : public Op {
public:
    explicit InitVariablesOp(std::vector<NodeRef> variables)
        : variables_(std::move(variables)) {}
    std::string type_name() const override { return "InitVariables"; }
    Tensor forward(const std::vector<Tensor>&) const override;
private:
    std::vector<NodeRef> variables_;
};

// ── T14: ReduceSumOp ─────────────────────────────────────────────────────────
// forward:  scalar = sum of all elements in input
// gradient: dL/dinput = fill(input.shape, g_scalar)
//           Implemented by ReduceSumGradOp.
class ReduceSumOp : public Op {
public:
    std::string type_name() const override { return "ReduceSum"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── T14: ReduceSumGradOp (terminal) ──────────────────────────────────────────
// inputs: [original_input, g_scalar]
// forward: tensor with shape(original_input) filled with g_scalar[0]
class ReduceSumGradOp : public Op {
public:
    std::string type_name() const override { return "ReduceSumGrad"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// ── T14: ReduceMeanOp ────────────────────────────────────────────────────────
// forward:  scalar = mean of all elements in input
// gradient: dL/dinput = fill(input.shape, g_scalar / N)
//           Implemented by ReduceMeanGradOp.
class ReduceMeanOp : public Op {
public:
    std::string type_name() const override { return "ReduceMean"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── T14: ReduceMeanGradOp (terminal) ─────────────────────────────────────────
// inputs: [original_input, g_scalar]
// forward: tensor with shape(original_input) filled with g_scalar[0] / N
class ReduceMeanGradOp : public Op {
public:
    std::string type_name() const override { return "ReduceMeanGrad"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// ── T14: SoftmaxCrossEntropyWithLogitsOp ─────────────────────────────────────
// forward:  scalar = mean_N(-sum_C y_c * log softmax(l_c))
//           Uses log-sum-exp trick for numerical stability.
//           logits: [C, N]  labels: [C, N]  (one-hot per column)
// gradient: dL/dlogits = SoftmaxCEGradOp(logits, labels, g_scalar)
//           = (softmax(logits) - labels) * g_scalar / N
//           No gradient through labels.
class SoftmaxCrossEntropyWithLogitsOp : public Op {
public:
    std::string type_name() const override {
        return "SoftmaxCrossEntropyWithLogits";
    }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── T14: SoftmaxCEGradOp (terminal) ──────────────────────────────────────────
// inputs: [logits, labels, g_scalar]
// forward: (softmax(logits) - labels) * g_scalar[0] / N
class SoftmaxCEGradOp : public Op {
public:
    std::string type_name() const override { return "SoftmaxCEGrad"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// ── NegOp (gradient-support) ──────────────────────────────────────────────────
// Negate all elements: out = -in[0].
// Used in SubOp::gradient for dL/db = -grad_in.
// Terminal — no higher-order gradient needed.
class NegOp : public Op {
public:
    std::string type_name() const override { return "Neg"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// ── Node ──────────────────────────────────────────────────────────────────────

class Node {
public:
    Node(std::shared_ptr<Op>  op,
         std::vector<NodeRef> inputs,
         std::string          name);

    const std::string&           name()   const { return name_;   }
    std::string                  type()   const { return op_->type_name(); }
    const std::vector<NodeRef>&  inputs() const { return inputs_; }
    const std::shared_ptr<Op>&   op()     const { return op_;     }

    void          set_output(Tensor t);
    const Tensor& output()        const;
    bool          evaluated()     const { return output_.has_value(); }
    void          reset();

private:
    std::string           name_;
    std::shared_ptr<Op>   op_;
    std::vector<NodeRef>  inputs_;
    std::optional<Tensor> output_;
};

// ── Graph ─────────────────────────────────────────────────────────────────────

class Graph {
public:
    // ── User-facing ops ───────────────────────────────────────────────────────
    NodeRef make_const    (Tensor value,                        std::string name = "");
    NodeRef make_add      (NodeRef a, NodeRef b,                std::string name = "");
    NodeRef make_mul      (NodeRef a, NodeRef b,                std::string name = "");
    NodeRef make_sub      (NodeRef a, NodeRef b,                std::string name = "");  // T13
    NodeRef make_relu     (NodeRef a,                           std::string name = "");
    NodeRef make_sigmoid  (NodeRef a,                           std::string name = "");  // T13
    NodeRef make_tanh     (NodeRef a,                           std::string name = "");  // T13
    NodeRef make_softmax  (NodeRef a,                           std::string name = "");  // T13
    NodeRef make_matmul   (NodeRef a, NodeRef b,                std::string name = "");

    // ── T11: trainable graph nodes ────────────────────────────────────────────
    NodeRef make_placeholder(std::vector<int64_t> shape,        std::string name = "");
    NodeRef make_variable   (Tensor initial_value,              std::string name = "");
    NodeRef make_init_variables(                                std::string name = "");

    // ── Gradient-support ops ──────────────────────────────────────────────────
    NodeRef make_step     (NodeRef a,               std::string name = "");
    NodeRef make_transpose(NodeRef a,               std::string name = "");
    NodeRef make_oneslike (NodeRef a,               std::string name = "");

    // ── T14: reduction and loss ops ───────────────────────────────────────────
    NodeRef make_reduce_sum (NodeRef a,                           std::string name = "");
    NodeRef make_reduce_mean(NodeRef a,                           std::string name = "");
    NodeRef make_softmax_cross_entropy_with_logits(NodeRef logits, NodeRef labels,
                                                    std::string name = "");

    // ── T14: gradient support ops (used internally by gradient() methods) ─────
    NodeRef make_reduce_sum_grad (NodeRef input, NodeRef g_scalar, std::string name = "");
    NodeRef make_reduce_mean_grad(NodeRef input, NodeRef g_scalar, std::string name = "");
    NodeRef make_softmax_ce_grad (NodeRef logits, NodeRef labels, NodeRef g_scalar,
                                   std::string name = "");
    NodeRef make_neg(NodeRef a, std::string name = "");   // NegOp: -a

    void reset_values();
    void clear();

    const std::vector<NodeRef>& nodes()     const { return nodes_;     }
    const std::vector<NodeRef>& variables() const { return variables_; }
    std::size_t                 size()      const { return nodes_.size(); }

private:
    std::vector<NodeRef> nodes_;
    std::vector<NodeRef> variables_;
    int                  next_id_ = 0;

    NodeRef register_node(std::shared_ptr<Op>  op,
                          std::vector<NodeRef> inputs,
                          const std::string&   type,
                          const std::string&   user_name);
};

// ── Default graph ─────────────────────────────────────────────────────────────

Graph& default_graph();
void   reset_default_graph();

}  // namespace vf
