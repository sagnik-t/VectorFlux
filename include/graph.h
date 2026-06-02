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
//
// Each Op subclass encapsulates one computation.  Session.run() calls
// forward() after resolving all input values.  gradients() calls gradient()
// when building the backward graph.

class Op {
public:
    virtual ~Op() = default;

    // Human-readable name used in node names and repr, e.g. "Add", "MatMul".
    virtual std::string type_name() const = 0;

    // Compute the output tensor from evaluated input tensors.
    virtual Tensor forward(const std::vector<Tensor>& inputs) const = 0;

    // Build backward graph nodes.
    // node     — the forward node (access node->inputs() for upstream NodeRefs)
    // grad_in  — NodeRef carrying  dL / d(node.output)
    // Returns one NodeRef per input of `node`, carrying dL / d(input_i).
    // Default: no gradients (leaf/non-differentiable ops).
    virtual std::vector<NodeRef> gradient(
            const NodeRef& /*node*/,
            const NodeRef& /*grad_in*/) const {
        return {};
    }
};

// ── Concrete ops ──────────────────────────────────────────────────────────────

// Leaf node that holds a fixed tensor value.
class ConstOp : public Op {
public:
    explicit ConstOp(Tensor value) : value_(std::move(value)) {}
    std::string type_name() const override { return "Const"; }
    Tensor forward(const std::vector<Tensor>&) const override { return value_; }
    // No gradient: ConstOp has no inputs to differentiate through.
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

class ReluOp : public Op {
public:
    std::string type_name() const override { return "Relu"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

class MatMulOp : public Op {
public:
    std::string type_name() const override { return "MatMul"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
    std::vector<NodeRef> gradient(const NodeRef& node,
                                   const NodeRef& grad_in) const override;
};

// ── Gradient-support ops ──────────────────────────────────────────────────────
//
// These are created by the backward pass; users rarely build them directly.

// Element-wise Heaviside step:  out[i] = (in[i] > 0) ? 1 : 0
// Used by ReluOp::gradient.
class StepOp : public Op {
public:
    std::string type_name() const override { return "Step"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// 2-D matrix transpose:  [M, N] → [N, M]
// Used by MatMulOp::gradient.
class TransposeOp : public Op {
public:
    std::string type_name() const override { return "Transpose"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// Tensor of ones with the same shape (and device) as the input.
// Used by gradients() to seed dL/dL = 1.
class OnesLikeOp : public Op {
public:
    std::string type_name() const override { return "OnesLike"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// ── T11: Placeholder and Variable ops ────────────────────────────────────────

// Placeholder: shape is declared at build time; value is injected at run time
// via feed_dict.  Calling forward() without a feed raises a clear error.
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

// Variable: trainable parameter with mutable state.
//   forward()    — returns current_value_ (called by Session.run())
//   assign(t)    — replaces current_value_ (called by optimizers)
//   initialize() — resets current_value_ to initial_value_ (called by InitVariablesOp)
class VariableOp : public Op {
public:
    explicit VariableOp(Tensor initial_value)
        : initial_value_(initial_value),
          current_value_(std::move(initial_value)) {}

    std::string type_name() const override { return "Variable"; }

    // Returns the current value; does NOT modify any state (const-correct).
    Tensor forward(const std::vector<Tensor>&) const override {
        return current_value_;
    }

    // Update the stored parameter (e.g. after an optimizer step).
    void assign(Tensor new_value) { current_value_ = std::move(new_value); }

    // Reset to the value supplied at construction.
    void initialize() { current_value_ = initial_value_; }

    const Tensor& value() const { return current_value_; }

private:
    Tensor initial_value_;   // frozen copy for re-initialisation
    Tensor current_value_;   // mutable working copy
};

// InitVariablesOp: when executed by Session.run(), calls initialize() on every
// Variable captured at construction time.  Returns a dummy scalar so it can
// be passed to sess.run() like any other node.
//
// global_variables_initializer() creates one of these with a snapshot of the
// graph's variable list at call time — variables added afterwards are not
// included (consistent with TF1 behaviour).
class InitVariablesOp : public Op {
public:
    explicit InitVariablesOp(std::vector<NodeRef> variables)
        : variables_(std::move(variables)) {}
    std::string type_name() const override { return "InitVariables"; }
    Tensor forward(const std::vector<Tensor>&) const override;
private:
    std::vector<NodeRef> variables_;
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
    NodeRef make_relu     (NodeRef a,                           std::string name = "");
    NodeRef make_matmul   (NodeRef a, NodeRef b,                std::string name = "");

    // ── T11: trainable graph nodes ────────────────────────────────────────────
    // make_placeholder: declare an input slot; fill it at run time via feed_dict.
    NodeRef make_placeholder(std::vector<int64_t> shape,        std::string name = "");
    // make_variable: create a trainable parameter initialised to `initial_value`.
    // The node is recorded in variables_ for global_variables_initializer().
    NodeRef make_variable   (Tensor initial_value,              std::string name = "");
    // make_init_variables: snapshot the current variable list and return a node
    // that, when run, resets all of them to their initial values.
    NodeRef make_init_variables(                                std::string name = "");

    // ── Gradient-support ops ──────────────────────────────────────────────────
    NodeRef make_step     (NodeRef a,               std::string name = "");
    NodeRef make_transpose(NodeRef a,               std::string name = "");
    NodeRef make_oneslike (NodeRef a,               std::string name = "");

    void reset_values();
    void clear();   // also clears the variables_ list

    const std::vector<NodeRef>& nodes()     const { return nodes_;     }
    const std::vector<NodeRef>& variables() const { return variables_; }
    std::size_t                 size()      const { return nodes_.size(); }

private:
    std::vector<NodeRef> nodes_;
    std::vector<NodeRef> variables_;   // Variable nodes in insertion order
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
