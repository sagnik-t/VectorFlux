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
    NodeRef make_const    (Tensor value,            std::string name = "");
    NodeRef make_add      (NodeRef a, NodeRef b,    std::string name = "");
    NodeRef make_mul      (NodeRef a, NodeRef b,    std::string name = "");
    NodeRef make_relu     (NodeRef a,               std::string name = "");
    NodeRef make_matmul   (NodeRef a, NodeRef b,    std::string name = "");

    // ── Gradient-support ops ──────────────────────────────────────────────────
    NodeRef make_step     (NodeRef a,               std::string name = "");
    NodeRef make_transpose(NodeRef a,               std::string name = "");
    NodeRef make_oneslike (NodeRef a,               std::string name = "");

    void reset_values();
    void clear();

    const std::vector<NodeRef>& nodes() const { return nodes_; }
    std::size_t                 size()  const { return nodes_.size(); }

private:
    std::vector<NodeRef> nodes_;
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
