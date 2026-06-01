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
// Each Op subclass encapsulates one computation.  Session.run() (T09) calls
// forward() after resolving all input values.

class Op {
public:
    virtual ~Op() = default;

    // Human-readable name used in node names and repr, e.g. "Add", "MatMul".
    virtual std::string type_name() const = 0;

    // Compute the output tensor from evaluated input tensors.
    // inputs[i] is the output of this node's i-th input node.
    virtual Tensor forward(const std::vector<Tensor>& inputs) const = 0;
};

// ── Concrete ops ──────────────────────────────────────────────────────────────

// Leaf node that holds a fixed tensor value.
// Used in tests and as a building block for Variable (T11).
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
};

class MulOp : public Op {
public:
    std::string type_name() const override { return "Mul"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

class ReluOp : public Op {
public:
    std::string type_name() const override { return "Relu"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

class MatMulOp : public Op {
public:
    std::string type_name() const override { return "MatMul"; }
    Tensor forward(const std::vector<Tensor>& in) const override;
};

// ── Node ──────────────────────────────────────────────────────────────────────
//
// One node in the computation DAG.  Structural fields (op, inputs, name) are
// set at construction and never change.  The output field is populated by
// Session.run() in T09 and cleared between runs.

class Node {
public:
    Node(std::shared_ptr<Op>  op,
         std::vector<NodeRef> inputs,
         std::string          name);

    // ── Structural (immutable after construction) ──────────────────────────────
    const std::string&           name()   const { return name_;   }
    std::string                  type()   const { return op_->type_name(); }
    const std::vector<NodeRef>&  inputs() const { return inputs_; }
    const std::shared_ptr<Op>&   op()     const { return op_;     }

    // ── Evaluated value (managed by Session in T09) ────────────────────────────
    void          set_output(Tensor t);   // stores result, marks node evaluated
    const Tensor& output()        const;  // throws if not yet evaluated
    bool          evaluated()     const { return output_.has_value(); }
    void          reset();                // clears output (between runs)

private:
    std::string           name_;
    std::shared_ptr<Op>   op_;
    std::vector<NodeRef>  inputs_;
    std::optional<Tensor> output_;
};

// ── Graph ─────────────────────────────────────────────────────────────────────
//
// Owns all nodes.  Factory methods create a node, register it, and return a
// shared_ptr handle (NodeRef).  The Session (T09) iterates over nodes() to
// execute a topological traversal.

class Graph {
public:
    // Factory methods — each creates a new node and registers it.
    // If name is empty an auto-name like "Add_3" is generated.
    NodeRef make_const  (Tensor value,            std::string name = "");
    NodeRef make_add    (NodeRef a, NodeRef b,    std::string name = "");
    NodeRef make_mul    (NodeRef a, NodeRef b,    std::string name = "");
    NodeRef make_relu   (NodeRef a,               std::string name = "");
    NodeRef make_matmul (NodeRef a, NodeRef b,    std::string name = "");

    // Reset all nodes' evaluated flags (called between Session.run() calls).
    void reset_values();

    // Remove all nodes from this graph (useful in tests).
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

// ── Default graph (TF1-style global) ─────────────────────────────────────────

Graph& default_graph();
void   reset_default_graph();   // clears default_graph(); use between tests

}  // namespace vf
