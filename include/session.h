#pragma once

#include "graph.h"
#include "tensor.h"

#include <unordered_map>
#include <vector>

namespace vf {

// ── Session ───────────────────────────────────────────────────────────────────
//
// TF1-style session: build the graph first, then call run() to execute it.
// run() performs a topological traversal of the subgraph reachable from the
// requested fetch nodes and returns their output tensors.
//
// feed_dict lets callers inject tensor values for any node, bypassing that
// node's Op.  This is the injection point for Placeholder values (T11); in
// T09 any node can be overridden.
//
// Usage:
//   vf::Session sess;
//   vf::Tensor out  = sess.run(output_node);
//   vf::Tensor out2 = sess.run(output_node, {{ph_node, feed_value}});
//   std::vector<vf::Tensor> outs = sess.run({n1, n2});

class Session {
public:
    // Default: operates on vf::default_graph().
    Session();

    // Operate on an explicit graph.
    explicit Session(Graph& graph);

    // ── run — single fetch ────────────────────────────────────────────────────
    Tensor run(
        const NodeRef&                               fetch,
        const std::unordered_map<NodeRef, Tensor>&  feed_dict = {});

    // ── run — multiple fetches ────────────────────────────────────────────────
    // Returns one Tensor per entry in fetches, in the same order.
    std::vector<Tensor> run(
        const std::vector<NodeRef>&                  fetches,
        const std::unordered_map<NodeRef, Tensor>&  feed_dict = {});

private:
    Graph& graph_;

    // DFS post-order over the subgraph reachable from fetches.
    // Result is in evaluation order (inputs before their consumers).
    static std::vector<NodeRef> topo_sort(const std::vector<NodeRef>& fetches);
};

}  // namespace vf
