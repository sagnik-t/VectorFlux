#include "session.h"

#include <functional>
#include <stdexcept>
#include <unordered_set>

namespace vf {

// ── Constructor ───────────────────────────────────────────────────────────────

Session::Session() : graph_(default_graph()) {}
Session::Session(Graph& graph) : graph_(graph) {}

// ── Topological sort ──────────────────────────────────────────────────────────
//
// Standard DFS post-order over the subgraph reachable from `fetches`.
// A node's raw pointer is used as the visited key so that two NodeRefs
// pointing at the same Node are treated as a single visit.

std::vector<NodeRef> Session::topo_sort(const std::vector<NodeRef>& fetches) {
    std::vector<NodeRef>     order;
    std::unordered_set<Node*> visited;

    std::function<void(const NodeRef&)> visit = [&](const NodeRef& node) {
        if (visited.count(node.get())) return;
        visited.insert(node.get());

        // Recurse into inputs first (ensures inputs precede their consumers).
        for (const auto& inp : node->inputs()) {
            visit(inp);
        }
        order.push_back(node);
    };

    for (const auto& fetch : fetches) visit(fetch);
    return order;   // already in evaluation order
}

// ── run — multiple fetches ────────────────────────────────────────────────────

std::vector<Tensor> Session::run(
    const std::vector<NodeRef>&                 fetches,
    const std::unordered_map<NodeRef, Tensor>&  feed_dict)
{
    if (fetches.empty()) return {};

    // 1. Build topo-ordered list of nodes we actually need.
    auto order = topo_sort(fetches);

    // 2. Reset the evaluated flag on every node we'll touch.
    //    (Nodes outside this subgraph are untouched.)
    for (auto& n : order) n->reset();

    // 3. Evaluate in topo order.
    for (auto& node : order) {
        // feed_dict override: inject the supplied tensor directly.
        auto it = feed_dict.find(node);
        if (it != feed_dict.end()) {
            node->set_output(it->second);
            continue;
        }

        // Gather already-evaluated input tensors.
        std::vector<Tensor> input_tensors;
        input_tensors.reserve(node->inputs().size());
        for (const auto& inp : node->inputs()) {
            // inp->output() throws if not evaluated — a bug, not user error.
            input_tensors.push_back(inp->output());
        }

        // Execute the Op and store the result.
        node->set_output(node->op()->forward(input_tensors));
    }

    // 4. Collect results in fetch order.
    std::vector<Tensor> results;
    results.reserve(fetches.size());
    for (const auto& fetch : fetches) {
        results.push_back(fetch->output());
    }
    return results;
}

// ── run — single fetch convenience ────────────────────────────────────────────

Tensor Session::run(
    const NodeRef&                              fetch,
    const std::unordered_map<NodeRef, Tensor>& feed_dict)
{
    return run(std::vector<NodeRef>{fetch}, feed_dict)[0];
}

}  // namespace vf
