#include "autograd.h"
#include "graph.h"

#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vf {

// ── Local topo sort ───────────────────────────────────────────────────────────
//
// DFS post-order from roots, keyed on raw Node* to deduplicate shared nodes.
// Identical to the one in session.cpp, kept local to avoid coupling.

static std::vector<NodeRef> topo_sort(const std::vector<NodeRef>& roots) {
    std::vector<NodeRef>      order;
    std::unordered_set<Node*> visited;

    std::function<void(const NodeRef&)> visit = [&](const NodeRef& node) {
        if (visited.count(node.get())) return;
        visited.insert(node.get());
        for (const auto& inp : node->inputs()) visit(inp);
        order.push_back(node);
    };

    for (const auto& r : roots) visit(r);
    return order;
}

// ── gradients ─────────────────────────────────────────────────────────────────

std::vector<NodeRef> gradients(const NodeRef&               ys,
                                const std::vector<NodeRef>&  xs)
{
    // 1. Topo-sorted list of all nodes reachable from ys (forward order).
    auto order = topo_sort({ys});

    // 2. grad_map: raw Node* → accumulated gradient NodeRef.
    //    Using raw pointer as key avoids the need for a shared_ptr hasher.
    std::unordered_map<Node*, NodeRef> grad_map;

    // 3. Seed: dL/dL = ones_like(L).
    //    For scalar loss this is the scalar 1; for tensor loss it is equivalent
    //    to computing d(sum(ys))/dx, which is the conventional choice.
    grad_map[ys.get()] = default_graph().make_oneslike(ys);

    // 4. Walk in reverse topo order, distributing gradient to each input.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const NodeRef& node = *it;

        auto git = grad_map.find(node.get());
        if (git == grad_map.end()) continue;   // node not connected to loss

        const NodeRef& g_out = git->second;

        // Ask the op for per-input gradient nodes.
        auto input_grads = node->op()->gradient(node, g_out);

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            if (i >= input_grads.size() || !input_grads[i]) continue;

            const NodeRef& inp = node->inputs()[i];
            NodeRef&       g   = input_grads[i];

            auto existing = grad_map.find(inp.get());
            if (existing != grad_map.end()) {
                // A node reached from multiple consumers: accumulate.
                grad_map[inp.get()] =
                    default_graph().make_add(existing->second, g);
            } else {
                grad_map[inp.get()] = g;
            }
        }
    }

    // 5. Collect requested gradients in xs order.
    std::vector<NodeRef> result;
    result.reserve(xs.size());
    for (const auto& x : xs) {
        auto it = grad_map.find(x.get());
        if (it == grad_map.end()) {
            throw std::runtime_error(
                "gradients(): node '" + x->name() +
                "' is not reachable from ys; gradient is structurally zero");
        }
        result.push_back(it->second);
    }
    return result;
}

}  // namespace vf
