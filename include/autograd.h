#pragma once

#include "graph.h"

#include <vector>

namespace vf {

// ── gradients ─────────────────────────────────────────────────────────────────
//
// Build the backward computation graph and return gradient NodeRefs.
//
// Computes  d(sum(ys)) / d(xs[i])  for each xs[i].
// The returned NodeRefs are new graph nodes; evaluate them with Session.run().
// They can be fetched together with the forward nodes in a single sess.run()
// call, so forward and backward execute in one shot.
//
// Throws std::runtime_error if any xs[i] is not reachable from ys.
//
// Usage:
//   auto grads      = vf::gradients(loss, {W, b});
//   auto [dW, db]   = vf::gradients(loss, {W, b});   // C++17 structured binding
//   auto [loss_val, dW_val] = sess.run({loss, dW});

std::vector<NodeRef> gradients(const NodeRef&               ys,
                                const std::vector<NodeRef>&  xs);

}  // namespace vf
