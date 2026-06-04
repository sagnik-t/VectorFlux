"""
vectorflux — A TensorFlow 1-style deep learning framework built from scratch.

This file is intentionally thin: it only re-exports symbols from the
internal modules that make up the package.  Add new features there, not here.

Module layout
-------------
_core.so        — C++/CUDA extension (Tensor, Node, eager ops, graph builders)
_device.py      — set_default_device / get_default_device
_variables.py   — Variable class, variable registry, reset_default_graph
_ops.py         — Overloaded symbolic/eager ops (add, matmul, relu, …), gradients
_session.py     — Session, TrainOp
_layers.py      — Dense, vf.nn, vf.layers
_losses.py      — vf.losses (mse, softmax_cross_entropy)
_optimizers.py  — GradientDescentOptimizer, AdamOptimizer, vf.train
"""

# ── C++ extension: core types and low-level builders ─────────────────────────
from ._core import (
    hello_cuda,
    Tensor,
    Node,
    # Eager-only ops (no symbolic overload needed)
    step,
    transpose,
    # Low-level graph builders (backward compat / power users)
    make_const,
    make_add, make_mul, make_sub, make_relu, make_matmul,
    make_sigmoid, make_tanh, make_softmax,
    make_step, make_transpose, make_oneslike,
    make_placeholder, make_variable,
    make_reduce_sum, make_reduce_mean,
    make_softmax_cross_entropy_with_logits,
    variable_assign,
    variable_get_value,
    global_variables_initializer,
)

# ── Device management ─────────────────────────────────────────────────────────
from ._device import set_default_device, get_default_device

# ── Variables ─────────────────────────────────────────────────────────────────
from ._variables import (
    Variable,
    reset_default_graph,
    _get_all_variables,
    _register_variable,
)

# ── Ops and gradients ─────────────────────────────────────────────────────────
from ._ops import (
    add, mul, sub,
    relu, sigmoid, tanh, softmax, matmul,
    reduce_sum, reduce_mean,
    gradients,
)

# ── Session / TrainOp ─────────────────────────────────────────────────────────
from ._session import Session, TrainOp

# ── Layers ────────────────────────────────────────────────────────────────────
from ._layers import Dense, nn, layers

# ── Losses ────────────────────────────────────────────────────────────────────
from ._losses import losses

# ── Optimizers ────────────────────────────────────────────────────────────────
from ._optimizers import GradientDescentOptimizer, AdamOptimizer, train


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Core types
    "Tensor", "Node", "Variable",
    # Graph-building ops
    "placeholder",
    "add", "mul", "sub", "relu", "sigmoid", "tanh", "softmax", "matmul",
    "reduce_sum", "reduce_mean",
    # Submodules
    "nn", "layers", "losses", "train",
    # High-level layer
    "Dense",
    # Training helpers
    "gradients",
    "variable_assign",
    "global_variables_initializer",
    "TrainOp",
    "GradientDescentOptimizer",
    "AdamOptimizer",
    # Session
    "Session",
    # Device management
    "set_default_device",
    "get_default_device",
    # Graph lifecycle
    "reset_default_graph",
    # Low-level builders (backward compat)
    "make_const",
    "make_add", "make_mul", "make_sub", "make_relu", "make_matmul",
    "make_sigmoid", "make_tanh", "make_softmax",
    "make_reduce_sum", "make_reduce_mean",
    "make_softmax_cross_entropy_with_logits",
    "make_step", "make_transpose", "make_oneslike",
    "make_placeholder", "make_variable",
    # Eager-only ops
    "step", "transpose",
    # Misc
    "hello_cuda",
]


def placeholder(shape, name=""):
    """Declare a Placeholder node.  Value must be supplied via feed_dict."""
    from ._core import make_placeholder
    return make_placeholder(shape, name)
