from ._core import (
    hello_cuda,
    Tensor,
    # Eager tensor ops — renamed so we can shadow them with overloaded versions.
    add      as _eager_add,
    mul      as _eager_mul,
    relu     as _eager_relu,
    matmul   as _eager_matmul,
    step,
    transpose,
    # Graph node type
    Node,
    # Low-level graph builders — kept for backward compat and internal use.
    make_const,
    make_add, make_mul, make_relu, make_matmul,
    make_step, make_transpose, make_oneslike,
    make_placeholder, make_variable,
    variable_assign,
    global_variables_initializer,
    reset_default_graph,
    gradients as _core_gradients,
    Session   as _CppSession,
)

import numpy as np


# ── placeholder ───────────────────────────────────────────────────────────────

def placeholder(shape, name=""):
    """Declare a Placeholder node.  Value must be supplied via feed_dict."""
    return make_placeholder(shape, name)


# ── Variable ──────────────────────────────────────────────────────────────────

class Variable:
    """
    A trainable parameter.  Wraps a Variable graph node and exposes:
      - .assign(value)    — update the stored value (accepts Tensor or ndarray)
      - .name / .type / .inputs / .evaluated — mirrors the Node interface

    Can be passed anywhere a NodeRef is expected in the graph API.

    Usage:
        W = vf.Variable(np.random.randn(784, 128).astype(np.float32))
        W.assign(new_weights)
        y = vf.matmul(W, x)
    """

    def __init__(self, initial_value, name=""):
        if not isinstance(initial_value, Tensor):
            initial_value = Tensor(np.asarray(initial_value, dtype=np.float32))
        self._node = make_variable(initial_value, name)

    # ── Node interface passthrough ─────────────────────────────────────────────
    @property
    def name(self):      return self._node.name
    @property
    def type(self):      return self._node.type
    @property
    def inputs(self):    return self._node.inputs
    @property
    def evaluated(self): return self._node.evaluated

    def __repr__(self):
        return f"Variable(name={self.name})"

    # ── Assignment ─────────────────────────────────────────────────────────────
    def assign(self, value):
        """Update the variable's current value (accepts Tensor or numpy array)."""
        if not isinstance(value, Tensor):
            value = Tensor(np.asarray(value, dtype=np.float32))
        variable_assign(self._node, value)


# ── Internal helpers (defined after Variable so isinstance checks work) ────────

def _unwrap(x):
    """Return the underlying NodeRef for a Variable, or x unchanged."""
    if isinstance(x, Variable):
        return x._node
    return x


def _is_node(x):
    """True if x is a graph node (NodeRef or Variable)."""
    return isinstance(x, (Node, Variable))


# ── Overloaded graph / eager ops ──────────────────────────────────────────────
#
# Each function dispatches on argument type:
#   - Any argument is a Node or Variable → wire a new node into the graph.
#   - All arguments are Tensors           → run eagerly (existing C++ path).

def add(a, b, name=""):
    """Element-wise add.  Symbolic when given nodes, eager when given Tensors."""
    if _is_node(a) or _is_node(b):
        return make_add(_unwrap(a), _unwrap(b), name)
    return _eager_add(a, b)


def mul(a, b, name=""):
    """Element-wise multiply.  Symbolic when given nodes, eager when given Tensors."""
    if _is_node(a) or _is_node(b):
        return make_mul(_unwrap(a), _unwrap(b), name)
    return _eager_mul(a, b)


def relu(a, name=""):
    """ReLU activation.  Symbolic when given a node, eager when given a Tensor."""
    if _is_node(a):
        return make_relu(_unwrap(a), name)
    return _eager_relu(a)


def matmul(a, b, name=""):
    """Matrix multiply.  Symbolic when given nodes, eager when given Tensors."""
    if _is_node(a) or _is_node(b):
        return make_matmul(_unwrap(a), _unwrap(b), name)
    return _eager_matmul(a, b)


# ── gradients ─────────────────────────────────────────────────────────────────

def gradients(ys, xs):
    """Compute d(sum(ys)) / d(xs[i]).  xs may contain Node or Variable objects."""
    return _core_gradients(_unwrap(ys), [_unwrap(x) for x in xs])


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """
    TF1-style session: build the graph first, then call .run() to execute it.

    Accepts Variable objects wherever a NodeRef is expected — both as fetch
    targets and as feed_dict keys.

    Usage:
        sess = vf.Session()
        result  = sess.run(output_node)
        results = sess.run([node_a, node_b])
        result  = sess.run(node, feed_dict={placeholder: value})
        result  = sess.run(node, feed_dict={variable: override_value})
    """

    def __init__(self):
        self._sess = _CppSession()

    def run(self, fetches, feed_dict=None):
        # Unwrap Variable keys in feed_dict so the C++ layer sees NodeRefs.
        cpp_fd = {}
        for k, v in (feed_dict or {}).items():
            cpp_fd[k._node if isinstance(k, Variable) else k] = v

        if isinstance(fetches, (list, tuple)):
            cpp_fetches = [f._node if isinstance(f, Variable) else f
                           for f in fetches]
            return self._sess.run(cpp_fetches, cpp_fd)
        else:
            cpp_fetch = fetches._node if isinstance(fetches, Variable) else fetches
            return self._sess.run(cpp_fetch, cpp_fd)


# ── vf.nn submodule ───────────────────────────────────────────────────────────

class _nn:
    """Neural-network ops as graph nodes.

    Usage: vf.nn.relu(node)
    (T13 will add sigmoid, softmax, tanh here.)
    """
    @staticmethod
    def relu(a, name=""):
        return relu(a, name)


nn = _nn()


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Core types
    "Tensor", "Node", "Variable",
    # Graph-building ops (TF1-style names, work symbolically or eagerly)
    "placeholder",
    "add", "mul", "relu", "matmul",
    "nn",
    # Training
    "gradients",
    "variable_assign",
    "global_variables_initializer",
    # Session
    "Session",
    # Graph lifecycle
    "reset_default_graph",
    # Low-level builders (kept for backward compat with existing tests)
    "make_const",
    "make_add", "make_mul", "make_relu", "make_matmul",
    "make_step", "make_transpose", "make_oneslike",
    "make_placeholder", "make_variable",
    # Eager-only ops (no graph equivalent needed)
    "step", "transpose",
    # Misc
    "hello_cuda",
]
