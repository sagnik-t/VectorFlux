from ._core import (
    hello_cuda,
    Tensor,
    # Eager tensor ops — renamed so we can shadow them with overloaded versions.
    add      as _eager_add,
    mul      as _eager_mul,
    sub      as _eager_sub,
    relu     as _eager_relu,
    sigmoid  as _eager_sigmoid,
    tanh     as _eager_tanh,
    softmax  as _eager_softmax,
    matmul   as _eager_matmul,
    step,
    transpose,
    # Graph node type
    Node,
    # Low-level graph builders — kept for backward compat and internal use.
    make_const,
    make_add, make_mul, make_sub, make_relu, make_matmul,
    make_sigmoid, make_tanh, make_softmax,
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

    def assign(self, value):
        """Update the variable's current value (accepts Tensor or numpy array)."""
        if not isinstance(value, Tensor):
            value = Tensor(np.asarray(value, dtype=np.float32))
        variable_assign(self._node, value)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unwrap(x):
    """Return the underlying NodeRef for a Variable, or x unchanged."""
    if isinstance(x, Variable):
        return x._node
    return x


def _is_node(x):
    """True if x is a graph node (NodeRef or Variable)."""
    return isinstance(x, (Node, Variable))


# ── Overloaded graph / eager ops ──────────────────────────────────────────────

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


def sub(a, b, name=""):
    """Element-wise subtract.  Symbolic when given nodes, eager when given Tensors."""
    if _is_node(a) or _is_node(b):
        return make_sub(_unwrap(a), _unwrap(b), name)
    return _eager_sub(a, b)


def relu(a, name=""):
    """ReLU.  Symbolic when given a node, eager when given a Tensor."""
    if _is_node(a):
        return make_relu(_unwrap(a), name)
    return _eager_relu(a)


def sigmoid(a, name=""):
    """Sigmoid activation.  Symbolic when given a node, eager when given a Tensor."""
    if _is_node(a):
        return make_sigmoid(_unwrap(a), name)
    return _eager_sigmoid(a)


def tanh(a, name=""):
    """Tanh activation.  Symbolic when given a node, eager when given a Tensor."""
    if _is_node(a):
        return make_tanh(_unwrap(a), name)
    return _eager_tanh(a)


def softmax(a, name=""):
    """Softmax along axis 0.  Symbolic when given a node, eager when given a Tensor."""
    if _is_node(a):
        return make_softmax(_unwrap(a), name)
    return _eager_softmax(a)


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
    TF1-style session.  Accepts Variable objects wherever a NodeRef is expected.

    Usage:
        sess = vf.Session()
        result  = sess.run(output_node)
        results = sess.run([node_a, node_b])
        result  = sess.run(node, feed_dict={placeholder: value})
    """

    def __init__(self):
        self._sess = _CppSession()

    def run(self, fetches, feed_dict=None):
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


# ── Dense layer ───────────────────────────────────────────────────────────────

class Dense:
    """
    Fully-connected layer:  out = W @ x  (optionally followed by an activation).

    Parameters
    ----------
    input_dim  : int   — number of input features
    units      : int   — number of output units
    activation : callable or None — e.g. vf.nn.relu, vf.sigmoid, vf.tanh
    name       : str   — prefix for the weight variable name

    Note: no bias term — broadcast-add is not yet implemented (planned for T16).

    Weight initialisation: Xavier uniform,  limit = sqrt(6 / (input_dim + units))

    Usage:
        layer = vf.Dense(784, 128, activation=vf.nn.relu)
        y     = layer(x)           # x is a NodeRef / placeholder
        init  = vf.global_variables_initializer()
        sess  = vf.Session()
        sess.run(init)
        result = sess.run(y, feed_dict={x: batch})
    """

    def __init__(self, input_dim, units, activation=None, name=""):
        self.input_dim  = input_dim
        self.units      = units
        self.activation = activation
        self.name       = name

        # Xavier uniform initialisation
        limit = np.sqrt(6.0 / (input_dim + units))
        nW    = np.random.uniform(-limit, limit,
                                   (units, input_dim)).astype(np.float32)
        w_name = (name + "/W") if name else ""
        self._W = Variable(Tensor(nW), name=w_name)

    @property
    def weights(self):
        """The weight Variable node."""
        return self._W

    def __call__(self, x):
        """Wire the layer into the graph and return the output node."""
        out = matmul(self._W, x)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def __repr__(self):
        act = self.activation.__name__ if callable(self.activation) else str(self.activation)
        return (f"Dense(input_dim={self.input_dim}, units={self.units}, "
                f"activation={act})")


# ── vf.nn submodule ───────────────────────────────────────────────────────────

class _nn:
    """Neural-network activation ops as graph nodes.

    Usage: vf.nn.relu(node), vf.nn.sigmoid(node), vf.nn.tanh(node),
           vf.nn.softmax(node)
    """
    @staticmethod
    def relu(a, name=""):    return relu(a, name)

    @staticmethod
    def sigmoid(a, name=""): return sigmoid(a, name)    # T13

    @staticmethod
    def tanh(a, name=""):    return tanh(a, name)       # T13

    @staticmethod
    def softmax(a, name=""): return softmax(a, name)    # T13


nn = _nn()


# ── vf.layers submodule ───────────────────────────────────────────────────────

class _layers:
    """Layer constructors.

    Usage: vf.layers.Dense(input_dim, units, activation=vf.nn.relu)
    """
    Dense = Dense


layers = _layers()


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Core types
    "Tensor", "Node", "Variable",
    # Graph-building ops
    "placeholder",
    "add", "mul", "sub", "relu", "sigmoid", "tanh", "softmax", "matmul",
    # Submodules
    "nn", "layers",
    # High-level layer
    "Dense",
    # Training
    "gradients",
    "variable_assign",
    "global_variables_initializer",
    # Session
    "Session",
    # Graph lifecycle
    "reset_default_graph",
    # Low-level builders (backward compat)
    "make_const",
    "make_add", "make_mul", "make_sub", "make_relu", "make_matmul",
    "make_sigmoid", "make_tanh", "make_softmax",
    "make_step", "make_transpose", "make_oneslike",
    "make_placeholder", "make_variable",
    # Eager-only ops
    "step", "transpose",
    # Misc
    "hello_cuda",
]
