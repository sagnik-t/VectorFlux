"""
vectorflux/_ops.py — Overloaded graph / eager ops and gradients.

Every public op is symbolic when passed a Node or Variable, and falls back
to an eager Tensor computation otherwise.  This lets the same function work
both inside a graph (build time) and in standalone numpy-style code.
"""

from ._core import (
    Tensor,
    Node,
    add      as _eager_add,
    mul      as _eager_mul,
    sub      as _eager_sub,
    relu     as _eager_relu,
    sigmoid  as _eager_sigmoid,
    tanh     as _eager_tanh,
    softmax  as _eager_softmax,
    matmul   as _eager_matmul,
    reduce_sum  as _eager_reduce_sum,
    reduce_mean as _eager_reduce_mean,
    make_add, make_mul, make_sub, make_relu, make_matmul,
    make_sigmoid, make_tanh, make_softmax,
    make_step, make_transpose, make_oneslike,
    make_reduce_sum, make_reduce_mean,
    gradients as _core_gradients,
)
from ._variables import Variable


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unwrap(x):
    """Return the underlying NodeRef for a Variable, or x unchanged."""
    if isinstance(x, Variable):
        return x._node
    return x


def _is_node(x):
    """True if x is a graph node (NodeRef or Variable)."""
    return isinstance(x, (Node, Variable))


def _tensor_to_numpy(tensor):
    """Move tensor to CPU if needed, then return a numpy array."""
    if tensor.device == 'cuda':
        tensor = tensor.to('cpu')
    return tensor.to_numpy()


# ── Element-wise ops ──────────────────────────────────────────────────────────

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


# ── Activation ops ────────────────────────────────────────────────────────────

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


# ── Reduction ops ─────────────────────────────────────────────────────────────

def reduce_sum(a, name=""):
    """Sum all elements → scalar [1].  Symbolic or eager."""
    if _is_node(a):
        return make_reduce_sum(_unwrap(a), name)
    return _eager_reduce_sum(a)


def reduce_mean(a, name=""):
    """Mean of all elements → scalar [1].  Symbolic or eager."""
    if _is_node(a):
        return make_reduce_mean(_unwrap(a), name)
    return _eager_reduce_mean(a)


# ── Gradients ─────────────────────────────────────────────────────────────────

def gradients(ys, xs):
    """Compute d(sum(ys)) / d(xs[i]).  xs may contain Node or Variable objects."""
    return _core_gradients(_unwrap(ys), [_unwrap(x) for x in xs])
