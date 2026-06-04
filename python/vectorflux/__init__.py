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
    # T14: eager reductions
    reduce_sum  as _eager_reduce_sum,
    reduce_mean as _eager_reduce_mean,
    # Graph node type
    Node,
    # Low-level graph builders — kept for backward compat and internal use.
    make_const,
    make_add, make_mul, make_sub, make_relu, make_matmul,
    make_sigmoid, make_tanh, make_softmax,
    make_step, make_transpose, make_oneslike,
    make_placeholder, make_variable,
    make_reduce_sum, make_reduce_mean,                  # T14
    make_softmax_cross_entropy_with_logits,             # T14
    variable_assign,
    variable_get_value,                                 # T14
    global_variables_initializer,
    reset_default_graph as _reset_graph_cpp,
    gradients as _core_gradients,
    Session   as _CppSession,
)

import numpy as np
import weakref


# ── Variable registry ─────────────────────────────────────────────────────────
# Tracks live Variable objects so optimizers can auto-discover all parameters.

_variable_registry: list = []


def _register_variable(var) -> None:
    _variable_registry.append(weakref.ref(var))


def _get_all_variables() -> list:
    """Return all currently live Variable objects in this graph scope."""
    live = [r() for r in _variable_registry if r() is not None]
    return live


def _clear_variable_registry() -> None:
    global _variable_registry
    _variable_registry = []


def reset_default_graph() -> None:
    """Clear the C++ graph and the Python variable registry."""
    _reset_graph_cpp()
    _clear_variable_registry()


# ── Default device ────────────────────────────────────────────────────────────
# Controls where Variables are placed and where feed_dict tensors are moved.
# Call vf.set_default_device('cuda') before building the graph to run on GPU.
# Session captures this at construction time, so set it before creating one.

_default_device: str = 'cpu'


def set_default_device(device: str) -> None:
    """
    Set the default device for the framework.

    Parameters
    ----------
    device : 'cpu' or 'cuda'

    Effects
    -------
    - New Variable weights are placed on this device.
    - Sessions created after this call auto-move feed_dict tensors to this device.
    - Optimizer updates preserve the variable's current device automatically.

    Call this once before building the graph, e.g.:
        vf.set_default_device('cuda')
        X_ph, Y_ph, logits, train_op = build_model()
        sess = vf.Session()
    """
    global _default_device
    if device not in ('cpu', 'cuda'):
        raise ValueError(
            f"set_default_device: unknown device '{device}'; "
            "expected 'cpu' or 'cuda'")
    _default_device = device


def get_default_device() -> str:
    """Return the current default device ('cpu' or 'cuda')."""
    return _default_device


# ── placeholder ───────────────────────────────────────────────────────────────

def placeholder(shape, name=""):
    """Declare a Placeholder node.  Value must be supplied via feed_dict."""
    return make_placeholder(shape, name)


# ── Variable ──────────────────────────────────────────────────────────────────

class Variable:
    """
    A trainable parameter.  Wraps a Variable graph node and exposes:
      - .assign(value)  — update the stored value (accepts Tensor or ndarray)
      - .numpy           — current value as a numpy array (always CPU)
      - .device          — current device ('cpu' or 'cuda')
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
        # Honour the default device: move to GPU if needed.
        if initial_value.device != _default_device:
            initial_value = initial_value.to(_default_device)
        self._node = make_variable(initial_value, name)
        _register_variable(self)

    @property
    def name(self):      return self._node.name
    @property
    def type(self):      return self._node.type
    @property
    def inputs(self):    return self._node.inputs
    @property
    def evaluated(self): return self._node.evaluated

    @property
    def device(self) -> str:
        """Current device of the stored parameter tensor ('cpu' or 'cuda')."""
        return variable_get_value(self._node).device

    @property
    def numpy(self) -> np.ndarray:
        """Current parameter values as a numpy array (always on CPU)."""
        val = variable_get_value(self._node)
        if val.device == "cuda":
            val = val.to("cpu")
        return val.to_numpy()

    def __repr__(self):
        return f"Variable(name={self.name})"

    def assign(self, value) -> None:
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


def _build_cpp_fd(feed_dict, device: str = 'cpu'):
    """
    Convert a Python feed_dict (Variable/Node → Tensor) to C++ format,
    auto-moving each tensor to `device` if it isn't already there.
    """
    cpp_fd = {}
    for k, v in (feed_dict or {}).items():
        key = k._node if isinstance(k, Variable) else k
        if v.device != device:
            v = v.to(device)
        cpp_fd[key] = v
    return cpp_fd


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Move to CPU if needed, then return numpy array."""
    if tensor.device == "cuda":
        tensor = tensor.to("cpu")
    return tensor.to_numpy()


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


# ── T14: overloaded reduction ops ─────────────────────────────────────────────

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


# ── gradients ─────────────────────────────────────────────────────────────────

def gradients(ys, xs):
    """Compute d(sum(ys)) / d(xs[i]).  xs may contain Node or Variable objects."""
    return _core_gradients(_unwrap(ys), [_unwrap(x) for x in xs])


# ── TrainOp ───────────────────────────────────────────────────────────────────

class TrainOp:
    """
    Returned by optimizer.minimize(loss).  Pass to sess.run() to execute one
    training step: evaluates loss + gradients, applies variable updates,
    and returns the loss Tensor.

    Usage:
        train_op = optimizer.minimize(loss)
        for x_batch, y_batch in batches:
            loss_val = sess.run(train_op, feed_dict={x: x_batch, y: y_batch})
            print(loss_val.to_numpy()[0])
    """

    def __init__(self, loss_node, var_list, grad_nodes, optimizer):
        self._loss_node  = _unwrap(loss_node)
        self._vars       = var_list          # list of Python Variable objects
        self._grad_nodes = [_unwrap(g) for g in grad_nodes]
        self._optimizer  = optimizer
        self._step       = 0

    def _execute(self, sess, feed_dict=None) -> Tensor:
        """Called by Session.run() when a TrainOp is passed as fetch."""
        self._step += 1
        cpp_fd   = _build_cpp_fd(feed_dict, sess._device)
        results  = sess._sess.run([self._loss_node] + self._grad_nodes, cpp_fd)
        loss_val = results[0]

        grad_numpy = [_tensor_to_numpy(results[i + 1])
                      for i in range(len(self._vars))]

        self._optimizer.apply_gradients(
            list(zip(self._vars, grad_numpy)), self._step)
        return loss_val


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """
    TF1-style session.  Accepts Variable objects wherever a NodeRef is expected.
    Also handles TrainOp: sess.run(train_op) executes a full training step.

    The session captures the default device at construction time and
    automatically moves all feed_dict tensors to that device before execution —
    so user code never needs to call .to('cuda') on batches manually.

    Usage:
        sess = vf.Session()
        result  = sess.run(output_node)
        results = sess.run([node_a, node_b])
        result  = sess.run(node, feed_dict={placeholder: value})
        loss    = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
    """

    def __init__(self):
        self._sess   = _CppSession()
        self._device = get_default_device()

    def run(self, fetches, feed_dict=None):
        # TrainOp: evaluate + apply updates
        if isinstance(fetches, TrainOp):
            return fetches._execute(self, feed_dict)

        cpp_fd = _build_cpp_fd(feed_dict, self._device)

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
        # Variable.__init__ will honour _default_device, placing W on GPU if set.
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
    """Neural-network activation and loss ops as graph nodes."""
    @staticmethod
    def relu(a, name=""):    return relu(a, name)

    @staticmethod
    def sigmoid(a, name=""): return sigmoid(a, name)

    @staticmethod
    def tanh(a, name=""):    return tanh(a, name)

    @staticmethod
    def softmax(a, name=""): return softmax(a, name)

    @staticmethod
    def softmax_cross_entropy(logits, labels, name=""):  # T14 alias
        return losses.softmax_cross_entropy(logits, labels, name)


nn = _nn()


# ── vf.layers submodule ───────────────────────────────────────────────────────

class _layers:
    """Layer constructors."""
    Dense = Dense


layers = _layers()


# ── vf.losses submodule ───────────────────────────────────────────────────────

class _losses:
    """
    Loss functions that return a scalar (shape [1]) graph node.

    vf.losses.mse(pred, target)
        Mean squared error: reduce_mean((pred - target)^2)
        pred and target must have the same shape.
        Fully differentiable via autograd (no custom grad op needed).

    vf.losses.softmax_cross_entropy(logits, labels)
        Fused softmax + cross-entropy loss.
        logits: [C, N]  labels: [C, N] one-hot
        Returns mean CE over N samples.
        Uses numerically stable log-sum-exp; custom gradient op.
    """

    @staticmethod
    def mse(pred, labels, name=""):
        """MSE loss: reduce_mean((pred - labels)^2)"""
        diff = sub(pred, labels)
        sq   = mul(diff, diff)
        return reduce_mean(sq, name)

    @staticmethod
    def softmax_cross_entropy(logits, labels, name=""):
        """
        Fused softmax cross-entropy loss (numerically stable).
        logits: [C, N]  labels: [C, N] one-hot
        """
        return make_softmax_cross_entropy_with_logits(
            _unwrap(logits), _unwrap(labels), name)


losses = _losses()


# ── vf.train submodule ────────────────────────────────────────────────────────

class GradientDescentOptimizer:
    """
    Vanilla SGD: w ← w - lr * grad

    Usage:
        optimizer = vf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op  = optimizer.minimize(loss)
        sess.run(train_op, feed_dict={...})
    """

    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def minimize(self, loss, var_list=None):
        """
        Build gradient nodes for all variables and return a TrainOp.

        Parameters
        ----------
        loss     : graph node that evaluates to a scalar loss
        var_list : list of Variable objects, or None to use all registered vars
        """
        if var_list is None:
            var_list = _get_all_variables()
        if not var_list:
            raise RuntimeError(
                "GradientDescentOptimizer.minimize: no variables found. "
                "Pass var_list explicitly or create Variable objects before "
                "calling minimize().")
        grad_nodes = gradients(loss, var_list)
        return TrainOp(loss, var_list, grad_nodes, self)

    def apply_gradients(self, var_grad_pairs, step: int) -> None:
        """Apply w ← w - lr * g for each (variable, gradient) pair,
        preserving the variable's current device."""
        for var, grad_np in var_grad_pairs:
            dev   = var.device
            new_w = Tensor(np.asarray(var.numpy - self.lr * grad_np,
                                      dtype=np.float32))
            if dev == 'cuda':
                new_w = new_w.to('cuda')
            var.assign(new_w)


class AdamOptimizer:
    """
    Adam optimizer: adaptive moment estimation.
    Default hyperparameters from the original paper (Kingma & Ba, 2015).

    Usage:
        optimizer = vf.train.AdamOptimizer(learning_rate=0.001)
        train_op  = optimizer.minimize(loss)
        sess.run(train_op, feed_dict={...})
    """

    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.lr      = learning_rate
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self._m: dict = {}   # first moment vectors  (keyed by id(Variable))
        self._v: dict = {}   # second moment vectors

    def minimize(self, loss, var_list=None):
        """Build gradient nodes and return a TrainOp."""
        if var_list is None:
            var_list = _get_all_variables()
        if not var_list:
            raise RuntimeError(
                "AdamOptimizer.minimize: no variables found. "
                "Pass var_list explicitly or create Variable objects before "
                "calling minimize().")
        grad_nodes = gradients(loss, var_list)
        return TrainOp(loss, var_list, grad_nodes, self)

    def apply_gradients(self, var_grad_pairs, step: int) -> None:
        """
        Apply Adam update for each (variable, gradient) pair.
        step is the 1-indexed step count provided by TrainOp.
        Preserves the variable's current device (CPU or CUDA).
        """
        for var, grad_np in var_grad_pairs:
            dev = var.device
            vid = id(var)
            if vid not in self._m:
                self._m[vid] = np.zeros_like(grad_np)
                self._v[vid] = np.zeros_like(grad_np)

            self._m[vid] = self.beta1 * self._m[vid] + (1.0 - self.beta1) * grad_np
            self._v[vid] = self.beta2 * self._v[vid] + (1.0 - self.beta2) * grad_np ** 2

            m_hat = self._m[vid] / (1.0 - self.beta1 ** step)
            v_hat = self._v[vid] / (1.0 - self.beta2 ** step)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            new_w  = Tensor(np.asarray(var.numpy - update, dtype=np.float32))
            if dev == 'cuda':
                new_w = new_w.to('cuda')
            var.assign(new_w)


class _train:
    """vf.train — optimizers."""
    GradientDescentOptimizer = GradientDescentOptimizer
    AdamOptimizer            = AdamOptimizer


train = _train()


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Core types
    "Tensor", "Node", "Variable",
    # Graph-building ops
    "placeholder",
    "add", "mul", "sub", "relu", "sigmoid", "tanh", "softmax", "matmul",
    "reduce_sum", "reduce_mean",                        # T14
    # Submodules
    "nn", "layers", "losses", "train",                 # T14: losses, train
    # High-level layer
    "Dense",
    # Training helpers
    "gradients",
    "variable_assign",
    "global_variables_initializer",
    "TrainOp",                                         # T14
    "GradientDescentOptimizer",                        # T14
    "AdamOptimizer",                                   # T14
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
    "make_reduce_sum", "make_reduce_mean",             # T14
    "make_softmax_cross_entropy_with_logits",          # T14
    "make_step", "make_transpose", "make_oneslike",
    "make_placeholder", "make_variable",
    # Eager-only ops
    "step", "transpose",
    # Misc
    "hello_cuda",
]
