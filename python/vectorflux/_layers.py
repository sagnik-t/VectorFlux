"""
vectorflux/_layers.py — Dense layer and vf.nn / vf.layers submodules.
"""

import numpy as np

from ._core import Tensor
from ._variables import Variable
from ._ops import matmul, relu, sigmoid, tanh, softmax


# ── Dense ─────────────────────────────────────────────────────────────────────

class Dense:
    """
    Fully-connected layer:  out = W @ x  (optionally followed by an activation).

    Parameters
    ----------
    input_dim  : int            — number of input features
    units      : int            — number of output units
    activation : callable|None — e.g. vf.nn.relu, vf.sigmoid, vf.tanh
    name       : str            — prefix for the weight variable name

    Note: no bias term — broadcast-add is not yet implemented.

    Weight initialisation: Xavier uniform, limit = sqrt(6 / (input_dim + units))
    Weights are automatically placed on the current default device.

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

        limit  = np.sqrt(6.0 / (input_dim + units))
        nW     = np.random.uniform(-limit, limit,
                                    (units, input_dim)).astype(np.float32)
        w_name = (name + "/W") if name else ""
        self._W = Variable(Tensor(nW), name=w_name)

    @property
    def weights(self):
        """The weight Variable."""
        return self._W

    def __call__(self, x):
        """Wire the layer into the graph and return the output node."""
        out = matmul(self._W, x)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def __repr__(self):
        act = (self.activation.__name__
               if callable(self.activation) else str(self.activation))
        return (f"Dense(input_dim={self.input_dim}, units={self.units}, "
                f"activation={act})")


# ── vf.nn submodule ───────────────────────────────────────────────────────────

class _NNModule:
    """Neural-network activation ops as graph nodes (vf.nn.*)."""

    @staticmethod
    def relu(a, name=""):    return relu(a, name)

    @staticmethod
    def sigmoid(a, name=""): return sigmoid(a, name)

    @staticmethod
    def tanh(a, name=""):    return tanh(a, name)

    @staticmethod
    def softmax(a, name=""): return softmax(a, name)

    @staticmethod
    def softmax_cross_entropy(logits, labels, name=""):
        # Lazy import to avoid circular dependency at module load time.
        from ._losses import losses
        return losses.softmax_cross_entropy(logits, labels, name)


nn = _NNModule()


# ── vf.layers submodule ───────────────────────────────────────────────────────

class _LayersModule:
    """Layer constructors (vf.layers.*)."""
    Dense = Dense


layers = _LayersModule()
