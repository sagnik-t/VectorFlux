"""
vectorflux/_optimizers.py — GradientDescentOptimizer, AdamOptimizer, vf.train.

Both optimizers preserve the variable's current device when applying updates:
weights on CUDA stay on CUDA after each step.
"""

import numpy as np

from ._core import Tensor
from ._variables import _get_all_variables
from ._ops import gradients
from ._session import TrainOp


# ── SGD ───────────────────────────────────────────────────────────────────────

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
        Build gradient nodes and return a TrainOp.

        Parameters
        ----------
        loss     : scalar graph node
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
        """Apply w ← w - lr * g, preserving each variable's device."""
        for var, grad_np in var_grad_pairs:
            dev   = var.device
            new_w = Tensor(np.asarray(var.numpy - self.lr * grad_np,
                                      dtype=np.float32))
            if dev == 'cuda':
                new_w = new_w.to('cuda')
            var.assign(new_w)


# ── Adam ──────────────────────────────────────────────────────────────────────

class AdamOptimizer:
    """
    Adam optimizer (Kingma & Ba, 2015).

    Default hyperparameters:
        lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8

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
        self._m: dict = {}
        self._v: dict = {}

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
        """Apply Adam update, preserving each variable's device."""
        for var, grad_np in var_grad_pairs:
            dev = var.device
            vid = id(var)

            if vid not in self._m:
                self._m[vid] = np.zeros_like(grad_np)
                self._v[vid] = np.zeros_like(grad_np)

            self._m[vid] = (self.beta1 * self._m[vid]
                            + (1.0 - self.beta1) * grad_np)
            self._v[vid] = (self.beta2 * self._v[vid]
                            + (1.0 - self.beta2) * grad_np ** 2)

            m_hat  = self._m[vid] / (1.0 - self.beta1 ** step)
            v_hat  = self._v[vid] / (1.0 - self.beta2 ** step)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            new_w = Tensor(np.asarray(var.numpy - update, dtype=np.float32))
            if dev == 'cuda':
                new_w = new_w.to('cuda')
            var.assign(new_w)


# ── vf.train submodule ────────────────────────────────────────────────────────

class _TrainModule:
    """Optimizer constructors (vf.train.*)."""
    GradientDescentOptimizer = GradientDescentOptimizer
    AdamOptimizer            = AdamOptimizer


train = _TrainModule()
