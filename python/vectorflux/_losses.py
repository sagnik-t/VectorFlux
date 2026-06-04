"""
vectorflux/_losses.py — Loss functions submodule (vf.losses).

All losses return a scalar graph node (shape [1]) and are fully
differentiable via the autograd engine.
"""

from ._core import make_softmax_cross_entropy_with_logits
from ._ops import sub, mul, reduce_mean, _unwrap


class _LossesModule:
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


losses = _LossesModule()
