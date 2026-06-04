"""
vectorflux/_variables.py — Variable class and graph-scoped variable registry.

The registry lets optimizers auto-discover all trainable parameters without
the user having to pass var_list explicitly.
"""

import numpy as np
import weakref

from ._core import (
    Tensor,
    make_variable,
    variable_assign,
    variable_get_value,
    reset_default_graph as _reset_graph_cpp,
)
from ._device import get_default_device


# ── Variable registry ─────────────────────────────────────────────────────────

_variable_registry: list = []


def _register_variable(var) -> None:
    _variable_registry.append(weakref.ref(var))


def _get_all_variables() -> list:
    """Return all currently live Variable objects in this graph scope."""
    return [r() for r in _variable_registry if r() is not None]


def _clear_variable_registry() -> None:
    global _variable_registry
    _variable_registry = []


def reset_default_graph() -> None:
    """Clear the C++ graph and the Python variable registry."""
    _reset_graph_cpp()
    _clear_variable_registry()


# ── Variable ──────────────────────────────────────────────────────────────────

class Variable:
    """
    A trainable parameter.  Wraps a Variable graph node and exposes:
      - .assign(value)  — update the stored value (accepts Tensor or ndarray)
      - .numpy           — current value as a numpy array (always CPU)
      - .device          — current device ('cpu' or 'cuda')
      - .name / .type / .inputs / .evaluated — mirrors the Node interface

    Can be passed anywhere a NodeRef is expected in the graph API.

    The initial value is automatically placed on the current default device
    (see vf.set_default_device).

    Usage:
        W = vf.Variable(np.random.randn(784, 128).astype(np.float32))
        W.assign(new_weights)
        y = vf.matmul(W, x)
    """

    def __init__(self, initial_value, name=""):
        if not isinstance(initial_value, Tensor):
            initial_value = Tensor(np.asarray(initial_value, dtype=np.float32))
        dev = get_default_device()
        if initial_value.device != dev:
            initial_value = initial_value.to(dev)
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
        if val.device == 'cuda':
            val = val.to('cpu')
        return val.to_numpy()

    def assign(self, value) -> None:
        """Update the variable's current value (accepts Tensor or numpy array)."""
        if not isinstance(value, Tensor):
            value = Tensor(np.asarray(value, dtype=np.float32))
        variable_assign(self._node, value)

    def __repr__(self):
        return f"Variable(name={self.name})"
