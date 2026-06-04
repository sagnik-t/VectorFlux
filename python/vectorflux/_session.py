"""
vectorflux/_session.py — Session and TrainOp.

Session wraps the C++ session and adds:
  - Variable unwrapping in fetch lists and feed_dict keys
  - Automatic device casting of feed_dict tensors to the session's device
  - TrainOp dispatch (one-step: forward + backward + optimizer update)

TrainOp is returned by optimizer.minimize() and drives the training loop.
"""

from ._core import Session as _CppSession
from ._device import get_default_device
from ._variables import Variable
from ._ops import _unwrap, _tensor_to_numpy


# ── feed_dict helper ──────────────────────────────────────────────────────────

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
        self._vars       = var_list
        self._grad_nodes = [_unwrap(g) for g in grad_nodes]
        self._optimizer  = optimizer
        self._step       = 0

    def _execute(self, sess, feed_dict=None):
        """Called by Session.run() when a TrainOp is passed as fetch."""
        self._step += 1
        cpp_fd  = _build_cpp_fd(feed_dict, sess._device)
        results = sess._sess.run([self._loss_node] + self._grad_nodes, cpp_fd)

        loss_val   = results[0]
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
        if isinstance(fetches, TrainOp):
            return fetches._execute(self, feed_dict)

        cpp_fd = _build_cpp_fd(feed_dict, self._device)

        if isinstance(fetches, (list, tuple)):
            cpp_fetches = [f._node if isinstance(f, Variable) else f
                           for f in fetches]
            return self._sess.run(cpp_fetches, cpp_fd)

        cpp_fetch = fetches._node if isinstance(fetches, Variable) else fetches
        return self._sess.run(cpp_fetch, cpp_fd)
