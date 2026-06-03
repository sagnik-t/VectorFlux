"""
tests/test_losses.py — T14: Loss functions

Tests cover:
  - reduce_sum: eager value/shape, symbolic node type/value, gradient (numeric)
  - reduce_mean: eager value/shape, symbolic node type/value, gradient (numeric)
  - vf.losses.mse: correct value vs numpy, zero-loss case, gradient numeric check
  - vf.losses.softmax_cross_entropy: correct value vs scipy, perfect-prediction
    limit, numerical stability (large logits), gradient numeric check
"""

import numpy as np
import pytest
import vectorflux as vf


# ── Helpers ────────────────────────────────────────────────────────────────────

def t(arr):
    return vf.Tensor(arr.astype(np.float32))


@pytest.fixture(autouse=True)
def clean_graph():
    vf.reset_default_graph()
    yield
    vf.reset_default_graph()


def np_softmax_ce(logits, labels):
    """Reference: mean CE over columns.  logits/labels: [C, N]."""
    C, N = logits.shape
    total = 0.0
    for j in range(N):
        col = logits[:, j]
        max_val = col.max()
        log_sum_exp = np.log(np.sum(np.exp(col - max_val))) + max_val
        dot = np.dot(labels[:, j], col)
        total += log_sum_exp - dot
    return total / N


def numeric_grad(f, x, eps=1e-3):
    """Finite-difference gradient of scalar function f w.r.t. flat array x."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus  = x.copy(); x_plus.flat[i]  += eps
        x_minus = x.copy(); x_minus.flat[i] -= eps
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


# ══════════════════════════════════════════════════════════════════════════════
# reduce_sum
# ══════════════════════════════════════════════════════════════════════════════

class TestReduceSum:
    def test_eager_1d(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = vf.reduce_sum(t(na)).to_numpy()
        np.testing.assert_allclose(result, [6.0], atol=1e-6)

    def test_eager_2d(self):
        na = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = vf.reduce_sum(t(na)).to_numpy()
        np.testing.assert_allclose(result, [na.sum()], atol=1e-6)

    def test_eager_shape(self):
        na = np.ones((4, 5), dtype=np.float32)
        assert vf.reduce_sum(t(na)).shape == (1,)

    def test_eager_single_element(self):
        na = np.array([7.5], dtype=np.float32)
        np.testing.assert_allclose(
            vf.reduce_sum(t(na)).to_numpy(), [7.5], atol=1e-6)

    def test_symbolic_node_type(self):
        a = vf.make_const(t(np.ones(3, dtype=np.float32)))
        node = vf.reduce_sum(a)
        assert node.type == "ReduceSum"

    def test_symbolic_value(self):
        na = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a  = vf.make_const(t(na))
        sess = vf.Session()
        result = sess.run(vf.reduce_sum(a)).to_numpy()
        np.testing.assert_allclose(result, [na.sum()], atol=1e-6)

    def test_gradient_shape(self):
        """Gradient of reduce_sum w.r.t. input has same shape as input."""
        W = vf.Variable(t(np.ones((2, 3), dtype=np.float32)))
        loss = vf.reduce_sum(W)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        dW_val = sess.run(dW)
        assert dW_val.shape == (2, 3)

    def test_gradient_value(self):
        """d(sum(x))/dx_i = 1 for all i."""
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W  = vf.Variable(t(na))
        loss = vf.reduce_sum(W)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        dW_val = sess.run(dW).to_numpy()
        np.testing.assert_allclose(dW_val, np.ones(3), atol=1e-6)

    def test_gradient_numerical(self):
        """Numerical gradient check for reduce_sum."""
        na = np.array([0.5, -1.0, 2.0], dtype=np.float32)
        W  = vf.Variable(t(na))
        loss = vf.reduce_sum(W)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()

        eps = 1e-3
        analytical = sess.run(dW).to_numpy()
        numerical = numeric_grad(
            lambda x: vf.reduce_sum(t(x)).to_numpy()[0], na)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# reduce_mean
# ══════════════════════════════════════════════════════════════════════════════

class TestReduceMean:
    def test_eager_1d(self):
        na = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        result = vf.reduce_mean(t(na)).to_numpy()
        np.testing.assert_allclose(result, [4.0], atol=1e-6)

    def test_eager_2d(self):
        na = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = vf.reduce_mean(t(na)).to_numpy()
        np.testing.assert_allclose(result, [na.mean()], atol=1e-6)

    def test_eager_shape(self):
        assert vf.reduce_mean(t(np.ones((3, 4), dtype=np.float32))).shape == (1,)

    def test_symbolic_node_type(self):
        a = vf.make_const(t(np.ones(3, dtype=np.float32)))
        assert vf.reduce_mean(a).type == "ReduceMean"

    def test_symbolic_value(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a  = vf.make_const(t(na))
        sess = vf.Session()
        result = sess.run(vf.reduce_mean(a)).to_numpy()
        np.testing.assert_allclose(result, [na.mean()], atol=1e-6)

    def test_gradient_shape(self):
        W = vf.Variable(t(np.ones((2, 3), dtype=np.float32)))
        loss = vf.reduce_mean(W)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        assert sess.run(dW).shape == (2, 3)

    def test_gradient_value(self):
        """d(mean(x))/dx_i = 1/N for all i."""
        na = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        N  = float(na.size)
        W  = vf.Variable(t(na))
        loss = vf.reduce_mean(W)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        dW_val = sess.run(dW).to_numpy()
        np.testing.assert_allclose(dW_val, np.full(4, 1.0 / N), atol=1e-6)

    def test_gradient_numerical(self):
        na = np.array([0.5, -1.0, 2.0, 3.0], dtype=np.float32)
        W  = vf.Variable(t(na))
        loss = vf.reduce_mean(W)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        analytical = sess.run(dW).to_numpy()
        numerical  = numeric_grad(
            lambda x: vf.reduce_mean(t(x)).to_numpy()[0], na)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# MSE loss
# ══════════════════════════════════════════════════════════════════════════════

class TestMSELoss:
    def test_value_vs_numpy(self):
        """vf.losses.mse matches numpy computation."""
        np_pred   = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_target = np.array([[1.5, 1.0], [2.5, 5.0]], dtype=np.float32)
        expected  = np.mean((np_pred - np_target) ** 2)

        pred   = vf.make_const(t(np_pred))
        target = vf.make_const(t(np_target))
        loss   = vf.losses.mse(pred, target)
        sess   = vf.Session()
        np.testing.assert_allclose(
            sess.run(loss).to_numpy()[0], expected, rtol=1e-5)

    def test_zero_loss(self):
        """Identical prediction and target → loss == 0."""
        na   = np.ones((3, 2), dtype=np.float32)
        pred = target = vf.make_const(t(na))
        loss = vf.losses.mse(pred, target)
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(loss).to_numpy()[0], 0.0, atol=1e-7)

    def test_output_shape(self):
        """MSE loss is a scalar [1]."""
        pred   = vf.make_const(t(np.zeros((4, 8), dtype=np.float32)))
        target = vf.make_const(t(np.ones( (4, 8), dtype=np.float32)))
        loss   = vf.losses.mse(pred, target)
        sess   = vf.Session()
        assert sess.run(loss).shape == (1,)

    def test_positive_loss(self):
        """Non-identical tensors → positive loss."""
        pred   = vf.make_const(t(np.zeros((3,), dtype=np.float32)))
        target = vf.make_const(t(np.ones( (3,), dtype=np.float32)))
        sess   = vf.Session()
        assert sess.run(vf.losses.mse(pred, target)).to_numpy()[0] > 0

    def test_gradient_shape(self):
        """Gradient of MSE w.r.t. predictions has same shape as predictions."""
        np_pred   = np.array([[1.0], [2.0]], dtype=np.float32)
        np_target = np.array([[0.0], [3.0]], dtype=np.float32)
        W     = vf.Variable(t(np_pred))
        ph    = vf.placeholder([2, 1])
        loss  = vf.losses.mse(W, ph)
        [dW]  = vf.gradients(loss, [W])
        sess  = vf.Session()
        dW_val = sess.run(dW, feed_dict={ph: t(np_target)})
        assert dW_val.shape == (2, 1)

    def test_gradient_numerical(self):
        """Numerical gradient check for MSE."""
        np_pred   = np.array([1.5, -0.5, 2.0], dtype=np.float32)
        np_target = np.array([1.0,  0.5, 3.0], dtype=np.float32)
        W     = vf.Variable(t(np_pred))
        ph    = vf.placeholder([3])
        loss  = vf.losses.mse(W, ph)
        [dW]  = vf.gradients(loss, [W])
        sess  = vf.Session()
        analytical = sess.run(dW, feed_dict={ph: t(np_target)}).to_numpy()
        # Numeric: vary each component of pred
        numerical = numeric_grad(
            lambda x: np.mean((x - np_target) ** 2), np_pred)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Softmax cross-entropy loss
# ══════════════════════════════════════════════════════════════════════════════

def _one_hot(indices, C):
    """indices: [N] int → one-hot [C, N]."""
    N = len(indices)
    oh = np.zeros((C, N), dtype=np.float32)
    for j, idx in enumerate(indices):
        oh[idx, j] = 1.0
    return oh


class TestSoftmaxCrossEntropy:
    def test_value_vs_numpy_1sample(self):
        """Single-sample CE matches manual computation."""
        C  = 4
        logits = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        labels = _one_hot([2], C)   # class index 2
        expected = np_softmax_ce(logits, labels)

        l_node = vf.make_const(t(logits))
        y_node = vf.make_const(t(labels))
        loss   = vf.losses.softmax_cross_entropy(l_node, y_node)
        sess   = vf.Session()
        np.testing.assert_allclose(
            sess.run(loss).to_numpy()[0], expected, rtol=1e-5)

    def test_value_vs_numpy_batch(self):
        """Batch CE matches numpy reference."""
        C, N = 5, 8
        np.random.seed(42)
        logits = np.random.randn(C, N).astype(np.float32)
        labels = _one_hot(np.random.randint(0, C, N), C)
        expected = np_softmax_ce(logits, labels)

        l_node = vf.make_const(t(logits))
        y_node = vf.make_const(t(labels))
        loss   = vf.losses.softmax_cross_entropy(l_node, y_node)
        sess   = vf.Session()
        np.testing.assert_allclose(
            sess.run(loss).to_numpy()[0], expected, rtol=1e-4)

    def test_output_shape(self):
        """CE loss is a scalar [1]."""
        C, N = 3, 4
        l_node = vf.make_const(t(np.zeros((C, N), dtype=np.float32)))
        y_node = vf.make_const(t(_one_hot([0, 1, 2, 0], C)))
        sess   = vf.Session()
        assert sess.run(vf.losses.softmax_cross_entropy(l_node, y_node)).shape == (1,)

    def test_perfect_prediction_low_loss(self):
        """Very confident correct prediction yields near-zero loss."""
        C = 3
        # Logits: large value on the correct class, near-zero elsewhere
        logits = np.array([[10.0], [-5.0], [-5.0]], dtype=np.float32)
        labels = _one_hot([0], C)
        l_node = vf.make_const(t(logits))
        y_node = vf.make_const(t(labels))
        sess   = vf.Session()
        loss_val = sess.run(vf.losses.softmax_cross_entropy(l_node, y_node)).to_numpy()[0]
        assert loss_val < 0.01

    def test_numerical_stability_large_logits(self):
        """No NaN/Inf with large logit values (log-sum-exp check)."""
        C, N = 4, 2
        logits = np.array([[1000.0, -1000.0],
                           [  0.0,     0.0],
                           [  0.0,     0.0],
                           [  0.0,     0.0]], dtype=np.float32)
        labels = _one_hot([0, 1], C)
        l_node = vf.make_const(t(logits))
        y_node = vf.make_const(t(labels))
        sess   = vf.Session()
        result = sess.run(vf.losses.softmax_cross_entropy(l_node, y_node)).to_numpy()[0]
        assert np.isfinite(result)

    def test_gradient_shape(self):
        """CE gradient w.r.t. logits has same shape as logits."""
        C, N = 4, 3
        np_logits = np.random.randn(C, N).astype(np.float32)
        np_labels = _one_hot(np.random.randint(0, C, N), C)

        W    = vf.Variable(t(np_logits))
        ph   = vf.placeholder([C, N])
        loss = vf.losses.softmax_cross_entropy(W, ph)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={ph: t(np_labels)})
        assert dW_val.shape == (C, N)

    def test_gradient_numerical(self):
        """Numerical gradient check for CE loss w.r.t. logits."""
        C, N = 3, 4
        np.random.seed(7)
        np_logits = np.random.randn(C, N).astype(np.float32)
        np_labels = _one_hot([0, 1, 2, 0], C)

        W    = vf.Variable(t(np_logits))
        ph   = vf.placeholder([C, N])
        loss = vf.losses.softmax_cross_entropy(W, ph)
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        analytical = sess.run(dW, feed_dict={ph: t(np_labels)}).to_numpy()

        numerical = numeric_grad(
            lambda x: np_softmax_ce(x.reshape(C, N), np_labels),
            np_logits)
        np.testing.assert_allclose(
            analytical.ravel(), numerical.ravel(), rtol=5e-3, atol=1e-4)

    def test_node_type(self):
        """Graph node type is correct."""
        l = vf.make_const(t(np.zeros((3, 2), dtype=np.float32)))
        y = vf.make_const(t(np.zeros((3, 2), dtype=np.float32)))
        assert vf.losses.softmax_cross_entropy(l, y).type == \
               "SoftmaxCrossEntropyWithLogits"

    def test_nn_alias(self):
        """vf.nn.softmax_cross_entropy is the same op."""
        l = vf.make_const(t(np.zeros((3, 2), dtype=np.float32)))
        y = vf.make_const(t(np.zeros((3, 2), dtype=np.float32)))
        assert vf.nn.softmax_cross_entropy(l, y).type == \
               "SoftmaxCrossEntropyWithLogits"
