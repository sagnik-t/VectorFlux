"""
tests/test_activations.py — T13: Dense layer + activations

Tests cover:
  - sub: eager (correct value, shape, device parity)
  - sigmoid: eager (values, numerics), graph node (type, correct value, gradient)
  - tanh: eager (values), graph node (type, correct value, gradient)
  - softmax: eager 1-D (sums to 1, correct), eager 2-D (each column sums to 1),
             graph node (type, correct value), numerical stability (large inputs)
  - vf.nn.sigmoid / vf.nn.tanh / vf.nn.softmax
  - Dense: construction, call returns node, correct forward value,
           different activations, weights accessible, Xavier init range,
           multi-layer stack
  - vf.layers.Dense alias
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


def np_sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def np_softmax_col(x):
    """Softmax along axis 0 for 1-D or 2-D (per column)."""
    if x.ndim == 1:
        e = np.exp(x - x.max())
        return e / e.sum()
    else:
        e = np.exp(x - x.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# sub
# ══════════════════════════════════════════════════════════════════════════════

class TestSub:
    def test_eager_correct(self):
        a = np.array([3.0, 5.0, 10.0], dtype=np.float32)
        b = np.array([1.0, 2.0,  4.0], dtype=np.float32)
        result = vf.sub(t(a), t(b))
        np.testing.assert_array_equal(result.to_numpy(), a - b)

    def test_eager_shape_preserved(self):
        a = np.ones((3, 4), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        assert vf.sub(t(a), t(b)).shape == (3, 4)

    def test_symbolic_node_type(self):
        ca = vf.make_const(t(np.array([1.0, 2.0])))
        cb = vf.make_const(t(np.array([0.5, 1.0])))
        assert vf.sub(ca, cb).type == "Sub"

    def test_symbolic_correct_value(self):
        na = np.array([3.0, 5.0], dtype=np.float32)
        nb = np.array([1.0, 2.0], dtype=np.float32)
        ca = vf.make_const(t(na))
        cb = vf.make_const(t(nb))
        sess = vf.Session()
        np.testing.assert_array_equal(
            sess.run(vf.sub(ca, cb)).to_numpy(), na - nb)


# ══════════════════════════════════════════════════════════════════════════════
# sigmoid
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmoid:
    def test_eager_values(self):
        na = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = vf.sigmoid(t(na)).to_numpy()
        np.testing.assert_allclose(result, np_sigmoid(na), atol=1e-6)

    def test_eager_output_range(self):
        na = np.linspace(-10, 10, 100, dtype=np.float32)
        result = vf.sigmoid(t(na)).to_numpy()
        assert np.all(result > 0) and np.all(result < 1)

    def test_eager_at_zero(self):
        result = vf.sigmoid(t(np.array([0.0]))).to_numpy()
        np.testing.assert_allclose(result, [0.5], atol=1e-6)

    def test_symbolic_node_type(self):
        a = vf.make_const(t(np.array([1.0, 2.0])))
        assert vf.sigmoid(a).type == "Sigmoid"

    def test_symbolic_correct_value(self):
        na = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        a  = vf.make_const(t(na))
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(vf.sigmoid(a)).to_numpy(), np_sigmoid(na), atol=1e-6)

    def test_gradient_shape(self):
        W = vf.Variable(t(np.array([[1.0, 2.0], [3.0, 4.0]])))
        x = vf.placeholder([2, 1])
        y = vf.sigmoid(vf.matmul(W, x))
        [dW] = vf.gradients(y, [W])
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={x: t(np.ones((2, 1), dtype=np.float32))})
        assert dW_val.shape == (2, 2)

    def test_gradient_correct(self):
        """Numerical gradient check for sigmoid."""
        na = np.array([[0.5], [-0.5]], dtype=np.float32)
        nW = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        W  = vf.Variable(t(nW))
        x  = vf.placeholder([2, 1])
        y  = vf.sigmoid(vf.matmul(W, x))

        # Sum output to get scalar loss
        # We'll check gradient numerically via finite differences on W[0,0]
        eps = 1e-3
        def f(w00):
            s = np_sigmoid(np.array([[w00, 0.0], [0.0, 1.0]]) @ na)
            return s.sum()

        numerical_dw00 = (f(nW[0, 0] + eps) - f(nW[0, 0] - eps)) / (2 * eps)

        # Sum node — use add chain (no explicit sum op yet; use a 2×1 ones const)
        ones = vf.make_const(t(np.ones((1, 2), dtype=np.float32)))
        loss = vf.matmul(ones, y)   # [1,1] scalar
        [dW] = vf.gradients(loss, [W])
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={x: t(na)}).to_numpy()
        np.testing.assert_allclose(dW_val[0, 0], numerical_dw00, rtol=1e-3)

    def test_nn_sigmoid(self):
        a = vf.make_const(t(np.array([0.0])))
        assert vf.nn.sigmoid(a).type == "Sigmoid"
        sess = vf.Session()
        np.testing.assert_allclose(sess.run(vf.nn.sigmoid(a)).to_numpy(), [0.5], atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# tanh
# ══════════════════════════════════════════════════════════════════════════════

class TestTanh:
    def test_eager_values(self):
        na = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = vf.tanh(t(na)).to_numpy()
        np.testing.assert_allclose(result, np.tanh(na), atol=1e-6)

    def test_eager_output_range(self):
        na = np.linspace(-5, 5, 50, dtype=np.float32)
        result = vf.tanh(t(na)).to_numpy()
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_eager_at_zero(self):
        result = vf.tanh(t(np.array([0.0]))).to_numpy()
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_symbolic_node_type(self):
        a = vf.make_const(t(np.array([1.0])))
        assert vf.tanh(a).type == "Tanh"

    def test_symbolic_correct_value(self):
        na = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        a  = vf.make_const(t(na))
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(vf.tanh(a)).to_numpy(), np.tanh(na), atol=1e-6)

    def test_gradient_shape(self):
        W = vf.Variable(t(np.eye(2, dtype=np.float32)))
        x = vf.placeholder([2, 1])
        y = vf.tanh(vf.matmul(W, x))
        [dW] = vf.gradients(y, [W])
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={x: t(np.ones((2, 1), dtype=np.float32))})
        assert dW_val.shape == (2, 2)

    def test_gradient_correct(self):
        """Numerical gradient check for tanh."""
        na = np.array([[0.3], [0.7]], dtype=np.float32)
        nW = np.eye(2, dtype=np.float32)
        W  = vf.Variable(t(nW))
        x  = vf.placeholder([2, 1])

        ones = vf.make_const(t(np.ones((1, 2), dtype=np.float32)))
        loss = vf.matmul(ones, vf.tanh(vf.matmul(W, x)))
        [dW] = vf.gradients(loss, [W])

        eps = 1e-3
        def f(w00):
            return np.tanh(
                np.array([[w00, 0.0], [0.0, 1.0]]) @ na
            ).sum()

        numerical_dw00 = (f(nW[0, 0] + eps) - f(nW[0, 0] - eps)) / (2 * eps)
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={x: t(na)}).to_numpy()
        np.testing.assert_allclose(dW_val[0, 0], numerical_dw00, rtol=1e-3)

    def test_nn_tanh(self):
        a = vf.make_const(t(np.array([0.0])))
        assert vf.nn.tanh(a).type == "Tanh"
        sess = vf.Session()
        np.testing.assert_allclose(sess.run(vf.nn.tanh(a)).to_numpy(), [0.0], atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# softmax
# ══════════════════════════════════════════════════════════════════════════════

class TestSoftmax:
    def test_eager_1d_sums_to_one(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = vf.softmax(t(na)).to_numpy()
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_eager_1d_correct(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = vf.softmax(t(na)).to_numpy()
        np.testing.assert_allclose(result, np_softmax_col(na), atol=1e-6)

    def test_eager_1d_nonneg(self):
        na = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = vf.softmax(t(na)).to_numpy()
        assert np.all(result >= 0)

    def test_eager_2d_columns_sum_to_one(self):
        na = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = vf.softmax(t(na)).to_numpy()
        np.testing.assert_allclose(result.sum(axis=0), [1.0, 1.0], atol=1e-6)

    def test_eager_2d_correct(self):
        na = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = vf.softmax(t(na)).to_numpy()
        np.testing.assert_allclose(result, np_softmax_col(na), atol=1e-6)

    def test_numerical_stability_large_input(self):
        """Large logits must not produce NaN or Inf."""
        na = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = vf.softmax(t(na)).to_numpy()
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_symbolic_node_type(self):
        a = vf.make_const(t(np.array([1.0, 2.0, 3.0])))
        assert vf.softmax(a).type == "Softmax"

    def test_symbolic_correct_value(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a  = vf.make_const(t(na))
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(vf.softmax(a)).to_numpy(), np_softmax_col(na), atol=1e-6)

    def test_nn_softmax(self):
        a = vf.make_const(t(np.array([1.0, 2.0, 3.0])))
        assert vf.nn.softmax(a).type == "Softmax"

    def test_uniform_input_gives_uniform_output(self):
        na = np.zeros(5, dtype=np.float32)
        result = vf.softmax(t(na)).to_numpy()
        np.testing.assert_allclose(result, np.full(5, 0.2), atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Dense layer
# ══════════════════════════════════════════════════════════════════════════════

class TestDense:
    def test_construction(self):
        layer = vf.Dense(4, 8)
        assert layer.input_dim == 4
        assert layer.units == 8

    def test_weights_variable(self):
        layer = vf.Dense(4, 8)
        assert isinstance(layer.weights, vf.Variable)

    def test_weights_shape(self):
        layer = vf.Dense(4, 8)
        sess = vf.Session()
        W = sess.run(layer.weights)
        assert W.shape == (8, 4)

    def test_call_returns_node(self):
        layer = vf.Dense(4, 8)
        x = vf.placeholder([4, 1])
        out = layer(x)
        assert isinstance(out, vf.Node)

    def test_forward_no_activation(self):
        nW = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        nx = np.array([[3.0], [4.0]], dtype=np.float32)
        layer = vf.Dense(2, 2)
        layer.weights.assign(t(nW))
        x   = vf.placeholder([2, 1])
        out = layer(x)
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(out, feed_dict={x: t(nx)}).to_numpy(), nW @ nx, atol=1e-5)

    def test_forward_with_relu(self):
        nW = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        nx = np.array([[1.0], [1.0]], dtype=np.float32)
        layer = vf.Dense(2, 2, activation=vf.nn.relu)
        layer.weights.assign(t(nW))
        x   = vf.placeholder([2, 1])
        out = layer(x)
        sess = vf.Session()
        expected = np.maximum(0, nW @ nx)
        np.testing.assert_allclose(
            sess.run(out, feed_dict={x: t(nx)}).to_numpy(), expected, atol=1e-5)

    def test_forward_with_sigmoid(self):
        nW = np.eye(2, dtype=np.float32)
        nx = np.array([[0.0], [0.0]], dtype=np.float32)
        layer = vf.Dense(2, 2, activation=vf.nn.sigmoid)
        layer.weights.assign(t(nW))
        x   = vf.placeholder([2, 1])
        out = layer(x)
        sess = vf.Session()
        result = sess.run(out, feed_dict={x: t(nx)}).to_numpy()
        np.testing.assert_allclose(result, np.full((2, 1), 0.5), atol=1e-5)

    def test_forward_with_tanh(self):
        nW = np.eye(2, dtype=np.float32)
        nx = np.array([[0.0], [0.0]], dtype=np.float32)
        layer = vf.Dense(2, 2, activation=vf.nn.tanh)
        layer.weights.assign(t(nW))
        x   = vf.placeholder([2, 1])
        out = layer(x)
        sess = vf.Session()
        result = sess.run(out, feed_dict={x: t(nx)}).to_numpy()
        np.testing.assert_allclose(result, np.zeros((2, 1)), atol=1e-5)

    def test_xavier_init_range(self):
        """Xavier limit = sqrt(6 / (fan_in + fan_out)); weights must be in range."""
        in_dim, out_dim = 128, 64
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        layer = vf.Dense(in_dim, out_dim)
        sess  = vf.Session()
        W = sess.run(layer.weights).to_numpy()
        assert W.min() >= -limit - 1e-5
        assert W.max() <=  limit + 1e-5

    def test_named_weights(self):
        layer = vf.Dense(4, 8, name="fc1")
        assert "fc1" in layer.weights.name

    def test_multi_layer_stack(self):
        """Two Dense layers chained: layer2(layer1(x))."""
        rng  = np.random.default_rng(42)
        nW1  = rng.standard_normal((4, 3)).astype(np.float32)
        nW2  = rng.standard_normal((2, 4)).astype(np.float32)
        nx   = np.ones((3, 1), dtype=np.float32)

        l1 = vf.Dense(3, 4, activation=vf.nn.relu)
        l2 = vf.Dense(4, 2)
        l1.weights.assign(t(nW1))
        l2.weights.assign(t(nW2))

        x   = vf.placeholder([3, 1])
        out = l2(l1(x))
        sess = vf.Session()
        result   = sess.run(out, feed_dict={x: t(nx)}).to_numpy()
        expected = nW2 @ np.maximum(0, nW1 @ nx)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_gradient_flows_through_dense(self):
        """vf.gradients should return a tensor of correct shape for Dense.weights."""
        l = vf.Dense(3, 2)
        x = vf.placeholder([3, 1])
        y = l(x)
        [dW] = vf.gradients(y, [l.weights])
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={x: t(np.ones((3, 1), dtype=np.float32))})
        assert dW_val.shape == (2, 3)


# ══════════════════════════════════════════════════════════════════════════════
# vf.layers alias
# ══════════════════════════════════════════════════════════════════════════════

class TestLayersAlias:
    def test_layers_dense_is_dense(self):
        assert vf.layers.Dense is vf.Dense

    def test_layers_dense_construction(self):
        layer = vf.layers.Dense(4, 8, activation=vf.nn.relu)
        assert layer.input_dim == 4
        assert layer.units == 8
