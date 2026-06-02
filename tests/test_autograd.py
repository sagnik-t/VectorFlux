"""
tests/test_autograd.py — T10: Reverse-mode autograd

Tests cover:
  - step, transpose, oneslike primitive ops
  - Analytical gradients for add, mul, relu, matmul
  - Numerical gradient checks (finite differences) for all ops
  - Gradient accumulation through a diamond DAG
  - Chained gradient through a multi-op graph
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


def numerical_gradient(sess, out_node, in_node, x_np, eps=1e-3):
    """
    Finite-difference estimate of  d(sum(out)) / d(in),
    matching what vf.gradients() computes (sum-of-output convention).
    """
    grad = np.zeros_like(x_np, dtype=np.float32)
    flat = x_np.ravel().copy()
    for i in range(len(flat)):
        x_plus = flat.copy(); x_plus[i] += eps
        x_minus = flat.copy(); x_minus[i] -= eps

        f_plus = sess.run(out_node,
                          feed_dict={in_node: t(x_plus.reshape(x_np.shape))
                                     }).to_numpy().sum()
        f_minus = sess.run(out_node,
                           feed_dict={in_node: t(x_minus.reshape(x_np.shape))
                                      }).to_numpy().sum()

        grad.ravel()[i] = (f_plus - f_minus) / (2.0 * eps)
    return grad


# ══════════════════════════════════════════════════════════════════════════════
# Primitive op correctness: step, transpose, oneslike
# ══════════════════════════════════════════════════════════════════════════════

class TestPrimitiveOps:
    # ── step ──────────────────────────────────────────────────────────────────

    def test_step_positive(self):
        a = vf.make_const(t(np.array([1.0, 2.0, 3.0])))
        s = vf.make_step(a)
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(s).to_numpy(), [1, 1, 1])

    def test_step_negative(self):
        a = vf.make_const(t(np.array([-1.0, -2.0, -3.0])))
        s = vf.make_step(a)
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(s).to_numpy(), [0, 0, 0])

    def test_step_mixed(self):
        # step(0) = 0  by subgradient convention
        a = vf.make_const(t(np.array([-1.0, 0.0, 1.0])))
        s = vf.make_step(a)
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(s).to_numpy(), [0, 0, 1])

    def test_step_2d(self):
        na = np.array([[-3.0, 2.0], [0.0, 5.0]], dtype=np.float32)
        a = vf.make_const(t(na))
        s = vf.make_step(a)
        sess = vf.Session()
        expected = (na > 0).astype(np.float32)
        np.testing.assert_array_equal(sess.run(s).to_numpy(), expected)

    # ── transpose ─────────────────────────────────────────────────────────────

    def test_transpose_basic(self):
        na = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)   # [2, 3]
        a = vf.make_const(t(na))
        r = vf.make_transpose(a)
        sess = vf.Session()
        out = sess.run(r)
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(out.to_numpy(), na.T)

    def test_transpose_square(self):
        na = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        a = vf.make_const(t(na))
        r = vf.make_transpose(a)
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(r).to_numpy(), na.T)

    def test_transpose_double_is_identity(self):
        na = np.arange(6, dtype=np.float32).reshape(2, 3)
        a = vf.make_const(t(na))
        r = vf.make_transpose(vf.make_transpose(a))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(r).to_numpy(), na)

    # ── oneslike ──────────────────────────────────────────────────────────────

    def test_oneslike_shape_1d(self):
        a = vf.make_const(t(np.zeros(4)))
        o = vf.make_oneslike(a)
        sess = vf.Session()
        out = sess.run(o)
        assert out.shape == (4,)
        np.testing.assert_array_equal(out.to_numpy(), np.ones(4))

    def test_oneslike_shape_2d(self):
        a = vf.make_const(t(np.zeros((2, 3))))
        o = vf.make_oneslike(a)
        sess = vf.Session()
        out = sess.run(o)
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out.to_numpy(), np.ones((2, 3)))


# ══════════════════════════════════════════════════════════════════════════════
# Analytical gradients
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyticalGradients:
    def test_grad_add_both_inputs(self):
        """d(a+b)/da = ones, d(a+b)/db = ones."""
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        nb = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_add(a, b)
        da, db = vf.gradients(out, [a, b])
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(da).to_numpy(), np.ones(3))
        np.testing.assert_array_equal(sess.run(db).to_numpy(), np.ones(3))

    def test_grad_mul_a(self):
        """d(a*b)/da = b."""
        na = np.array([2.0, 3.0], dtype=np.float32)
        nb = np.array([4.0, 5.0], dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_mul(a, b)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(da).to_numpy(), nb)

    def test_grad_mul_b(self):
        """d(a*b)/db = a."""
        na = np.array([2.0, 3.0], dtype=np.float32)
        nb = np.array([4.0, 5.0], dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_mul(a, b)
        [db] = vf.gradients(out, [b])
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(db).to_numpy(), na)

    def test_grad_relu_positive(self):
        """relu gradient is 1 where input > 0."""
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = vf.make_const(t(na))
        out = vf.make_relu(a)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(da).to_numpy(), np.ones(3))

    def test_grad_relu_negative(self):
        """relu gradient is 0 where input < 0."""
        na = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        a = vf.make_const(t(na))
        out = vf.make_relu(a)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(da).to_numpy(), np.zeros(3))

    def test_grad_relu_mixed(self):
        """relu gradient matches step(a) element-wise."""
        na = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
        a = vf.make_const(t(na))
        out = vf.make_relu(a)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        expected = (na > 0).astype(np.float32)
        np.testing.assert_array_equal(sess.run(da).to_numpy(), expected)

    def test_grad_matmul_a_shape(self):
        """dL/dA has same shape as A."""
        na = np.ones((2, 3), dtype=np.float32)
        nb = np.ones((3, 4), dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_matmul(a, b)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        assert sess.run(da).shape == (2, 3)

    def test_grad_matmul_b_shape(self):
        """dL/dB has same shape as B."""
        na = np.ones((2, 3), dtype=np.float32)
        nb = np.ones((3, 4), dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_matmul(a, b)
        [db] = vf.gradients(out, [b])
        sess = vf.Session()
        assert sess.run(db).shape == (3, 4)

    def test_grad_matmul_a_value(self):
        """dL/dA = ones @ B^T when seeded with ones_like."""
        na = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)   # [2, 2]
        nb = np.eye(2, dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_matmul(a, b)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        # seed = ones[2,2]; dA = ones @ B^T = ones @ I = ones
        np.testing.assert_allclose(sess.run(da).to_numpy(),
                                   np.ones((2, 2)) @ nb.T, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# Numerical gradient checks
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalGradients:
    """
    Compare analytical gradients from vf.gradients() to finite differences.
    Test values deliberately avoid x = 0 so relu is differentiable.
    """

    def test_numerical_add_a(self):
        rng = np.random.default_rng(0)
        na = rng.uniform(1, 3, (3, 4)).astype(np.float32)
        nb = rng.uniform(1, 3, (3, 4)).astype(np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_add(a, b)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        # float32 finite differences accumulate ~3e-3 round-off over 12 elements
        np.testing.assert_allclose(
            sess.run(da).to_numpy(),
            numerical_gradient(sess, out, a, na),
            atol=5e-3)

    def test_numerical_mul_a(self):
        rng = np.random.default_rng(1)
        na = rng.uniform(1, 3, (3,)).astype(np.float32)
        nb = rng.uniform(1, 3, (3,)).astype(np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_mul(a, b)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(da).to_numpy(),
            numerical_gradient(sess, out, a, na),
            atol=2e-3)

    def test_numerical_mul_b(self):
        rng = np.random.default_rng(2)
        na = rng.uniform(1, 3, (4,)).astype(np.float32)
        nb = rng.uniform(1, 3, (4,)).astype(np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_mul(a, b)
        [db] = vf.gradients(out, [b])
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(db).to_numpy(),
            numerical_gradient(sess, out, b, nb),
            atol=2e-3)

    def test_numerical_relu(self):
        # All positive values to keep gradient = 1 everywhere
        rng = np.random.default_rng(3)
        na = rng.uniform(0.5, 3, (5,)).astype(np.float32)
        a = vf.make_const(t(na))
        out = vf.make_relu(a)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(da).to_numpy(),
            numerical_gradient(sess, out, a, na),
            atol=1e-4)

    def test_numerical_matmul_a(self):
        rng = np.random.default_rng(4)
        na = rng.standard_normal((2, 3)).astype(np.float32)
        nb = rng.standard_normal((3, 4)).astype(np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_matmul(a, b)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(da).to_numpy(),
            numerical_gradient(sess, out, a, na),
            atol=1e-3)

    def test_numerical_matmul_b(self):
        rng = np.random.default_rng(5)
        na = rng.standard_normal((2, 3)).astype(np.float32)
        nb = rng.standard_normal((3, 4)).astype(np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        out = vf.make_matmul(a, b)
        [db] = vf.gradients(out, [b])
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(db).to_numpy(),
            numerical_gradient(sess, out, b, nb),
            atol=1e-3)

    def test_numerical_chain_matmul_relu(self):
        """Gradient flows through matmul → relu → add chain."""
        rng = np.random.default_rng(6)
        nW = rng.standard_normal((3, 4)).astype(np.float32)
        nx = rng.uniform(0.5, 2, (4, 2)).astype(np.float32)
        nb = np.ones((3, 2), dtype=np.float32)
        W = vf.make_const(t(nW))
        x = vf.make_const(t(nx))
        bias = vf.make_const(t(nb))
        out = vf.make_add(vf.make_relu(vf.make_matmul(W, x)), bias)
        [dW] = vf.gradients(out, [W])
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(dW).to_numpy(),
            numerical_gradient(sess, out, W, nW),
            atol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Gradient accumulation
# ══════════════════════════════════════════════════════════════════════════════

class TestGradientAccumulation:
    def test_diamond_add(self):
        """a is used twice in b + c → gradient should be 2 * ones."""
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_relu(a)
        c = vf.make_relu(a)
        out = vf.make_add(b, c)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        # d(relu(a) + relu(a))/da = step(a) + step(a) = 2*ones (since na > 0)
        np.testing.assert_array_equal(
            sess.run(da).to_numpy(),
            np.full(3, 2.0, dtype=np.float32))

    def test_diamond_numerical(self):
        """Diamond DAG accumulation matches finite differences."""
        rng = np.random.default_rng(7)
        na = rng.uniform(0.5, 2, (4,)).astype(np.float32)
        a = vf.make_const(t(na))
        b = vf.make_relu(a)
        c = vf.make_mul(a, b)   # a * relu(a)
        out = vf.make_add(b, c)  # relu(a) + a*relu(a)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(da).to_numpy(),
            numerical_gradient(sess, out, a, na),
            atol=1e-3)

    def test_mul_same_input(self):
        """a * a — gradient should be 2*a."""
        na = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        a = vf.make_const(t(na))
        out = vf.make_mul(a, a)
        [da] = vf.gradients(out, [a])
        sess = vf.Session()
        # d(a*a)/da = a + a = 2*a
        np.testing.assert_allclose(
            sess.run(da).to_numpy(),
            2.0 * na, atol=1e-5)
