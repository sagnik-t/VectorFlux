"""
tests/test_api.py — T12: Clean Python API

Tests cover:
  - vf.placeholder() convenience wrapper
  - vf.Variable class: init, assign, Node interface passthrough,
    accepts numpy arrays in assign()
  - Overloaded ops (add, mul, relu, matmul): symbolic path with Nodes,
    eager path with Tensors
  - vf.nn.relu graph op
  - Session accepts Variable objects in both fetches and feed_dict keys;
    feed_dict override of a Variable does not persist
  - vf.gradients accepts Variable objects in xs
  - Integration: two-layer graph built entirely with the clean API
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


# ══════════════════════════════════════════════════════════════════════════════
# placeholder
# ══════════════════════════════════════════════════════════════════════════════

class TestPlaceholder:
    def test_returns_placeholder_node(self):
        ph = vf.placeholder([3])
        assert ph.type == "Placeholder"

    def test_name_propagated(self):
        ph = vf.placeholder([3], name="x")
        assert ph.name == "x"

    def test_fed_via_session(self):
        ph = vf.placeholder([3])
        sess = vf.Session()
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = sess.run(ph, feed_dict={ph: t(na)})
        np.testing.assert_array_equal(result.to_numpy(), na)


# ══════════════════════════════════════════════════════════════════════════════
# Variable
# ══════════════════════════════════════════════════════════════════════════════

class TestVariable:
    def test_type(self):
        W = vf.Variable(t(np.zeros(3)))
        assert W.type == "Variable"

    def test_name_user(self):
        W = vf.Variable(t(np.zeros(3)), name="W")
        assert W.name == "W"

    def test_reads_initial_value(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W = vf.Variable(t(na))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(W).to_numpy(), na)

    def test_assign_method(self):
        W = vf.Variable(t(np.zeros(3)))
        W.assign(t(np.array([4.0, 5.0, 6.0])))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(W).to_numpy(), [4.0, 5.0, 6.0])

    def test_assign_accepts_numpy_array(self):
        """Variable.assign() coerces a raw numpy array to Tensor."""
        W = vf.Variable(t(np.zeros(3)))
        W.assign(np.array([7.0, 8.0, 9.0], dtype=np.float32))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(W).to_numpy(), [7.0, 8.0, 9.0])

    def test_no_inputs(self):
        W = vf.Variable(t(np.zeros(3)))
        assert len(W.inputs) == 0

    def test_used_directly_in_matmul(self):
        """Variable can be passed directly to vf.matmul without .node."""
        nW = np.eye(2, dtype=np.float32)
        W = vf.Variable(t(nW))
        x = vf.make_const(t(np.ones((2, 1), dtype=np.float32)))
        y = vf.matmul(W, x)
        sess = vf.Session()
        np.testing.assert_allclose(sess.run(y).to_numpy(), np.ones((2, 1)))

    def test_repr_contains_name(self):
        W = vf.Variable(t(np.zeros(2)), name="W")
        assert "W" in repr(W)


# ══════════════════════════════════════════════════════════════════════════════
# Overloaded ops — symbolic path (inputs are Nodes / Variables)
# ══════════════════════════════════════════════════════════════════════════════

class TestSymbolicOps:
    def test_add_with_nodes_returns_node(self):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        assert vf.add(a, b).type == "Add"

    def test_mul_with_nodes_returns_node(self):
        a = vf.make_const(t(np.array([2.0])))
        b = vf.make_const(t(np.array([3.0])))
        assert vf.mul(a, b).type == "Mul"

    def test_relu_with_node_returns_node(self):
        a = vf.make_const(t(np.array([-1.0, 2.0])))
        assert vf.relu(a).type == "Relu"

    def test_matmul_with_nodes_returns_node(self):
        a = vf.make_const(t(np.zeros((2, 3), dtype=np.float32)))
        b = vf.make_const(t(np.zeros((3, 4), dtype=np.float32)))
        assert vf.matmul(a, b).type == "MatMul"

    def test_add_symbolic_correct_value(self):
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([3.0, 4.0])))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(vf.add(a, b)).to_numpy(), [4.0, 6.0])

    def test_matmul_symbolic_correct_value(self):
        na = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(np.eye(2, dtype=np.float32)))
        sess = vf.Session()
        np.testing.assert_allclose(sess.run(vf.matmul(a, b)).to_numpy(), na)

    def test_add_with_variable(self):
        """Overloaded add dispatches correctly when one argument is a Variable."""
        W = vf.Variable(t(np.array([1.0, 2.0])))
        x = vf.make_const(t(np.array([3.0, 4.0])))
        y = vf.add(W, x)
        assert y.type == "Add"
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(y).to_numpy(), [4.0, 6.0])

    def test_relu_with_variable(self):
        W = vf.Variable(t(np.array([-1.0, 2.0])))
        r = vf.relu(W)
        assert r.type == "Relu"
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(r).to_numpy(), [0.0, 2.0])


# ══════════════════════════════════════════════════════════════════════════════
# Overloaded ops — eager path (inputs are Tensors)
# ══════════════════════════════════════════════════════════════════════════════

class TestEagerOps:
    def test_add_with_tensors_returns_tensor(self):
        result = vf.add(t(np.array([1.0, 2.0])), t(np.array([3.0, 4.0])))
        assert isinstance(result, vf.Tensor)
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 6.0])

    def test_mul_with_tensors(self):
        result = vf.mul(t(np.array([2.0, 3.0])), t(np.array([4.0, 5.0])))
        assert isinstance(result, vf.Tensor)
        np.testing.assert_array_equal(result.to_numpy(), [8.0, 15.0])

    def test_relu_with_tensor(self):
        result = vf.relu(t(np.array([-1.0, 0.0, 2.0])))
        assert isinstance(result, vf.Tensor)
        np.testing.assert_array_equal(result.to_numpy(), [0.0, 0.0, 2.0])

    def test_matmul_with_tensors(self):
        nb = np.array([[1.0], [2.0]], dtype=np.float32)
        result = vf.matmul(t(np.eye(2, dtype=np.float32)), t(nb))
        assert isinstance(result, vf.Tensor)
        np.testing.assert_allclose(result.to_numpy(), nb)


# ══════════════════════════════════════════════════════════════════════════════
# vf.nn
# ══════════════════════════════════════════════════════════════════════════════

class TestNn:
    def test_nn_relu_returns_node(self):
        a = vf.make_const(t(np.array([-1.0, 2.0])))
        assert vf.nn.relu(a).type == "Relu"

    def test_nn_relu_correct_value(self):
        a = vf.make_const(t(np.array([-1.0, 0.0, 2.0])))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(vf.nn.relu(a)).to_numpy(), [0.0, 0.0, 2.0])

    def test_nn_relu_with_variable(self):
        W = vf.Variable(t(np.array([-1.0, 2.0])))
        r = vf.nn.relu(W)
        assert r.type == "Relu"
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(r).to_numpy(), [0.0, 2.0])


# ══════════════════════════════════════════════════════════════════════════════
# Session + Variable unwrapping
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionApi:
    def test_run_variable_as_fetch(self):
        """sess.run(Variable) works directly — no .node needed."""
        W = vf.Variable(t(np.array([1.0, 2.0, 3.0])))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(W).to_numpy(), [1.0, 2.0, 3.0])

    def test_run_variable_in_list(self):
        """sess.run([Variable, Node]) works."""
        W = vf.Variable(t(np.array([1.0, 2.0])))
        x = vf.make_const(t(np.array([3.0, 4.0])))
        sess = vf.Session()
        results = sess.run([W, x])
        np.testing.assert_array_equal(results[0].to_numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(results[1].to_numpy(), [3.0, 4.0])

    def test_feed_dict_with_variable_key(self):
        """feed_dict accepts a Variable as key, overriding its value for that run."""
        W = vf.Variable(t(np.zeros(2)))
        x = vf.make_const(t(np.array([1.0, 1.0])))
        y = vf.add(W, x)
        sess = vf.Session()
        result = sess.run(y, feed_dict={W: t(np.array([5.0, 5.0]))})
        np.testing.assert_array_equal(result.to_numpy(), [6.0, 6.0])

    def test_feed_dict_variable_override_does_not_persist(self):
        """A feed_dict override of a Variable must not change its stored value."""
        W = vf.Variable(t(np.array([1.0, 2.0])))
        sess = vf.Session()
        sess.run(W, feed_dict={W: t(np.array([99.0, 99.0]))})
        np.testing.assert_array_equal(sess.run(W).to_numpy(), [1.0, 2.0])

    def test_existing_node_feed_dict_still_works(self):
        """Passing a plain NodeRef in feed_dict is unaffected."""
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([0.0, 0.0])))
        c = vf.add(a, b)
        sess = vf.Session()
        result = sess.run(c, feed_dict={a: t(np.array([10.0, 20.0]))})
        np.testing.assert_array_equal(result.to_numpy(), [10.0, 20.0])


# ══════════════════════════════════════════════════════════════════════════════
# gradients with Variable
# ══════════════════════════════════════════════════════════════════════════════

class TestGradientsApi:
    def test_gradients_with_variable_in_xs(self):
        """vf.gradients accepts Variable objects in xs."""
        nW = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        nx = np.ones((2, 1), dtype=np.float32)
        W = vf.Variable(t(nW))
        x = vf.placeholder([2, 1])
        y = vf.matmul(W, x)
        [dW] = vf.gradients(y, [W])
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={x: t(nx)})
        assert dW_val.shape == (2, 2)

    def test_gradients_with_mixed_xs(self):
        """vf.gradients accepts a mix of Variable and plain NodeRef in xs."""
        nW = np.eye(2, dtype=np.float32)
        W = vf.Variable(t(nW))
        x = vf.placeholder([2, 1])
        y = vf.matmul(W, x)
        [dW] = vf.gradients(y, [W])   # Variable
        sess = vf.Session()
        dW_val = sess.run(dW, feed_dict={x: t(np.ones((2, 1), dtype=np.float32))})
        assert dW_val.shape == (2, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_two_layer_graph(self):
        """y = W2 @ relu(W1 @ x) — built entirely with the clean API."""
        rng = np.random.default_rng(0)
        nW1 = rng.standard_normal((4, 3)).astype(np.float32)
        nW2 = rng.standard_normal((2, 4)).astype(np.float32)
        nx  = np.ones((3, 1), dtype=np.float32)

        W1 = vf.Variable(t(nW1))
        W2 = vf.Variable(t(nW2))
        x  = vf.placeholder([3, 1], name="x")

        h = vf.nn.relu(vf.matmul(W1, x))
        y = vf.matmul(W2, h)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        result = sess.run(y, feed_dict={x: t(nx)})
        expected = nW2 @ np.maximum(0, nW1 @ nx)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)

    def test_sgd_step_with_clean_api(self):
        """One SGD step using Variable.assign() and vf.gradients."""
        nW = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        nx = np.ones((2, 1), dtype=np.float32)
        lr = 0.1

        W = vf.Variable(t(nW), name="W")
        x = vf.placeholder([2, 1], name="x")
        y = vf.matmul(W, x)
        [dW] = vf.gradients(y, [W])

        sess = vf.Session()
        sess.run(vf.global_variables_initializer())

        dW_val = sess.run(dW, feed_dict={x: t(nx)}).to_numpy()
        W_val  = sess.run(W).to_numpy()
        W.assign(W_val - lr * dW_val)

        np.testing.assert_allclose(sess.run(W).to_numpy(), nW - lr * dW_val, atol=1e-5)
