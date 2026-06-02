"""
tests/test_variables.py — T11: Placeholder + Variable

Tests cover:
  - Placeholder: node type/name, fed via feed_dict, raises without feed,
    different values across runs, used in downstream computation
  - Variable: node type/name, reads initial value, assign() updates value,
    update persists across runs, 2D shapes, used in downstream computation,
    error when variable_assign called on a non-Variable
  - global_variables_initializer: resets single and multiple variables,
    snapshot semantics (vars created after the call are not included)
  - Integration: linear layer with Placeholder input + Variable weights,
    one manual SGD step using vf.gradients + variable_assign
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
# Placeholder
# ══════════════════════════════════════════════════════════════════════════════

class TestPlaceholder:
    def test_type(self):
        ph = vf.make_placeholder([3])
        assert ph.type == "Placeholder"

    def test_name_auto(self):
        ph = vf.make_placeholder([3])
        assert "Placeholder" in ph.name

    def test_name_user(self):
        ph = vf.make_placeholder([3], name="x")
        assert ph.name == "x"

    def test_no_inputs(self):
        ph = vf.make_placeholder([3])
        assert len(ph.inputs) == 0

    def test_feed_1d(self):
        ph = vf.make_placeholder([3])
        sess = vf.Session()
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = sess.run(ph, feed_dict={ph: t(na)})
        np.testing.assert_array_equal(result.to_numpy(), na)

    def test_feed_2d(self):
        ph = vf.make_placeholder([2, 3])
        sess = vf.Session()
        na = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = sess.run(ph, feed_dict={ph: t(na)})
        np.testing.assert_array_equal(result.to_numpy(), na)

    def test_raises_without_feed(self):
        ph = vf.make_placeholder([3])
        sess = vf.Session()
        with pytest.raises(Exception):
            sess.run(ph)

    def test_feed_changes_across_runs(self):
        ph = vf.make_placeholder([2])
        sess = vf.Session()
        r1 = sess.run(ph, feed_dict={ph: t(np.array([1.0, 2.0]))})
        r2 = sess.run(ph, feed_dict={ph: t(np.array([10.0, 20.0]))})
        np.testing.assert_array_equal(r1.to_numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(r2.to_numpy(), [10.0, 20.0])

    def test_feed_does_not_mutate_node(self):
        """A second run without feed_dict must still raise (not use stale value)."""
        ph = vf.make_placeholder([2])
        sess = vf.Session()
        sess.run(ph, feed_dict={ph: t(np.array([1.0, 2.0]))})
        with pytest.raises(Exception):
            sess.run(ph)

    def test_used_in_downstream_matmul(self):
        """ph feeds into a matmul with a const weight."""
        W = vf.make_const(t(np.eye(3, dtype=np.float32)))
        x = vf.make_placeholder([3, 1], name="x")
        y = vf.make_matmul(W, x)
        sess = vf.Session()
        nx = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        result = sess.run(y, feed_dict={x: t(nx)})
        np.testing.assert_allclose(result.to_numpy(), nx)

    def test_two_placeholders_fed_independently(self):
        pa = vf.make_placeholder([2], name="a")
        pb = vf.make_placeholder([2], name="b")
        out = vf.make_add(pa, pb)
        sess = vf.Session()
        result = sess.run(out, feed_dict={
            pa: t(np.array([1.0, 2.0])),
            pb: t(np.array([3.0, 4.0])),
        })
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 6.0])


# ══════════════════════════════════════════════════════════════════════════════
# Variable
# ══════════════════════════════════════════════════════════════════════════════

class TestVariable:
    def test_type(self):
        v = vf.make_variable(t(np.zeros(3)))
        assert v.type == "Variable"

    def test_name_auto(self):
        v = vf.make_variable(t(np.zeros(3)))
        assert "Variable" in v.name

    def test_name_user(self):
        v = vf.make_variable(t(np.zeros(3)), name="W")
        assert v.name == "W"

    def test_no_inputs(self):
        v = vf.make_variable(t(np.zeros(3)))
        assert len(v.inputs) == 0

    def test_reads_initial_value(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = vf.make_variable(t(na))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(v).to_numpy(), na)

    def test_assign_updates_value(self):
        v = vf.make_variable(t(np.zeros(3)))
        new_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        vf.variable_assign(v, t(new_val))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(v).to_numpy(), new_val)

    def test_assign_persists_across_runs(self):
        """After assign, every subsequent run returns the new value."""
        v = vf.make_variable(t(np.zeros(2)))
        vf.variable_assign(v, t(np.array([7.0, 8.0])))
        sess = vf.Session()
        r1 = sess.run(v).to_numpy().copy()
        r2 = sess.run(v).to_numpy().copy()
        np.testing.assert_array_equal(r1, [7.0, 8.0])
        np.testing.assert_array_equal(r2, [7.0, 8.0])

    def test_assign_multiple_times(self):
        v = vf.make_variable(t(np.zeros(2)))
        sess = vf.Session()
        for val in [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]:
            vf.variable_assign(v, t(np.array(val, dtype=np.float32)))
            np.testing.assert_array_equal(sess.run(v).to_numpy(), val)

    def test_2d_variable(self):
        na = np.eye(3, dtype=np.float32)
        v = vf.make_variable(t(na))
        sess = vf.Session()
        np.testing.assert_array_equal(sess.run(v).to_numpy(), na)

    def test_variable_in_matmul(self):
        nW = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        W = vf.make_variable(t(nW))
        x = vf.make_const(t(np.array([[1.0], [1.0]], dtype=np.float32)))
        y = vf.make_matmul(W, x)
        sess = vf.Session()
        np.testing.assert_allclose(
            sess.run(y).to_numpy(), nW @ np.array([[1.0], [1.0]]))

    def test_assign_updates_downstream_computation(self):
        """Downstream graph sees new value after assign."""
        W = vf.make_variable(t(np.zeros((2, 2))))
        x = vf.make_const(t(np.ones((2, 1), dtype=np.float32)))
        y = vf.make_matmul(W, x)
        sess = vf.Session()
        # Before assign: zero weight → zero output
        np.testing.assert_array_equal(sess.run(y).to_numpy(), np.zeros((2, 1)))
        # After assign: identity weight → ones output
        vf.variable_assign(W, t(np.eye(2, dtype=np.float32)))
        np.testing.assert_array_equal(sess.run(y).to_numpy(), np.ones((2, 1)))

    def test_variable_assign_on_non_variable_raises(self):
        c = vf.make_const(t(np.zeros(3)))
        with pytest.raises(Exception, match="Variable|not a Variable"):
            vf.variable_assign(c, t(np.ones(3)))


# ══════════════════════════════════════════════════════════════════════════════
# global_variables_initializer
# ══════════════════════════════════════════════════════════════════════════════

class TestGlobalVariablesInitializer:
    def test_returns_a_node(self):
        init = vf.global_variables_initializer()
        assert init is not None
        assert init.type == "InitVariables"

    def test_runs_without_error_no_variables(self):
        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)   # must not raise even with an empty variable list

    def test_runs_without_error_with_variable(self):
        vf.make_variable(t(np.zeros(3)))
        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

    def test_resets_single_variable(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = vf.make_variable(t(na))
        init = vf.global_variables_initializer()
        sess = vf.Session()
        # Mutate the variable
        vf.variable_assign(v, t(np.zeros(3)))
        np.testing.assert_array_equal(sess.run(v).to_numpy(), np.zeros(3))
        # Re-initialize — must restore original value
        sess.run(init)
        np.testing.assert_array_equal(sess.run(v).to_numpy(), na)

    def test_resets_multiple_variables(self):
        na = np.array([1.0, 2.0], dtype=np.float32)
        nb = np.array([3.0, 4.0], dtype=np.float32)
        va = vf.make_variable(t(na), name="va")
        vb = vf.make_variable(t(nb), name="vb")
        init = vf.global_variables_initializer()
        sess = vf.Session()
        vf.variable_assign(va, t(np.zeros(2)))
        vf.variable_assign(vb, t(np.zeros(2)))
        sess.run(init)
        np.testing.assert_array_equal(sess.run(va).to_numpy(), na)
        np.testing.assert_array_equal(sess.run(vb).to_numpy(), nb)

    def test_snapshot_semantics(self):
        """Variables created AFTER global_variables_initializer() are NOT reset."""
        na = np.array([1.0, 2.0], dtype=np.float32)
        nb = np.array([5.0, 6.0], dtype=np.float32)
        va = vf.make_variable(t(na))
        init = vf.global_variables_initializer()   # snapshot: {va}
        vb = vf.make_variable(t(nb))               # created after snapshot
        sess = vf.Session()
        vf.variable_assign(va, t(np.zeros(2)))
        vf.variable_assign(vb, t(np.zeros(2)))
        sess.run(init)
        np.testing.assert_array_equal(sess.run(va).to_numpy(), na)       # reset
        np.testing.assert_array_equal(sess.run(vb).to_numpy(), np.zeros(2))  # not reset

    def test_init_can_be_run_multiple_times(self):
        na = np.array([9.0, 8.0], dtype=np.float32)
        v = vf.make_variable(t(na))
        init = vf.global_variables_initializer()
        sess = vf.Session()
        for _ in range(3):
            vf.variable_assign(v, t(np.zeros(2)))
            sess.run(init)
            np.testing.assert_array_equal(sess.run(v).to_numpy(), na)


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Placeholder + Variable together
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_linear_layer_forward(self):
        """y = W @ x + b  with W=Variable, b=Variable, x=Placeholder."""
        nW = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        nb = np.array([[0.5], [0.5]], dtype=np.float32)
        nx = np.array([[2.0], [3.0]], dtype=np.float32)

        W = vf.make_variable(t(nW), name="W")
        b = vf.make_variable(t(nb), name="b")
        x = vf.make_placeholder([2, 1], name="x")
        y = vf.make_add(vf.make_matmul(W, x), b)

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        result = sess.run(y, feed_dict={x: t(nx)})
        np.testing.assert_allclose(result.to_numpy(), nW @ nx + nb, atol=1e-5)

    def test_linear_layer_different_batches(self):
        """Same graph, different feed_dict inputs give correct outputs."""
        nW = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        W = vf.make_variable(t(nW))
        x = vf.make_placeholder([2, 1])
        y = vf.make_matmul(W, x)
        sess = vf.Session()

        for nx in [np.array([[1.0], [1.0]]),
                   np.array([[2.0], [3.0]]),
                   np.array([[-1.0], [4.0]])]:
            nx = nx.astype(np.float32)
            result = sess.run(y, feed_dict={x: t(nx)})
            np.testing.assert_allclose(result.to_numpy(), nW @ nx, atol=1e-5)

    def test_manual_sgd_step(self):
        """One manual SGD step: W ← W − lr * dL/dW."""
        nW = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        nx = np.array([[1.0], [1.0]], dtype=np.float32)
        lr = 0.1

        W = vf.make_variable(t(nW), name="W")
        x = vf.make_placeholder([2, 1], name="x")
        y = vf.make_matmul(W, x)
        [dW] = vf.gradients(y, [W])

        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        # Compute gradient
        dW_val = sess.run(dW, feed_dict={x: t(nx)}).to_numpy()
        # Read current weight
        W_val = sess.run(W).to_numpy()
        # Apply update
        new_W = W_val - lr * dW_val
        vf.variable_assign(W, t(new_W))
        # Verify
        np.testing.assert_allclose(
            sess.run(W).to_numpy(), nW - lr * dW_val, atol=1e-5)

    def test_init_restores_after_sgd(self):
        """global_variables_initializer undoes the SGD update."""
        nW = np.ones((2, 2), dtype=np.float32)
        W = vf.make_variable(t(nW))
        x = vf.make_placeholder([2, 1])
        y = vf.make_matmul(W, x)
        [dW] = vf.gradients(y, [W])
        init = vf.global_variables_initializer()
        sess = vf.Session()
        sess.run(init)

        # Run an SGD step
        dW_val = sess.run(dW, feed_dict={x: t(np.ones((2, 1), dtype=np.float32))})
        vf.variable_assign(W, t(sess.run(W).to_numpy() - 0.1 * dW_val.to_numpy()))

        # Re-init must restore original weights
        sess.run(init)
        np.testing.assert_allclose(sess.run(W).to_numpy(), nW, atol=1e-5)
