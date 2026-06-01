"""
tests/test_session.py — T09: Forward pass / Session.run()

Tests cover:
  - Basic single-node and chained evaluation
  - Multiple fetch nodes
  - feed_dict override (single and multiple nodes)
  - Topological correctness (diamond DAG, shared intermediates)
  - Repeatability (run() can be called multiple times)
  - Only the required subgraph is evaluated
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
# Basic evaluation
# ══════════════════════════════════════════════════════════════════════════════

class TestBasicRun:
    def test_const_node(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        node = vf.make_const(t(na))
        sess = vf.Session()
        result = sess.run(node)
        np.testing.assert_array_equal(result.to_numpy(), na)

    def test_add_two_consts(self):
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([3.0, 4.0])))
        c = vf.make_add(a, b)
        sess = vf.Session()
        result = sess.run(c)
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 6.0])

    def test_mul_two_consts(self):
        a = vf.make_const(t(np.array([2.0, 3.0])))
        b = vf.make_const(t(np.array([4.0, 5.0])))
        c = vf.make_mul(a, b)
        sess = vf.Session()
        result = sess.run(c)
        np.testing.assert_array_equal(result.to_numpy(), [8.0, 15.0])

    def test_relu_node(self):
        a = vf.make_const(t(np.array([-1.0, 0.0, 2.0])))
        b = vf.make_relu(a)
        sess = vf.Session()
        result = sess.run(b)
        np.testing.assert_array_equal(result.to_numpy(), [0.0, 0.0, 2.0])

    def test_matmul_node(self):
        na = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        nb = np.eye(2, dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        c = vf.make_matmul(a, b)
        sess = vf.Session()
        result = sess.run(c)
        np.testing.assert_allclose(result.to_numpy(), na @ nb, atol=1e-6)

    def test_chained_ops(self):
        """matmul → relu → add chain."""
        na = np.array([[1.0, -1.0], [-2.0, 3.0]], dtype=np.float32)
        nb = np.eye(2, dtype=np.float32)
        nc = np.ones((2, 2), dtype=np.float32)
        w = vf.make_const(t(na))
        x = vf.make_const(t(nb))
        bias = vf.make_const(t(nc))
        mm   = vf.make_matmul(w, x)
        act  = vf.make_relu(mm)
        out  = vf.make_add(act, bias)
        sess = vf.Session()
        result = sess.run(out)
        expected = np.maximum(0, na @ nb) + nc
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)

    def test_2d_shapes_preserved(self):
        na = np.arange(6, dtype=np.float32).reshape(2, 3)
        nb = np.zeros((2, 3), dtype=np.float32)
        a = vf.make_const(t(na))
        b = vf.make_const(t(nb))
        c = vf.make_add(a, b)
        sess = vf.Session()
        assert sess.run(c).shape == (2, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Multiple fetches
# ══════════════════════════════════════════════════════════════════════════════

class TestMultipleFetches:
    def test_two_independent_nodes(self):
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([3.0, 4.0])))
        sess = vf.Session()
        results = sess.run([a, b])
        assert len(results) == 2
        np.testing.assert_array_equal(results[0].to_numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(results[1].to_numpy(), [3.0, 4.0])

    def test_fetch_input_and_output(self):
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([3.0, 4.0])))
        c = vf.make_add(a, b)
        sess = vf.Session()
        results = sess.run([a, c])
        np.testing.assert_array_equal(results[0].to_numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(results[1].to_numpy(), [4.0, 6.0])

    def test_fetch_order_preserved(self):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        c = vf.make_const(t(np.array([3.0])))
        sess = vf.Session()
        results = sess.run([c, a, b])
        assert results[0].to_numpy()[0] == pytest.approx(3.0)
        assert results[1].to_numpy()[0] == pytest.approx(1.0)
        assert results[2].to_numpy()[0] == pytest.approx(2.0)

    def test_same_node_fetched_twice(self):
        a = vf.make_const(t(np.array([5.0])))
        sess = vf.Session()
        results = sess.run([a, a])
        assert len(results) == 2
        assert results[0].to_numpy()[0] == pytest.approx(5.0)
        assert results[1].to_numpy()[0] == pytest.approx(5.0)


# ══════════════════════════════════════════════════════════════════════════════
# feed_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedDict:
    def test_override_single_const(self):
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([3.0, 4.0])))
        c = vf.make_add(a, b)
        sess = vf.Session()
        result = sess.run(c, feed_dict={a: t(np.array([10.0, 20.0]))})
        np.testing.assert_array_equal(result.to_numpy(), [13.0, 24.0])

    def test_override_both_inputs(self):
        a = vf.make_const(t(np.zeros(2)))
        b = vf.make_const(t(np.zeros(2)))
        c = vf.make_add(a, b)
        sess = vf.Session()
        result = sess.run(c, feed_dict={
            a: t(np.array([1.0, 2.0])),
            b: t(np.array([3.0, 4.0])),
        })
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 6.0])

    def test_empty_feed_dict(self):
        a = vf.make_const(t(np.array([5.0])))
        sess = vf.Session()
        result = sess.run(a, feed_dict={})
        assert result.to_numpy()[0] == pytest.approx(5.0)

    def test_feed_does_not_mutate_node(self):
        """Running with feed_dict must not permanently change the const value."""
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([3.0, 4.0])))
        c = vf.make_add(a, b)
        sess = vf.Session()
        sess.run(c, feed_dict={a: t(np.array([100.0, 200.0]))})
        # Second run without feed_dict must use the original value.
        result = sess.run(c)
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 6.0])

    def test_feed_different_values_across_runs(self):
        a = vf.make_const(t(np.zeros(2)))
        b = vf.make_const(t(np.array([1.0, 1.0])))
        c = vf.make_add(a, b)
        sess = vf.Session()
        r1 = sess.run(c, feed_dict={a: t(np.array([1.0, 2.0]))})
        r2 = sess.run(c, feed_dict={a: t(np.array([10.0, 20.0]))})
        np.testing.assert_array_equal(r1.to_numpy(), [2.0, 3.0])
        np.testing.assert_array_equal(r2.to_numpy(), [11.0, 21.0])


# ══════════════════════════════════════════════════════════════════════════════
# Topological correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestTopologicalCorrectness:
    def test_diamond_dag(self):
        """Shared input → two relu branches → add.  Topo sort must not duplicate work."""
        a = vf.make_const(t(np.array([2.0, -3.0])))
        b = vf.make_relu(a)     # [2, 0]
        c = vf.make_relu(a)     # [2, 0]
        d = vf.make_add(b, c)  # [4, 0]
        sess = vf.Session()
        result = sess.run(d)
        np.testing.assert_array_equal(result.to_numpy(), [4.0, 0.0])

    def test_shared_intermediate(self):
        """s = a + b; p = a * b; out = s + p"""
        na = np.array([1.0, 2.0], dtype=np.float32)
        nb = np.array([3.0, 4.0], dtype=np.float32)
        a   = vf.make_const(t(na))
        b   = vf.make_const(t(nb))
        s   = vf.make_add(a, b)
        p   = vf.make_mul(a, b)
        out = vf.make_add(s, p)
        sess = vf.Session()
        result = sess.run(out)
        np.testing.assert_array_equal(result.to_numpy(), (na + nb) + (na * nb))

    def test_deep_chain(self):
        """10 stacked relu nodes should all evaluate correctly."""
        node = vf.make_const(t(np.array([1.0, -2.0, 3.0])))
        for _ in range(10):
            node = vf.make_relu(node)
        sess = vf.Session()
        result = sess.run(node)
        np.testing.assert_array_equal(result.to_numpy(), [1.0, 0.0, 3.0])

    def test_evaluated_flag_set_after_run(self):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_relu(a)
        sess = vf.Session()
        assert not a.evaluated
        assert not b.evaluated
        sess.run(b)
        assert a.evaluated
        assert b.evaluated

    def test_unreachable_node_not_evaluated(self):
        """A node not in the fetch's dependency graph must stay unevaluated."""
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        c = vf.make_add(a, b)
        # d shares input a but is not in c's subgraph
        d = vf.make_relu(a)
        sess = vf.Session()
        sess.run(c)
        # c, a, b were evaluated; d was not fetched and must remain unevaluated
        assert not d.evaluated


# ══════════════════════════════════════════════════════════════════════════════
# Repeatability
# ══════════════════════════════════════════════════════════════════════════════

class TestRepeatability:
    def test_same_result_on_repeated_runs(self):
        a = vf.make_const(t(np.array([1.0, 2.0, 3.0])))
        b = vf.make_relu(a)
        sess = vf.Session()
        r1 = sess.run(b).to_numpy().copy()
        r2 = sess.run(b).to_numpy().copy()
        np.testing.assert_array_equal(r1, r2)

    def test_multiple_sessions_same_graph(self):
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_const(t(np.array([3.0, 4.0])))
        c = vf.make_add(a, b)
        s1 = vf.Session()
        s2 = vf.Session()
        r1 = s1.run(c).to_numpy()
        r2 = s2.run(c).to_numpy()
        np.testing.assert_array_equal(r1, r2)

    def test_interleaved_runs(self):
        """Run session A, then session B on same graph, then A again — stable."""
        a = vf.make_const(t(np.array([5.0])))
        b = vf.make_relu(a)
        c = vf.make_relu(b)
        sess = vf.Session()
        r1 = sess.run(b).to_numpy()[0]
        r2 = sess.run(c).to_numpy()[0]
        r3 = sess.run(b).to_numpy()[0]
        assert r1 == pytest.approx(5.0)
        assert r2 == pytest.approx(5.0)
        assert r3 == pytest.approx(5.0)
