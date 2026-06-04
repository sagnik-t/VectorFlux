"""
tests/test_graph.py — T08: Op base class + DAG graph nodes
Tests verify graph structure only — no execution (that's T09/Session.run()).
"""

import numpy as np
import pytest
import vectorflux as vf

# ── Helpers ────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# Node type names
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeTypes:
    def test_const_type(self, t):
        n = vf.make_const(t(np.array([1.0])))
        assert n.type == "Const"

    def test_add_type(self, t):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        assert vf.make_add(a, b).type == "Add"

    def test_mul_type(self, t):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        assert vf.make_mul(a, b).type == "Mul"

    def test_relu_type(self, t):
        a = vf.make_const(t(np.array([1.0])))
        assert vf.make_relu(a).type == "Relu"

    def test_matmul_type(self, t):
        a = vf.make_const(t(np.zeros((2, 3), dtype=np.float32)))
        b = vf.make_const(t(np.zeros((3, 4), dtype=np.float32)))
        assert vf.make_matmul(a, b).type == "MatMul"

# ══════════════════════════════════════════════════════════════════════════════
# Input connections
# ══════════════════════════════════════════════════════════════════════════════

class TestInputConnections:
    def test_const_has_no_inputs(self, t):
        n = vf.make_const(t(np.array([1.0])))
        assert len(n.inputs) == 0

    def test_add_has_two_inputs(self, t):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        c = vf.make_add(a, b)
        assert len(c.inputs) == 2

    def test_relu_has_one_input(self, t):
        a = vf.make_const(t(np.array([1.0])))
        r = vf.make_relu(a)
        assert len(r.inputs) == 1

    def test_input_identity_preserved(self, t):
        """inputs[i] must be the exact same Python object we passed in."""
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        c = vf.make_add(a, b)
        assert c.inputs[0] is a
        assert c.inputs[1] is b

    def test_input_order_preserved(self, t):
        """First input stays first, second stays second."""
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        c = vf.make_add(a, b)
        assert c.inputs[0].name == a.name
        assert c.inputs[1].name == b.name

# ══════════════════════════════════════════════════════════════════════════════
# Node names
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeNames:
    def test_auto_name_contains_type(self, t):
        n = vf.make_add(
            vf.make_const(t(np.array([1.0]))),
            vf.make_const(t(np.array([2.0]))),
        )
        assert "Add" in n.name

    def test_user_name_preserved(self, t):
        n = vf.make_const(t(np.array([1.0])), name="my_const")
        assert n.name == "my_const"

    def test_auto_names_are_unique(self, t):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        assert a.name != b.name

# ══════════════════════════════════════════════════════════════════════════════
# Evaluated flag
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluatedFlag:
    def test_new_node_not_evaluated(self, t):
        n = vf.make_const(t(np.array([1.0])))
        assert not n.evaluated

    def test_op_nodes_not_evaluated(self, t):
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_const(t(np.array([2.0])))
        c = vf.make_add(a, b)
        assert not c.evaluated

# ══════════════════════════════════════════════════════════════════════════════
# Graph node count and reset
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphStructure:
    def test_single_const_registered(self, t):
        # fixture cleared the graph; one make_const → one node
        vf.make_const(t(np.array([1.0])))
        # we can't directly query graph size from Python yet,
        # but we can verify the node came back
        n = vf.make_const(t(np.array([1.0])))
        assert n is not None

    def test_reset_clears_graph(self, t):
        vf.make_const(t(np.array([1.0])))
        vf.make_const(t(np.array([2.0])))
        vf.reset_default_graph()
        # After reset, auto-names restart from 0
        n = vf.make_const(t(np.array([3.0])))
        assert "0" in n.name   # first node after reset gets id 0

    def test_repr_contains_type_and_name(self, t):
        n = vf.make_const(t(np.array([1.0])), name="x")
        r = repr(n)
        assert "Const" in r
        assert "x" in r

# ══════════════════════════════════════════════════════════════════════════════
# DAG topology
# ══════════════════════════════════════════════════════════════════════════════

class TestDAGTopology:
    def test_linear_chain(self, t):
        """A → B → C (chain of three nodes)."""
        a = vf.make_const(t(np.array([-1.0, 2.0])))
        b = vf.make_relu(a)
        c = vf.make_relu(b)
        assert c.inputs[0] is b
        assert b.inputs[0] is a

    def test_diamond(self, t):
        """A feeds both B and C; B and C both feed D."""
        a = vf.make_const(t(np.array([1.0, 2.0])))
        b = vf.make_relu(a)
        c = vf.make_relu(a)
        d = vf.make_add(b, c)
        assert d.inputs[0] is b
        assert d.inputs[1] is c
        assert b.inputs[0] is a
        assert c.inputs[0] is a

    def test_shared_input_same_object(self, t):
        """A node used as input to two different ops is the same object."""
        a = vf.make_const(t(np.array([1.0])))
        b = vf.make_relu(a)
        c = vf.make_relu(a)
        assert b.inputs[0] is c.inputs[0]

    def test_deep_chain(self, t):
        """Build a chain of 10 relu nodes without error."""
        node = vf.make_const(t(np.array([1.0])))
        for _ in range(10):
            node = vf.make_relu(node)
        assert node.type == "Relu"
        assert len(node.inputs) == 1
