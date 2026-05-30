"""
tests/test_ops.py — T04: Basic CPU ops
Tests for vf.add, vf.mul, vf.relu, vf.matmul.
All numerical results are validated against NumPy.
"""

import numpy as np
import pytest
import vectorflux as vf


# ── Helpers ────────────────────────────────────────────────────────────────────

def t(arr):
    """Convenience: numpy array → vf.Tensor."""
    return vf.Tensor(arr.astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# add
# ══════════════════════════════════════════════════════════════════════════════

class TestAdd:
    def test_1d_basic(self):
        a = t(np.array([1, 2, 3]))
        b = t(np.array([4, 5, 6]))
        c = vf.add(a, b)
        np.testing.assert_array_equal(c.to_numpy(), [5, 7, 9])

    def test_2d(self):
        a = t(np.ones((2, 3)))
        b = t(np.full((2, 3), 2.0))
        c = vf.add(a, b)
        np.testing.assert_array_equal(c.to_numpy(), np.full((2, 3), 3.0))

    def test_result_shape(self):
        a = t(np.zeros((4, 5)))
        b = t(np.zeros((4, 5)))
        assert vf.add(a, b).shape == (4, 5)

    def test_result_is_new_tensor(self):
        """add must not mutate either input."""
        orig_a = np.array([1, 2, 3], dtype=np.float32)
        orig_b = np.array([4, 5, 6], dtype=np.float32)
        a = t(orig_a)
        b = t(orig_b)
        _ = vf.add(a, b)
        np.testing.assert_array_equal(a.to_numpy(), orig_a)
        np.testing.assert_array_equal(b.to_numpy(), orig_b)

    def test_vs_numpy(self):
        rng = np.random.default_rng(0)
        na = rng.standard_normal((8, 16)).astype(np.float32)
        nb = rng.standard_normal((8, 16)).astype(np.float32)
        np.testing.assert_allclose(vf.add(t(na), t(nb)).to_numpy(), na + nb)

    def test_shape_mismatch_raises(self):
        a = vf.Tensor([2, 3])
        b = vf.Tensor([2, 4])
        with pytest.raises(Exception, match="shape mismatch|Shape"):
            vf.add(a, b)

    def test_rank_mismatch_raises(self):
        a = vf.Tensor([6])
        b = vf.Tensor([2, 3])
        with pytest.raises(Exception):
            vf.add(a, b)


# ══════════════════════════════════════════════════════════════════════════════
# mul
# ══════════════════════════════════════════════════════════════════════════════

class TestMul:
    def test_1d_basic(self):
        a = t(np.array([1, 2, 3]))
        b = t(np.array([4, 5, 6]))
        c = vf.mul(a, b)
        np.testing.assert_array_equal(c.to_numpy(), [4, 10, 18])

    def test_2d(self):
        a = t(np.full((3, 3), 2.0))
        b = t(np.full((3, 3), 3.0))
        np.testing.assert_array_equal(vf.mul(a, b).to_numpy(),
                                      np.full((3, 3), 6.0))

    def test_mul_by_zero(self):
        a = t(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
        b = t(np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_array_equal(vf.mul(a, b).to_numpy(),
                                      np.zeros((2, 3)))

    def test_vs_numpy(self):
        rng = np.random.default_rng(1)
        na = rng.standard_normal((5, 7)).astype(np.float32)
        nb = rng.standard_normal((5, 7)).astype(np.float32)
        np.testing.assert_allclose(vf.mul(t(na), t(nb)).to_numpy(), na * nb)

    def test_shape_mismatch_raises(self):
        with pytest.raises(Exception):
            vf.mul(vf.Tensor([2, 3]), vf.Tensor([3, 2]))


# ══════════════════════════════════════════════════════════════════════════════
# relu
# ══════════════════════════════════════════════════════════════════════════════

class TestRelu:
    def test_mixed_signs(self):
        a = t(np.array([-3, -1, 0, 1, 3]))
        np.testing.assert_array_equal(vf.relu(a).to_numpy(),
                                      [0, 0, 0, 1, 3])

    def test_all_positive_unchanged(self):
        na = np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(vf.relu(t(na)).to_numpy(), na)

    def test_all_negative_zeros(self):
        na = np.array([-5, -2, -0.1], dtype=np.float32)
        np.testing.assert_array_equal(vf.relu(t(na)).to_numpy(),
                                      np.zeros(3, dtype=np.float32))

    def test_zero_boundary(self):
        """Exactly 0.0 should map to 0.0."""
        a = t(np.array([0.0]))
        assert vf.relu(a).to_numpy()[0] == pytest.approx(0.0)

    def test_2d_shape_preserved(self):
        na = np.arange(-4, 5, dtype=np.float32).reshape(3, 3)
        out = vf.relu(t(na))
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(out.to_numpy(), np.maximum(0, na))

    def test_vs_numpy(self):
        rng = np.random.default_rng(2)
        na = rng.standard_normal((10, 10)).astype(np.float32)
        np.testing.assert_allclose(vf.relu(t(na)).to_numpy(), np.maximum(0, na))

    def test_does_not_mutate_input(self):
        na = np.array([-1, 2, -3], dtype=np.float32)
        a = t(na)
        _ = vf.relu(a)
        np.testing.assert_array_equal(a.to_numpy(), na)


# ══════════════════════════════════════════════════════════════════════════════
# matmul
# ══════════════════════════════════════════════════════════════════════════════

class TestMatmul:
    def test_basic_2x3_3x2(self):
        na = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype=np.float32)       # [2, 3]
        nb = np.array([[7,  8],
                       [9,  10],
                       [11, 12]], dtype=np.float32)         # [3, 2]
        c = vf.matmul(t(na), t(nb))
        assert c.shape == (2, 2)
        np.testing.assert_allclose(c.to_numpy(), na @ nb)

    def test_identity(self):
        """Multiplying by the identity matrix is a no-op."""
        na = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        eye = np.eye(3, dtype=np.float32)
        np.testing.assert_allclose(vf.matmul(t(na), t(eye)).to_numpy(), na,
                                   atol=1e-6)

    def test_square_random(self):
        rng = np.random.default_rng(3)
        na = rng.standard_normal((4, 4)).astype(np.float32)
        nb = rng.standard_normal((4, 4)).astype(np.float32)
        np.testing.assert_allclose(
            vf.matmul(t(na), t(nb)).to_numpy(), na @ nb, atol=1e-5)

    def test_non_square(self):
        rng = np.random.default_rng(4)
        na = rng.standard_normal((3, 7)).astype(np.float32)
        nb = rng.standard_normal((7, 5)).astype(np.float32)
        out = vf.matmul(t(na), t(nb))
        assert out.shape == (3, 5)
        np.testing.assert_allclose(out.to_numpy(), na @ nb, atol=1e-5)

    def test_vector_dot_product(self):
        """[1, N] @ [N, 1]  → [1, 1] scalar-in-matrix."""
        na = np.array([[1, 2, 3]], dtype=np.float32)   # row vector
        nb = np.array([[1], [2], [3]], dtype=np.float32)  # col vector
        out = vf.matmul(t(na), t(nb))
        assert out.shape == (1, 1)
        assert out.to_numpy()[0, 0] == pytest.approx(14.0)

    def test_inner_dim_mismatch_raises(self):
        with pytest.raises(Exception, match="inner dimensions|dimension"):
            vf.matmul(vf.Tensor([2, 3]), vf.Tensor([4, 2]))

    def test_1d_input_raises(self):
        with pytest.raises(Exception, match="2-D"):
            vf.matmul(vf.Tensor([4]), vf.Tensor([4]))

    def test_3d_input_raises(self):
        with pytest.raises(Exception, match="2-D"):
            vf.matmul(vf.Tensor([2, 3, 4]), vf.Tensor([4, 2]))
