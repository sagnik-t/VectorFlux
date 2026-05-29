"""
tests/test_ops_cuda.py — T06: Element-wise CUDA ops
Tests for vf.add, vf.mul, vf.relu on CUDA tensors.
All numerical results are validated against the CPU path / NumPy.
"""

import numpy as np
import pytest
import vectorflux as vf


# ── Helpers ────────────────────────────────────────────────────────────────────

def ct(arr):
    """numpy → vf.Tensor on CUDA."""
    return vf.Tensor(arr.astype(np.float32)).to("cuda")

def cpu_t(arr):
    """numpy → vf.Tensor on CPU."""
    return vf.Tensor(arr.astype(np.float32))

def to_np(t):
    """CUDA tensor → numpy (round-trip through CPU)."""
    return t.to("cpu").to_numpy()


# ══════════════════════════════════════════════════════════════════════════════
# add
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaAdd:
    def test_output_device_is_cuda(self):
        a = ct(np.array([1.0, 2.0, 3.0]))
        b = ct(np.array([4.0, 5.0, 6.0]))
        assert vf.add(a, b).device == "cuda"

    def test_1d_matches_numpy(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        nb = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(to_np(vf.add(ct(na), ct(nb))), na + nb)

    def test_2d_random(self):
        rng = np.random.default_rng(0)
        na = rng.standard_normal((8, 16)).astype(np.float32)
        nb = rng.standard_normal((8, 16)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.add(ct(na), ct(nb))), na + nb, atol=1e-6)

    def test_large_tensor(self):
        rng = np.random.default_rng(1)
        na = rng.standard_normal((512, 512)).astype(np.float32)
        nb = rng.standard_normal((512, 512)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.add(ct(na), ct(nb))), na + nb, atol=1e-5)

    def test_shape_preserved(self):
        na = np.zeros((3, 4, 5), dtype=np.float32)
        nb = np.zeros((3, 4, 5), dtype=np.float32)
        assert vf.add(ct(na), ct(nb)).shape == (3, 4, 5)

    def test_shape_mismatch_raises(self):
        a = vf.Tensor([2, 3]).to("cuda")
        b = vf.Tensor([2, 4]).to("cuda")
        with pytest.raises(Exception):
            vf.add(a, b)

    def test_cross_device_raises(self):
        """One CPU tensor + one CUDA tensor must raise."""
        a = ct(np.array([1.0, 2.0]))
        b = cpu_t(np.array([1.0, 2.0]))
        with pytest.raises(Exception):
            vf.add(a, b)


# ══════════════════════════════════════════════════════════════════════════════
# mul
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaMul:
    def test_output_device_is_cuda(self):
        a = ct(np.array([1.0, 2.0]))
        b = ct(np.array([3.0, 4.0]))
        assert vf.mul(a, b).device == "cuda"

    def test_1d_matches_numpy(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        nb = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(to_np(vf.mul(ct(na), ct(nb))), na * nb)

    def test_2d_random(self):
        rng = np.random.default_rng(2)
        na = rng.standard_normal((8, 16)).astype(np.float32)
        nb = rng.standard_normal((8, 16)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.mul(ct(na), ct(nb))), na * nb, atol=1e-6)

    def test_large_tensor(self):
        rng = np.random.default_rng(3)
        na = rng.standard_normal((512, 512)).astype(np.float32)
        nb = rng.standard_normal((512, 512)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.mul(ct(na), ct(nb))), na * nb, atol=1e-5)

    def test_shape_preserved(self):
        na = np.ones((4, 5), dtype=np.float32)
        nb = np.ones((4, 5), dtype=np.float32)
        assert vf.mul(ct(na), ct(nb)).shape == (4, 5)

    def test_cross_device_raises(self):
        a = ct(np.array([1.0, 2.0]))
        b = cpu_t(np.array([1.0, 2.0]))
        with pytest.raises(Exception):
            vf.mul(a, b)


# ══════════════════════════════════════════════════════════════════════════════
# relu
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaRelu:
    def test_output_device_is_cuda(self):
        a = ct(np.array([-1.0, 2.0, -3.0]))
        assert vf.relu(a).device == "cuda"

    def test_mixed_signs(self):
        na = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(to_np(vf.relu(ct(na))), [0, 0, 0, 1, 3])

    def test_all_negative_gives_zeros(self):
        na = np.array([-5.0, -2.0, -0.1], dtype=np.float32)
        np.testing.assert_array_equal(to_np(vf.relu(ct(na))), np.zeros(3))

    def test_all_positive_unchanged(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(to_np(vf.relu(ct(na))), na)

    def test_2d_matches_numpy(self):
        rng = np.random.default_rng(4)
        na = rng.standard_normal((8, 16)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.relu(ct(na))), np.maximum(0, na))

    def test_large_tensor(self):
        rng = np.random.default_rng(5)
        na = rng.standard_normal((512, 512)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.relu(ct(na))), np.maximum(0, na))

    def test_shape_preserved(self):
        na = np.zeros((3, 4), dtype=np.float32)
        assert vf.relu(ct(na)).shape == (3, 4)


# ══════════════════════════════════════════════════════════════════════════════
# matmul — must still raise on CUDA until T07
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaMatmulNotYet:
    def test_matmul_cuda_raises(self):
        a = vf.Tensor([2, 3]).to("cuda")
        b = vf.Tensor([3, 2]).to("cuda")
        with pytest.raises(Exception, match="T07"):
            vf.matmul(a, b)
