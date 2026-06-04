"""
tests/test_ops_cuda.py — T06/T07: Element-wise CUDA ops + cuBLAS matmul
Tests for vf.add, vf.mul, vf.relu, vf.matmul on CUDA tensors.
All numerical results are validated against the CPU path / NumPy.
"""

import numpy as np
import pytest
import vectorflux as vf

# ── Helpers ────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# add
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaAdd:
    def test_output_device_is_cuda(self, ct, cpu_t, to_np):
        a = ct(np.array([1.0, 2.0, 3.0]))
        b = ct(np.array([4.0, 5.0, 6.0]))
        assert vf.add(a, b).device == "cuda"

    def test_1d_matches_numpy(self, ct, cpu_t, to_np):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        nb = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(to_np(vf.add(ct(na), ct(nb))), na + nb)

    def test_2d_random(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(0)
        na = rng.standard_normal((8, 16)).astype(np.float32)
        nb = rng.standard_normal((8, 16)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.add(ct(na), ct(nb))), na + nb, atol=1e-6)

    def test_large_tensor(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(1)
        na = rng.standard_normal((512, 512)).astype(np.float32)
        nb = rng.standard_normal((512, 512)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.add(ct(na), ct(nb))), na + nb, atol=1e-5)

    def test_shape_preserved(self, ct, cpu_t, to_np):
        na = np.zeros((3, 4, 5), dtype=np.float32)
        nb = np.zeros((3, 4, 5), dtype=np.float32)
        assert vf.add(ct(na), ct(nb)).shape == (3, 4, 5)

    def test_shape_mismatch_raises(self, ct, cpu_t, to_np):
        a = vf.Tensor([2, 3]).to("cuda")
        b = vf.Tensor([2, 4]).to("cuda")
        with pytest.raises(Exception):
            vf.add(a, b)

    def test_cross_device_raises(self, ct, cpu_t, to_np):
        a = ct(np.array([1.0, 2.0]))
        b = cpu_t(np.array([1.0, 2.0]))
        with pytest.raises(Exception):
            vf.add(a, b)

# ══════════════════════════════════════════════════════════════════════════════
# mul
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaMul:
    def test_output_device_is_cuda(self, ct, cpu_t, to_np):
        a = ct(np.array([1.0, 2.0]))
        b = ct(np.array([3.0, 4.0]))
        assert vf.mul(a, b).device == "cuda"

    def test_1d_matches_numpy(self, ct, cpu_t, to_np):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        nb = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(to_np(vf.mul(ct(na), ct(nb))), na * nb)

    def test_2d_random(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(2)
        na = rng.standard_normal((8, 16)).astype(np.float32)
        nb = rng.standard_normal((8, 16)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.mul(ct(na), ct(nb))), na * nb, atol=1e-6)

    def test_large_tensor(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(3)
        na = rng.standard_normal((512, 512)).astype(np.float32)
        nb = rng.standard_normal((512, 512)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.mul(ct(na), ct(nb))), na * nb, atol=1e-5)

    def test_shape_preserved(self, ct, cpu_t, to_np):
        na = np.ones((4, 5), dtype=np.float32)
        nb = np.ones((4, 5), dtype=np.float32)
        assert vf.mul(ct(na), ct(nb)).shape == (4, 5)

    def test_cross_device_raises(self, ct, cpu_t, to_np):
        a = ct(np.array([1.0, 2.0]))
        b = cpu_t(np.array([1.0, 2.0]))
        with pytest.raises(Exception):
            vf.mul(a, b)

# ══════════════════════════════════════════════════════════════════════════════
# relu
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaRelu:
    def test_output_device_is_cuda(self, ct, cpu_t, to_np):
        a = ct(np.array([-1.0, 2.0, -3.0]))
        assert vf.relu(a).device == "cuda"

    def test_mixed_signs(self, ct, cpu_t, to_np):
        na = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(to_np(vf.relu(ct(na))), [0, 0, 0, 1, 3])

    def test_all_negative_gives_zeros(self, ct, cpu_t, to_np):
        na = np.array([-5.0, -2.0, -0.1], dtype=np.float32)
        np.testing.assert_array_equal(to_np(vf.relu(ct(na))), np.zeros(3))

    def test_all_positive_unchanged(self, ct, cpu_t, to_np):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(to_np(vf.relu(ct(na))), na)

    def test_2d_matches_numpy(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(4)
        na = rng.standard_normal((8, 16)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.relu(ct(na))), np.maximum(0, na))

    def test_large_tensor(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(5)
        na = rng.standard_normal((512, 512)).astype(np.float32)
        np.testing.assert_allclose(to_np(vf.relu(ct(na))), np.maximum(0, na))

    def test_shape_preserved(self, ct, cpu_t, to_np):
        na = np.zeros((3, 4), dtype=np.float32)
        assert vf.relu(ct(na)).shape == (3, 4)

# ══════════════════════════════════════════════════════════════════════════════
# matmul — cuBLAS (T07)
# ══════════════════════════════════════════════════════════════════════════════

class TestCudaMatmul:
    def test_output_device_is_cuda(self, ct, cpu_t, to_np):
        a = ct(np.ones((2, 3), dtype=np.float32))
        b = ct(np.ones((3, 4), dtype=np.float32))
        assert vf.matmul(a, b).device == "cuda"

    def test_output_shape(self, ct, cpu_t, to_np):
        a = ct(np.ones((2, 3), dtype=np.float32))
        b = ct(np.ones((3, 4), dtype=np.float32))
        assert vf.matmul(a, b).shape == (2, 4)

    def test_basic_2x3_3x2(self, ct, cpu_t, to_np):
        na = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype=np.float32)
        nb = np.array([[7,  8],
                       [9,  10],
                       [11, 12]], dtype=np.float32)
        out = to_np(vf.matmul(ct(na), ct(nb)))
        np.testing.assert_allclose(out, na @ nb, atol=1e-4)

    def test_identity(self, ct, cpu_t, to_np):
        na = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        eye = np.eye(3, dtype=np.float32)
        out = to_np(vf.matmul(ct(na), ct(eye)))
        np.testing.assert_allclose(out, na, atol=1e-5)

    def test_square_random(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(6)
        na = rng.standard_normal((4, 4)).astype(np.float32)
        nb = rng.standard_normal((4, 4)).astype(np.float32)
        np.testing.assert_allclose(
            to_np(vf.matmul(ct(na), ct(nb))), na @ nb, atol=1e-4)

    def test_non_square(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(7)
        na = rng.standard_normal((3, 7)).astype(np.float32)
        nb = rng.standard_normal((7, 5)).astype(np.float32)
        out = to_np(vf.matmul(ct(na), ct(nb)))
        assert out.shape == (3, 5)
        np.testing.assert_allclose(out, na @ nb, atol=1e-4)

    def test_large(self, ct, cpu_t, to_np):
        rng = np.random.default_rng(8)
        na = rng.standard_normal((128, 256)).astype(np.float32)
        nb = rng.standard_normal((256, 64)).astype(np.float32)
        np.testing.assert_allclose(
            to_np(vf.matmul(ct(na), ct(nb))), na @ nb, atol=1e-2)

    def test_matches_cpu(self, ct, cpu_t, to_np):
        """CUDA result must agree with the CPU naive matmul."""
        rng = np.random.default_rng(9)
        na = rng.standard_normal((16, 32)).astype(np.float32)
        nb = rng.standard_normal((32, 8)).astype(np.float32)
        cpu_out = vf.matmul(cpu_t(na), cpu_t(nb)).to_numpy()
        cuda_out = to_np(vf.matmul(ct(na), ct(nb)))
        np.testing.assert_allclose(cuda_out, cpu_out, atol=1e-4)

    def test_vector_dot_product(self, ct, cpu_t, to_np):
        """[1, N] @ [N, 1] → [1, 1]"""
        na = np.array([[1, 2, 3]], dtype=np.float32)
        nb = np.array([[1], [2], [3]], dtype=np.float32)
        out = to_np(vf.matmul(ct(na), ct(nb)))
        assert out.shape == (1, 1)
        assert out[0, 0] == pytest.approx(14.0, abs=1e-4)

    def test_inner_dim_mismatch_raises(self, ct, cpu_t, to_np):
        with pytest.raises(Exception, match="inner dimensions|dimension"):
            vf.matmul(vf.Tensor([2, 3]).to("cuda"), vf.Tensor([4, 2]).to("cuda"))

    def test_1d_input_raises(self, ct, cpu_t, to_np):
        with pytest.raises(Exception, match="2-D"):
            vf.matmul(vf.Tensor([4]).to("cuda"), vf.Tensor([4]).to("cuda"))

    def test_cross_device_raises(self, ct, cpu_t, to_np):
        a = ct(np.ones((2, 3), dtype=np.float32))
        b = cpu_t(np.ones((3, 2), dtype=np.float32))
        with pytest.raises(Exception):
            vf.matmul(a, b)
