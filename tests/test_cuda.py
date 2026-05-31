"""
tests/test_cuda.py — T05: CUDA Tensor backend
Tests for .to('cuda') / .to('cpu') transfers and device-aware metadata.
"""

import numpy as np
import pytest
import vectorflux as vf


# ── Helpers ────────────────────────────────────────────────────────────────────

def cpu_tensor(arr):
    return vf.Tensor(arr.astype(np.float32))


# ── Device property ────────────────────────────────────────────────────────────

class TestDeviceProperty:
    def test_default_is_cpu(self):
        t = vf.Tensor([2, 3])
        assert t.device == "cpu"

    def test_to_cuda_device_string(self):
        t = cpu_tensor(np.array([1.0, 2.0, 3.0]))
        g = t.to("cuda")
        assert g.device == "cuda"

    def test_to_cpu_device_string(self):
        t = cpu_tensor(np.array([1.0]))
        assert t.to("cuda").to("cpu").device == "cpu"


# ── Shape / metadata preserved through transfer ────────────────────────────────

class TestMetadataPreserved:
    def test_shape_1d(self):
        t = cpu_tensor(np.arange(4, dtype=np.float32))
        assert t.to("cuda").shape == (4,)

    def test_shape_2d(self):
        t = cpu_tensor(np.zeros((3, 5), dtype=np.float32))
        assert t.to("cuda").shape == (3, 5)

    def test_numel(self):
        t = cpu_tensor(np.zeros((2, 3, 4), dtype=np.float32))
        assert t.to("cuda").numel() == 24

    def test_ndim(self):
        t = cpu_tensor(np.zeros((2, 3), dtype=np.float32))
        assert t.to("cuda").ndim == 2

    def test_strides_unchanged(self):
        t = cpu_tensor(np.zeros((2, 3, 4), dtype=np.float32))
        assert t.to("cuda").strides == (12, 4, 1)


# ── Data round-trips ───────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_1d_values(self):
        na = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cpu_tensor(na).to("cuda").to("cpu").to_numpy()
        np.testing.assert_array_equal(result, na)

    def test_2d_values(self):
        na = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = cpu_tensor(na).to("cuda").to("cpu").to_numpy()
        np.testing.assert_array_equal(result, na)

    def test_large_tensor(self):
        rng = np.random.default_rng(42)
        na = rng.standard_normal((128, 128)).astype(np.float32)
        result = cpu_tensor(na).to("cuda").to("cpu").to_numpy()
        np.testing.assert_allclose(result, na)

    def test_zeros_roundtrip(self):
        na = np.zeros((4, 4), dtype=np.float32)
        result = cpu_tensor(na).to("cuda").to("cpu").to_numpy()
        np.testing.assert_array_equal(result, na)

    def test_negative_values(self):
        na = np.array([-1.0, -2.0, 0.0, 3.14], dtype=np.float32)
        result = cpu_tensor(na).to("cuda").to("cpu").to_numpy()
        np.testing.assert_allclose(result, na)


# ── No-op transfers ────────────────────────────────────────────────────────────

class TestNoOp:
    def test_cpu_to_cpu_is_copy(self):
        na = np.array([1.0, 2.0], dtype=np.float32)
        t = cpu_tensor(na)
        t2 = t.to("cpu")
        assert t2.device == "cpu"
        np.testing.assert_array_equal(t2.to_numpy(), na)

    def test_cuda_to_cuda_is_copy(self):
        na = np.array([1.0, 2.0], dtype=np.float32)
        g = cpu_tensor(na).to("cuda")
        g2 = g.to("cuda")
        assert g2.device == "cuda"
        np.testing.assert_array_equal(g2.to("cpu").to_numpy(), na)


# ── Repr ───────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_cuda_repr_shows_device(self):
        t = cpu_tensor(np.array([1.0])).to("cuda")
        assert "cuda" in repr(t)

    def test_cuda_repr_no_data_values(self):
        # CUDA tensors must NOT try to print from device memory
        t = cpu_tensor(np.array([1.0, 2.0, 3.0])).to("cuda")
        r = repr(t)
        assert "on cuda" in r

    def test_cpu_repr_unchanged(self):
        t = cpu_tensor(np.array([1.0, 2.0]))
        r = repr(t)
        assert "device=cpu" in r


# ── Error cases ────────────────────────────────────────────────────────────────

class TestErrors:
    def test_to_numpy_on_cuda_raises(self):
        t = cpu_tensor(np.array([1.0])).to("cuda")
        with pytest.raises(Exception, match="cpu"):
            t.to_numpy()

    def test_at_on_cuda_raises(self):
        t = cpu_tensor(np.array([1.0, 2.0])).to("cuda")
        with pytest.raises(Exception):
            t.at([0])

    def test_invalid_device_raises(self):
        t = cpu_tensor(np.array([1.0]))
        with pytest.raises(Exception):
            t.to("tpu")
