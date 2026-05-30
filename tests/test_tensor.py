import numpy as np
import pytest
import vectorflux as vf


# ── Construction ───────────────────────────────────────────────────────────────

def test_shape_metadata():
    t = vf.Tensor([2, 3])
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.numel() == 6
    assert t.device == "cpu"


def test_zero_init():
    t = vf.Tensor([3])
    np.testing.assert_array_equal(t.to_numpy(), np.zeros(3, dtype=np.float32))


def test_from_numpy_1d():
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    t = vf.Tensor(a)
    assert t.shape == (4,)
    np.testing.assert_array_equal(t.to_numpy(), a)


def test_from_numpy_2d():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = vf.Tensor(a)
    assert t.shape == (2, 3)
    np.testing.assert_array_equal(t.to_numpy(), a)


def test_from_numpy_forcecast_float64():
    # forcecast + c_style should silently convert float64 → float32
    a = np.arange(4, dtype=np.float64).reshape(2, 2)
    t = vf.Tensor(a)
    np.testing.assert_array_almost_equal(t.to_numpy(), a.astype(np.float32))


def test_from_numpy_non_contiguous():
    # Transposed array is non-contiguous; c_style flag should force a copy
    a = np.arange(6, dtype=np.float32).reshape(2, 3).T   # shape (3, 2), F-order
    t = vf.Tensor(a)
    assert t.shape == (3, 2)
    np.testing.assert_array_equal(t.to_numpy(), np.ascontiguousarray(a))


# ── Strides ────────────────────────────────────────────────────────────────────

def test_strides_row_major():
    t = vf.Tensor([2, 3, 4])
    assert t.strides == (12, 4, 1)


def test_strides_1d():
    t = vf.Tensor([5])
    assert t.strides == (1,)


# ── Element access ─────────────────────────────────────────────────────────────

def test_at_read():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = vf.Tensor(a)
    assert t.at([0, 0]) == pytest.approx(0.0)
    assert t.at([0, 2]) == pytest.approx(2.0)
    assert t.at([1, 2]) == pytest.approx(5.0)


def test_at_write():
    t = vf.Tensor([2, 3])
    t.set([1, 1], 42.0)
    assert t.at([1, 1]) == pytest.approx(42.0)
    assert t.at([0, 0]) == pytest.approx(0.0)   # others untouched


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_scalar_tensor():
    t = vf.Tensor([])
    assert t.ndim == 0
    assert t.numel() == 1


def test_out_of_range_raises():
    t = vf.Tensor([2, 3])
    with pytest.raises(Exception):
        t.at([0, 5])


def test_rank_mismatch_raises():
    t = vf.Tensor([2, 3])
    with pytest.raises(Exception):
        t.at([0])            # rank 1 index into rank-2 tensor


# ── Repr ───────────────────────────────────────────────────────────────────────

def test_repr_contains_shape():
    t = vf.Tensor([2, 3])
    r = repr(t)
    assert "shape=[2, 3]" in r
    assert "device=cpu" in r


def test_repr_shows_data():
    a = np.array([1, 2, 3], dtype=np.float32)
    t = vf.Tensor(a)
    r = repr(t)
    assert "1" in r and "2" in r and "3" in r


# ── NumPy round-trip ──────────────────────────────────────────────────────────

def test_to_numpy_is_copy():
    a = np.array([1, 2, 3], dtype=np.float32)
    t = vf.Tensor(a)
    out = t.to_numpy()
    out[0] = 99.0           # mutating the numpy output must not affect the tensor
    assert t.at([0]) == pytest.approx(1.0)
