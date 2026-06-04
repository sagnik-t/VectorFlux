"""
tests/conftest.py — Shared pytest fixtures for the VectorFlux test suite.

Fixtures provided
-----------------
reset_graph  (autouse)  — wipes the C++ graph + variable registry before
                          and after every test; removes the need for a
                          per-file clean_graph fixture.

t            (factory)  — numpy array → CPU Tensor.
                          Usage: a = t(np.array([1.0, 2.0]))

ct           (factory)  — numpy array → CUDA Tensor.
                          Usage: a = ct(np.ones((3, 3)))

cpu_t        (factory)  — numpy array → CPU Tensor (explicit alias used in
                          CUDA test files alongside ct for clarity).

to_np        (factory)  — Tensor → numpy array (moves to CPU first).
                          Usage: arr = to_np(cuda_tensor)

Markers
-------
cuda  — mark a test as requiring a CUDA-capable GPU.
        Usage: @pytest.mark.cuda
"""

import numpy as np
import pytest
import vectorflux as vf


# ── Graph lifecycle ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_graph():
    """Wipe the default graph and variable registry around every test."""
    vf.reset_default_graph()
    yield
    vf.reset_default_graph()


# ── Tensor factories ──────────────────────────────────────────────────────────

@pytest.fixture
def t():
    """numpy array → CPU vf.Tensor (float32)."""
    def _make(arr, dtype=np.float32):
        return vf.Tensor(np.asarray(arr, dtype=dtype))
    return _make


@pytest.fixture
def ct():
    """numpy array → CUDA vf.Tensor (float32)."""
    def _make(arr):
        return vf.Tensor(arr.astype(np.float32)).to("cuda")
    return _make


@pytest.fixture
def cpu_t():
    """numpy array → CPU vf.Tensor (float32). Explicit alias for CUDA test files."""
    def _make(arr):
        return vf.Tensor(arr.astype(np.float32))
    return _make


@pytest.fixture
def to_np():
    """vf.Tensor → numpy array (moves to CPU first if needed)."""
    def _convert(tensor):
        return tensor.to("cpu").to_numpy()
    return _convert


# ── Custom markers ────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda: mark test as requiring a CUDA-capable GPU",
    )
