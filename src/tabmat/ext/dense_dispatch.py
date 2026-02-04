"""Dense operations with automatic backend dispatch.

This module provides the same API as the C++ dense module but automatically
dispatches to either the C++ or Rust backend based on the current setting.
"""

import os
from typing import Literal

import numpy as np

# Global backend setting
_backend_env = os.environ.get("TABMAT_BACKEND", "cpp").lower()
_BACKEND: Literal["cpp", "rust"] = "rust" if _backend_env == "rust" else "cpp"


def set_backend(backend: Literal["cpp", "rust"]):
    """Set the backend for dense matrix operations."""
    global _BACKEND
    if backend not in ("cpp", "rust"):
        raise ValueError(f"Unknown backend: {backend}. Use 'cpp' or 'rust'.")
    _BACKEND = backend


def get_backend_name() -> str:
    """Get the name of the current backend."""
    return _BACKEND


# Lazy-loaded backend modules
_cpp_module = None
_rust_module = None


def _get_cpp():
    global _cpp_module
    if _cpp_module is None:
        from tabmat.ext import dense as _cpp_module  # type: ignore[attr-defined]
    return _cpp_module


def _get_rust():
    global _rust_module
    if _rust_module is None:
        from tabmat.tabmat_rust_ext import tabmat_rust_ext as _rust_module
    return _rust_module


def dense_sandwich(X, d, rows, cols, thresh1d=32, kratio=16, innerblock=128):
    """Dense sandwich product: X.T @ diag(d) @ X."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = X.dtype

        if orig_dtype != np.float64:
            X = X.astype(np.float64)
            d = d.astype(np.float64)

        result = rust.dense_sandwich(X, d, rows, cols)

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().dense_sandwich(X, d, rows, cols, thresh1d, kratio, innerblock)


def dense_rmatvec(X, v, rows, cols):
    """Dense transpose matrix-vector multiplication: X.T @ v."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = X.dtype

        if orig_dtype != np.float64:
            X = X.astype(np.float64)
            v = v.astype(np.float64)

        result = rust.dense_rmatvec(X, v, rows, cols)

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().dense_rmatvec(X, v, rows, cols)


def dense_matvec(X, v, rows, cols):
    """Dense matrix-vector multiplication: X @ v."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = X.dtype

        if orig_dtype != np.float64:
            X = X.astype(np.float64)
            v = v.astype(np.float64)

        result = rust.dense_matvec(X, v, rows, cols)

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().dense_matvec(X, v, rows, cols)


def transpose_square_dot_weights(X, weights, shift):
    """Compute weighted squared column norms with shift for dense matrix."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = X.dtype

        if orig_dtype != np.float64:
            X = X.astype(np.float64)
            weights = weights.astype(np.float64)
            shift = shift.astype(np.float64)

        result = rust.dense_transpose_square_dot_weights(X, weights, shift)

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().transpose_square_dot_weights(X, weights, shift)
