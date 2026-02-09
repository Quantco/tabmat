"""Categorical operations with automatic backend dispatch.

This module provides the same API as the C++ categorical module but automatically
dispatches to either the C++ or Rust backend based on the current setting.

Note: The Rust backend has simpler function signatures without row/column
restrictions. When restrictions are present, we fall back to the C++ backend.
"""

import os
from typing import Literal

import numpy as np

# Global backend setting
_backend_env = os.environ.get("TABMAT_BACKEND", "cpp").lower()
_BACKEND: Literal["cpp", "rust"] = "rust" if _backend_env == "rust" else "cpp"


def set_backend(backend: Literal["cpp", "rust"]):
    """Set the backend for categorical matrix operations."""
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
        from tabmat.ext import categorical as _cpp_module  # type: ignore[attr-defined]
    return _cpp_module


def _get_rust():
    global _rust_module
    if _rust_module is None:
        from tabmat.tabmat_rust_ext import tabmat_rust_ext as _rust_module
    return _rust_module


def transpose_matvec_fast(indices, other, n_cols, dtype, rows, cols, out):
    """Transpose matrix-vector multiplication (fast path, no drop_first)."""
    # Rust doesn't support row/col restrictions, fall back to C++ if present
    has_restrictions = rows is not None or cols is not None
    if _BACKEND == "rust" and not has_restrictions:
        rust = _get_rust()
        orig_dtype = other.dtype
        if orig_dtype != np.float64:
            other = other.astype(np.float64)
            out_f64 = out.astype(np.float64)
        else:
            out_f64 = out

        rust.transpose_matvec(indices, other, out_f64, False)

        if orig_dtype != np.float64:
            out[:] = out_f64.astype(orig_dtype)
        else:
            out[:] = out_f64
    else:
        _get_cpp().transpose_matvec_fast(indices, other, n_cols, dtype, rows, cols, out)


def transpose_matvec_complex(
    indices, other, n_cols, dtype, rows, cols, out, drop_first
):
    """Transpose matrix-vector multiplication (complex path, with drop_first)."""
    # Rust doesn't support row/col restrictions, fall back to C++ if present
    has_restrictions = rows is not None or cols is not None
    if _BACKEND == "rust" and not has_restrictions:
        rust = _get_rust()
        orig_dtype = other.dtype
        if orig_dtype != np.float64:
            other = other.astype(np.float64)
            out_f64 = out.astype(np.float64)
        else:
            out_f64 = out

        rust.transpose_matvec(indices, other, out_f64, drop_first)

        if orig_dtype != np.float64:
            out[:] = out_f64.astype(orig_dtype)
        else:
            out[:] = out_f64
    else:
        _get_cpp().transpose_matvec_complex(
            indices, other, n_cols, dtype, rows, cols, out, drop_first
        )


def matvec_fast(indices, other, n_rows, cols, n_cols, out):
    """Matrix-vector multiplication (fast path, no drop_first)."""
    # Rust doesn't support col restrictions, fall back to C++ if present
    if _BACKEND == "rust" and cols is None:
        rust = _get_rust()
        orig_dtype = other.dtype
        if orig_dtype != np.float64:
            other = other.astype(np.float64)
            out_f64 = out.astype(np.float64)
        else:
            out_f64 = out

        rust.matvec(indices, other, out_f64, False)

        if orig_dtype != np.float64:
            out[:] = out_f64.astype(orig_dtype)
        else:
            out[:] = out_f64
    else:
        _get_cpp().matvec_fast(indices, other, n_rows, cols, n_cols, out)


def matvec_complex(indices, other, n_rows, cols, n_cols, out, drop_first):
    """Matrix-vector multiplication (complex path, with drop_first)."""
    # Rust doesn't support col restrictions, fall back to C++ if present
    if _BACKEND == "rust" and cols is None:
        rust = _get_rust()
        orig_dtype = other.dtype
        if orig_dtype != np.float64:
            other = other.astype(np.float64)
            out_f64 = out.astype(np.float64)
        else:
            out_f64 = out

        rust.matvec(indices, other, out_f64, drop_first)

        if orig_dtype != np.float64:
            out[:] = out_f64.astype(orig_dtype)
        else:
            out[:] = out_f64
    else:
        _get_cpp().matvec_complex(indices, other, n_rows, cols, n_cols, out, drop_first)


def sandwich_categorical_fast(indices, d, rows, dtype, n_cols):
    """Sandwich product for categorical matrix (fast path, no drop_first)."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = d.dtype
        if orig_dtype != np.float64:
            d = d.astype(np.float64)

        result = rust.sandwich_categorical(indices, d, rows, n_cols, False)

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().sandwich_categorical_fast(indices, d, rows, dtype, n_cols)


def sandwich_categorical_complex(indices, d, rows, dtype, n_cols, drop_first):
    """Sandwich product for categorical matrix (complex path, with drop_first)."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = d.dtype
        if orig_dtype != np.float64:
            d = d.astype(np.float64)

        result = rust.sandwich_categorical(indices, d, rows, n_cols, drop_first)

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().sandwich_categorical_complex(
            indices, d, rows, dtype, n_cols, drop_first
        )


def multiply_complex(indices, d, ncols, dtype, drop_first):
    """Multiply a CategoricalMatrix by a vector d.

    Note: This function is not implemented in Rust, always uses C++ backend.
    """
    return _get_cpp().multiply_complex(indices, d, ncols, dtype, drop_first)


def subset_categorical_complex(indices, ncols, drop_first):
    """Construct inputs to transform a CategoricalMatrix into a csr_matrix.

    Note: This function is not implemented in Rust, always uses C++ backend.
    """
    return _get_cpp().subset_categorical_complex(indices, ncols, drop_first)
