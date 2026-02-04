"""Split matrix operations with automatic backend dispatch.

This module provides the same API as the C++ split module but automatically
dispatches to either the C++ or Rust backend based on the current setting.
"""

import os
from typing import Literal

import numpy as np

# Global backend setting
_backend_env = os.environ.get("TABMAT_BACKEND", "cpp").lower()
_BACKEND: Literal["cpp", "rust"] = "rust" if _backend_env == "rust" else "cpp"


def set_backend(backend: Literal["cpp", "rust"]):
    """Set the backend for split matrix operations."""
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
        from tabmat.ext import split as _cpp_module  # type: ignore[attr-defined]
    return _cpp_module


def _get_rust():
    global _rust_module
    if _rust_module is None:
        from tabmat.tabmat_rust_ext import tabmat_rust_ext as _rust_module
    return _rust_module


def sandwich_cat_cat(
    i_indices,
    j_indices,
    i_ncol,
    j_ncol,
    d,
    rows,
    dtype,
    i_drop_first,
    j_drop_first,
    i_has_missings,
    j_has_missings,
):
    """Cross-sandwich product between two categorical matrices."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = d.dtype
        if orig_dtype != np.float64:
            d = d.astype(np.float64)

        result = rust.sandwich_cat_cat(
            i_indices, j_indices, d, rows, i_ncol, j_ncol, i_drop_first, j_drop_first
        )

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().sandwich_cat_cat(
            i_indices,
            j_indices,
            i_ncol,
            j_ncol,
            d,
            rows,
            dtype,
            i_drop_first,
            j_drop_first,
            i_has_missings,
            j_has_missings,
        )


def sandwich_cat_dense(
    i_indices,
    i_ncol,
    d,
    mat_j,
    rows,
    j_cols,
    is_c_contiguous,
    has_missings,
    drop_first,
):
    """Cross-sandwich product between categorical and dense matrices."""
    if _BACKEND == "rust":
        rust = _get_rust()
        orig_dtype = d.dtype
        if orig_dtype != np.float64:
            d = d.astype(np.float64)
            mat_j = mat_j.astype(np.float64)

        result = rust.sandwich_cat_dense(
            i_indices, d, mat_j, rows, j_cols, i_ncol, is_c_contiguous, drop_first
        )

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().sandwich_cat_dense(
            i_indices,
            i_ncol,
            d,
            mat_j,
            rows,
            j_cols,
            is_c_contiguous,
            has_missings,
            drop_first,
        )


# Re-export functions that don't have Rust implementations
def split_col_subsets(self, cols):
    """Split column subsets - only available in C++ backend."""
    return _get_cpp().split_col_subsets(self, cols)


def is_sorted(a):
    """Check if array is sorted - only available in C++ backend."""
    return _get_cpp().is_sorted(a)
