"""Sparse operations with automatic backend dispatch.

This module provides the same API as the C++ sparse module but automatically
dispatches to either the C++ or Rust backend based on the current setting.
"""

import os
from typing import Literal

# Global backend setting
_backend_env = os.environ.get("TABMAT_BACKEND", "cpp").lower()
_BACKEND: Literal["cpp", "rust"] = "rust" if _backend_env == "rust" else "cpp"


def set_backend(backend: Literal["cpp", "rust"]):
    """Set the backend for sparse matrix operations."""
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
        from tabmat.ext import sparse as _cpp_module  # type: ignore[attr-defined]
    return _cpp_module


def _get_rust():
    global _rust_module
    if _rust_module is None:
        from tabmat.tabmat_rust_ext import tabmat_rust_ext as _rust_module
    return _rust_module


def csr_matvec_unrestricted(X, v, out, X_indices):
    """CSR matrix-vector multiplication: out += X @ v."""
    if _BACKEND == "rust":
        import numpy as np

        rust = _get_rust()
        orig_dtype = X.dtype
        if orig_dtype != np.float64:
            data = X.data.astype(np.float64)
            v = v.astype(np.float64)
        else:
            data = X.data

        if out is None:
            out = np.zeros(X.shape[0], dtype=np.float64)
        elif out.dtype != np.float64:
            out = out.astype(np.float64)

        rust.csr_matvec_unrestricted(data, X.indices, X.indptr, v, out)

        if orig_dtype != np.float64:
            out = out.astype(orig_dtype)
        return out
    else:
        return _get_cpp().csr_matvec_unrestricted(X, v, out, X_indices)


def csr_matvec(X, v, rows, cols):
    """CSR matrix-vector multiplication with row/column restrictions."""
    if _BACKEND == "rust":
        import numpy as np

        rust = _get_rust()
        orig_dtype = X.dtype
        if orig_dtype != np.float64:
            data = X.data.astype(np.float64)
            v = v.astype(np.float64)
        else:
            data = X.data

        result = rust.csr_matvec(data, X.indices, X.indptr, v, rows, cols, X.shape[1])

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().csr_matvec(X, v, rows, cols)


def csc_rmatvec_unrestricted(XT, v, out, XT_indices):
    """CSC transpose matrix-vector multiplication: out += XT.T @ v."""
    if _BACKEND == "rust":
        import numpy as np

        rust = _get_rust()
        orig_dtype = XT.dtype
        if orig_dtype != np.float64:
            data = XT.data.astype(np.float64)
            v = v.astype(np.float64)
        else:
            data = XT.data

        if out is None:
            out = np.zeros(XT.shape[1], dtype=np.float64)
        elif out.dtype != np.float64:
            out = out.astype(np.float64)

        rust.csc_rmatvec_unrestricted(data, XT.indices, XT.indptr, v, out)

        if orig_dtype != np.float64:
            out = out.astype(orig_dtype)
        return out
    else:
        return _get_cpp().csc_rmatvec_unrestricted(XT, v, out, XT_indices)


def csc_rmatvec(XT, v, rows, cols):
    """CSC transpose matrix-vector multiplication with restrictions."""
    if _BACKEND == "rust":
        import numpy as np

        rust = _get_rust()
        orig_dtype = XT.dtype
        if orig_dtype != np.float64:
            data = XT.data.astype(np.float64)
            v = v.astype(np.float64)
        else:
            data = XT.data

        result = rust.csc_rmatvec(
            data, XT.indices, XT.indptr, v, rows, cols, XT.shape[0]
        )

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().csc_rmatvec(XT, v, rows, cols)


def sparse_sandwich(A, AT, d, rows, cols):
    """Sparse sandwich product: A.T @ diag(d) @ A."""
    if _BACKEND == "rust":
        import numpy as np

        rust = _get_rust()
        orig_dtype = A.dtype
        if orig_dtype != np.float64:
            a_data = A.data.astype(np.float64)
            at_data = AT.data.astype(np.float64)
            d = d.astype(np.float64)
        else:
            a_data = A.data
            at_data = AT.data

        result = rust.sparse_sandwich(
            a_data,
            A.indices,
            A.indptr,
            at_data,
            AT.indices,
            AT.indptr,
            d,
            rows,
            cols,
            d.shape[0],
            A.shape[1],
        )

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().sparse_sandwich(A, AT, d, rows, cols)


def csr_dense_sandwich(A, B, d, rows, A_cols, B_cols):
    """CSR-dense sandwich: A.T @ diag(d) @ B."""
    if _BACKEND == "rust":
        import numpy as np

        rust = _get_rust()
        orig_dtype = A.dtype
        is_c_contiguous = B.flags["C_CONTIGUOUS"]

        if orig_dtype != np.float64:
            a_data = A.data.astype(np.float64)
            d = d.astype(np.float64)
            B = B.astype(np.float64)
        else:
            a_data = A.data

        result = rust.csr_dense_sandwich(
            a_data,
            A.indices,
            A.indptr,
            B,
            d,
            rows,
            A_cols,
            B_cols,
            A.shape[1],
            is_c_contiguous,
        )

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().csr_dense_sandwich(A, B, d, rows, A_cols, B_cols)


def transpose_square_dot_weights(data, indices, indptr, weights, dtype):
    """Compute weighted squared column norms for CSC matrix."""
    if _BACKEND == "rust":
        import numpy as np

        rust = _get_rust()
        orig_dtype = data.dtype

        if orig_dtype != np.float64:
            data = data.astype(np.float64)
            weights = weights.astype(np.float64)

        result = rust.transpose_square_dot_weights(data, indices, indptr, weights)

        if orig_dtype != np.float64:
            result = result.astype(orig_dtype)
        return result
    else:
        return _get_cpp().transpose_square_dot_weights(
            data, indices, indptr, weights, dtype
        )
