"""Backend selection for categorical and sparse matrix operations.

This module provides a unified interface for categorical and sparse matrix
operations that can use either the C++ (Cython) or Rust backend.

Usage:
    from tabmat.ext.backend import get_backend, set_backend

    # Get current backend
    backend = get_backend()

    # Set backend to "rust" or "cpp"
    set_backend("rust")

    # Use functions from the backend
    backend.transpose_matvec_fast(...)
    backend.csr_matvec_unrestricted(...)
"""

import os
from typing import Literal

# Global backend setting
_backend_env = os.environ.get("TABMAT_BACKEND", "cpp").lower()
_BACKEND: Literal["cpp", "rust"] = "rust" if _backend_env == "rust" else "cpp"

# Backend modules (lazy loaded)
_cpp_backend = None
_rust_backend = None


def _load_cpp_backend():
    """Load the C++ (Cython) backend."""
    global _cpp_backend
    if _cpp_backend is None:
        from tabmat.ext import (  # type: ignore[attr-defined]
            categorical as _cpp_categorical,
        )
        from tabmat.ext import sparse as _cpp_sparse  # type: ignore[attr-defined]
        from tabmat.ext import split as _cpp_split  # type: ignore[attr-defined]

        class CppBackend:
            """C++ backend wrapper."""

            # From categorical.pyx
            transpose_matvec_fast = staticmethod(_cpp_categorical.transpose_matvec_fast)
            transpose_matvec_complex = staticmethod(
                _cpp_categorical.transpose_matvec_complex
            )
            matvec_fast = staticmethod(_cpp_categorical.matvec_fast)
            matvec_complex = staticmethod(_cpp_categorical.matvec_complex)
            sandwich_categorical_fast = staticmethod(
                _cpp_categorical.sandwich_categorical_fast
            )
            sandwich_categorical_complex = staticmethod(
                _cpp_categorical.sandwich_categorical_complex
            )
            get_col_included = staticmethod(_cpp_categorical.get_col_included)
            multiply_complex = staticmethod(_cpp_categorical.multiply_complex)
            subset_categorical_complex = staticmethod(
                _cpp_categorical.subset_categorical_complex
            )

            # From split.pyx
            sandwich_cat_dense = staticmethod(_cpp_split.sandwich_cat_dense)
            sandwich_cat_cat = staticmethod(_cpp_split.sandwich_cat_cat)

            # From sparse.pyx
            csr_matvec_unrestricted = staticmethod(_cpp_sparse.csr_matvec_unrestricted)
            csr_matvec = staticmethod(_cpp_sparse.csr_matvec)
            csc_rmatvec_unrestricted = staticmethod(
                _cpp_sparse.csc_rmatvec_unrestricted
            )
            csc_rmatvec = staticmethod(_cpp_sparse.csc_rmatvec)
            sparse_sandwich = staticmethod(_cpp_sparse.sparse_sandwich)
            csr_dense_sandwich = staticmethod(_cpp_sparse.csr_dense_sandwich)
            transpose_square_dot_weights = staticmethod(
                _cpp_sparse.transpose_square_dot_weights
            )

            name = "cpp"

        _cpp_backend = CppBackend()
    return _cpp_backend


def _load_rust_backend():
    """Load the Rust backend."""
    global _rust_backend
    if _rust_backend is None:
        try:
            from tabmat.tabmat_rust_ext import tabmat_rust_ext as _rust_ext
        except ImportError as e:
            raise ImportError(
                "Rust backend not available. Build the Rust extension "
                "or use set_backend('cpp')."
            ) from e

        class RustBackend:
            """Rust backend wrapper.

            The Rust backend uses unified functions with drop_first parameters.
            These wrapper methods provide API compatibility with the C++ backend.

            Note: Some functions still use the C++ backend because they're not
            performance-critical or not yet implemented in Rust.
            """

            # Fall back to C++ for functions not yet in Rust
            @staticmethod
            def get_col_included(cols, n_cols):
                return _load_cpp_backend().get_col_included(cols, n_cols)

            @staticmethod
            def multiply_complex(indices, d, ncols, dtype, drop_first):
                return _load_cpp_backend().multiply_complex(
                    indices, d, ncols, dtype, drop_first
                )

            @staticmethod
            def subset_categorical_complex(indices, ncols, drop_first):
                return _load_cpp_backend().subset_categorical_complex(
                    indices, ncols, drop_first
                )

            # Wrappers for transpose_matvec that match C++ API
            @staticmethod
            def transpose_matvec_fast(
                indices, other, n_cols, dtype, rows, other_col, out
            ):
                _rust_ext.transpose_matvec(indices, other, out, False)

            @staticmethod
            def transpose_matvec_complex(
                indices, other, n_cols, dtype, rows, other_col, out, drop_first
            ):
                _rust_ext.transpose_matvec(indices, other, out, drop_first)

            # Wrappers for matvec that match C++ API
            @staticmethod
            def matvec_fast(indices, other, n_rows, rows, n_cols, out):
                _rust_ext.matvec(indices, other, out, False)

            @staticmethod
            def matvec_complex(indices, other, n_rows, rows, n_cols, out, drop_first):
                _rust_ext.matvec(indices, other, out, drop_first)

            # Wrappers for sandwich_categorical that match C++ API
            @staticmethod
            def sandwich_categorical_fast(indices, d, rows, dtype, n_cols):
                return _rust_ext.sandwich_categorical(indices, d, rows, n_cols, False)

            @staticmethod
            def sandwich_categorical_complex(
                indices, d, rows, dtype, n_cols, drop_first
            ):
                return _rust_ext.sandwich_categorical(
                    indices, d, rows, n_cols, drop_first
                )

            # Wrapper for sandwich_cat_dense that matches C++ API
            @staticmethod
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
                return _rust_ext.sandwich_cat_dense(
                    i_indices,
                    d,
                    mat_j,
                    rows,
                    j_cols,
                    i_ncol,
                    is_c_contiguous,
                    drop_first,
                )

            # Wrapper for sandwich_cat_cat that matches C++ API
            @staticmethod
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
                return _rust_ext.sandwich_cat_cat(
                    i_indices,
                    j_indices,
                    d,
                    rows,
                    i_ncol,
                    j_ncol,
                    i_drop_first,
                    j_drop_first,
                )

            # Sparse operations
            # Note: Rust backend only supports float64, so we convert if needed
            @staticmethod
            def csr_matvec_unrestricted(X, v, out, X_indices):
                import numpy as np

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
                _rust_ext.csr_matvec_unrestricted(data, X.indices, X.indptr, v, out)
                if orig_dtype != np.float64:
                    out = out.astype(orig_dtype)
                return out

            @staticmethod
            def csr_matvec(X, v, rows, cols):
                import numpy as np

                orig_dtype = X.dtype
                if orig_dtype != np.float64:
                    data = X.data.astype(np.float64)
                    v = v.astype(np.float64)
                else:
                    data = X.data
                result = _rust_ext.csr_matvec(
                    data, X.indices, X.indptr, v, rows, cols, X.shape[1]
                )
                if orig_dtype != np.float64:
                    result = result.astype(orig_dtype)
                return result

            @staticmethod
            def csc_rmatvec_unrestricted(XT, v, out, XT_indices):
                import numpy as np

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
                _rust_ext.csc_rmatvec_unrestricted(data, XT.indices, XT.indptr, v, out)
                if orig_dtype != np.float64:
                    out = out.astype(orig_dtype)
                return out

            @staticmethod
            def csc_rmatvec(XT, v, rows, cols):
                import numpy as np

                orig_dtype = XT.dtype
                if orig_dtype != np.float64:
                    data = XT.data.astype(np.float64)
                    v = v.astype(np.float64)
                else:
                    data = XT.data
                result = _rust_ext.csc_rmatvec(
                    data, XT.indices, XT.indptr, v, rows, cols, XT.shape[0]
                )
                if orig_dtype != np.float64:
                    result = result.astype(orig_dtype)
                return result

            @staticmethod
            def sparse_sandwich(A, AT, d, rows, cols):
                import numpy as np

                orig_dtype = A.dtype
                if orig_dtype != np.float64:
                    a_data = A.data.astype(np.float64)
                    at_data = AT.data.astype(np.float64)
                    d = d.astype(np.float64)
                else:
                    a_data = A.data
                    at_data = AT.data
                result = _rust_ext.sparse_sandwich(
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

            @staticmethod
            def csr_dense_sandwich(A, B, d, rows, A_cols, B_cols):
                import numpy as np

                orig_dtype = A.dtype
                is_c_contiguous = B.flags["C_CONTIGUOUS"]
                if orig_dtype != np.float64:
                    a_data = A.data.astype(np.float64)
                    d = d.astype(np.float64)
                    B = B.astype(np.float64)
                else:
                    a_data = A.data
                result = _rust_ext.csr_dense_sandwich(
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

            @staticmethod
            def transpose_square_dot_weights(data, indices, indptr, weights, dtype):
                import numpy as np

                orig_dtype = data.dtype
                if orig_dtype != np.float64:
                    data = data.astype(np.float64)
                    weights = weights.astype(np.float64)
                result = _rust_ext.transpose_square_dot_weights(
                    data, indices, indptr, weights
                )
                if orig_dtype != np.float64:
                    result = result.astype(orig_dtype)
                return result

            name = "rust"

        _rust_backend = RustBackend()
    return _rust_backend


def get_backend():
    """Get the currently active backend.

    Returns:
        Backend object with categorical matrix operations.
    """
    if _BACKEND == "rust":
        return _load_rust_backend()
    else:
        return _load_cpp_backend()


def set_backend(backend: Literal["cpp", "rust"]):
    """Set the backend for categorical matrix operations.

    Args:
        backend: Either "cpp" for C++ (Cython) or "rust" for Rust backend.
    """
    global _BACKEND
    if backend not in ("cpp", "rust"):
        raise ValueError(f"Unknown backend: {backend}. Use 'cpp' or 'rust'.")
    _BACKEND = backend


def available_backends():
    """Return list of available backends.

    Returns:
        List of backend names that are available.
    """
    available = ["cpp"]  # C++ is always available

    try:
        _load_rust_backend()
        available.append("rust")
    except ImportError:
        pass

    return available
