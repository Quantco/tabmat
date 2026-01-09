"""
Compatibility wrapper for Rust extensions.

This module provides backward compatibility by wrapping Rust implementations
with the same API as the old Cython extensions.
"""

import numpy as np

# Try to import Rust functions with internal names
try:
    from .tabmat_ext import (
        categorical_sandwich as _rust_categorical_sandwich,
    )
    from .tabmat_ext import (
        csc_rmatvec as _rust_csc_rmatvec,
    )
    from .tabmat_ext import (
        csc_rmatvec_unrestricted as _rust_csc_rmatvec_unrestricted,
    )
    from .tabmat_ext import (
        csr_dense_sandwich as _rust_csr_dense_sandwich,
    )
    from .tabmat_ext import (
        csr_matvec as _rust_csr_matvec,
    )
    from .tabmat_ext import (
        csr_matvec_unrestricted as _rust_csr_matvec_unrestricted,
    )
    from .tabmat_ext import (
        dense_matvec as _rust_dense_matvec,
    )
    from .tabmat_ext import (
        dense_rmatvec as _rust_dense_rmatvec,
    )
    from .tabmat_ext import (
        dense_sandwich as _rust_dense_sandwich,
    )
    from .tabmat_ext import (
        dense_transpose_square_dot_weights as _rust_dense_transpose_square_dot_weights,
    )
    from .tabmat_ext import (
        get_col_included as _rust_get_col_included,
    )
    from .tabmat_ext import (
        is_sorted as _rust_is_sorted,
    )
    from .tabmat_ext import (
        matvec_complex as _rust_matvec_complex,
    )
    from .tabmat_ext import (
        matvec_fast as _rust_matvec_fast,
    )
    from .tabmat_ext import (
        multiply_complex as _rust_multiply_complex,
    )
    from .tabmat_ext import (
        sandwich_cat_cat as _rust_sandwich_cat_cat,
    )
    from .tabmat_ext import (
        sandwich_cat_dense as _rust_sandwich_cat_dense,
    )
    from .tabmat_ext import (
        sandwich_categorical_complex as _rust_sandwich_categorical_complex,
    )
    from .tabmat_ext import (
        sandwich_categorical_fast as _rust_sandwich_categorical_fast,
    )
    from .tabmat_ext import (
        sparse_sandwich as _rust_sparse_sandwich,
    )
    from .tabmat_ext import (
        sparse_transpose_square_dot_weights as _rust_sparse_transpose_square_dot_weights,  # noqa: E501
    )
    from .tabmat_ext import (
        split_col_subsets as _rust_split_col_subsets,
    )
    from .tabmat_ext import (
        subset_categorical_complex as _rust_subset_categorical_complex,
    )
    from .tabmat_ext import (
        transpose_matvec_complex as _rust_transpose_matvec_complex,
    )
    from .tabmat_ext import (
        transpose_matvec_fast as _rust_transpose_matvec_fast,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    # Fall back to old Cython extensions
    from .categorical import (
        categorical_sandwich as _rust_categorical_sandwich,
    )
    from .categorical import (
        get_col_included as _rust_get_col_included,
    )
    from .categorical import (
        matvec_complex as _rust_matvec_complex,
    )
    from .categorical import (
        matvec_fast as _rust_matvec_fast,
    )
    from .categorical import (
        multiply_complex as _rust_multiply_complex,
    )
    from .categorical import (
        sandwich_categorical_complex as _rust_sandwich_categorical_complex,
    )
    from .categorical import (
        sandwich_categorical_fast as _rust_sandwich_categorical_fast,
    )
    from .categorical import (
        subset_categorical_complex as _rust_subset_categorical_complex,
    )
    from .categorical import (
        transpose_matvec_complex as _rust_transpose_matvec_complex,
    )
    from .categorical import (
        transpose_matvec_fast as _rust_transpose_matvec_fast,
    )
    from .dense import (
        dense_matvec as _rust_dense_matvec,
    )
    from .dense import (
        dense_rmatvec as _rust_dense_rmatvec,
    )
    from .dense import (
        dense_sandwich as _rust_dense_sandwich,
    )
    from .dense import (
        transpose_square_dot_weights as _rust_dense_transpose_square_dot_weights,
    )
    from .sparse import (
        csc_rmatvec as _rust_csc_rmatvec,
    )
    from .sparse import (
        csc_rmatvec_unrestricted as _rust_csc_rmatvec_unrestricted,
    )
    from .sparse import (
        csr_dense_sandwich as _rust_csr_dense_sandwich,
    )
    from .sparse import (
        csr_matvec as _rust_csr_matvec,
    )
    from .sparse import (
        csr_matvec_unrestricted as _rust_csr_matvec_unrestricted,
    )
    from .sparse import (
        sparse_sandwich as _rust_sparse_sandwich,
    )
    from .sparse import (
        transpose_square_dot_weights as _rust_sparse_transpose_square_dot_weights,
    )
    from .split import (
        is_sorted as _rust_is_sorted,
    )
    from .split import (
        sandwich_cat_cat as _rust_sandwich_cat_cat,
    )
    from .split import (
        sandwich_cat_dense as _rust_sandwich_cat_dense,
    )
    from .split import (
        split_col_subsets as _rust_split_col_subsets,
    )

# Create compatibility wrappers that match Cython API

# Dense functions - add dtype conversion wrappers
if RUST_AVAILABLE:

    def dense_sandwich(x, d, rows, cols):
        """Wrapper with dtype conversion"""
        x = np.asarray(x, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        return _rust_dense_sandwich(x, d, rows, cols)

    def dense_rmatvec(x, v, rows, cols):
        """Wrapper with dtype conversion"""
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        return _rust_dense_rmatvec(x, v, rows, cols)

    def dense_matvec(x, v, rows, cols):
        """Wrapper with dtype conversion"""
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        return _rust_dense_matvec(x, v, rows, cols)
else:
    dense_sandwich = _rust_dense_sandwich
    dense_rmatvec = _rust_dense_rmatvec
    dense_matvec = _rust_dense_matvec


# transpose_square_dot_weights has different versions for dense and sparse
# We'll create a module-level namespace to hold both
class _TransposeSquareDotWeights:
    """Container for dense and sparse versions of transpose_square_dot_weights"""

    dense = _rust_dense_transpose_square_dot_weights

    if RUST_AVAILABLE:

        @staticmethod
        def sparse(data, indices, indptr, weights, dtype):
            """Sparse version that ignores dtype parameter"""
            return _rust_sparse_transpose_square_dot_weights(
                data, indices, indptr, weights
            )
    else:

        @staticmethod
        def sparse(data, indices, indptr, weights, dtype):
            """Fall back to Cython sparse version"""
            return _rust_sparse_transpose_square_dot_weights(
                data, indices, indptr, weights, dtype
            )


# For dense_matrix.py: imports transpose_square_dot_weights and uses it directly
# For sparse_matrix.py: imports transpose_square_dot_weights and uses it directly
# Both need to get the right version. Since dense_matrix imports first,
# we'll export the dense version by default and let sparse_matrix get the sparse version
# Actually, let's check which modules import this...


# For now, create wrapper functions that detect which version to use based on arguments
def transpose_square_dot_weights(*args, **kwargs):
    """Dispatch to dense or sparse version based on arguments"""
    # Dense version: (X: ndarray, weights: array, shift: array)
    # Sparse version: (data: array, indices: array, indptr: array, weights: array, dtype) # noqa: E501
    if len(args) == 3:
        # Dense version - ensure arrays are float64
        x, weights, shift = args
        x = np.asarray(x, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        shift = np.asarray(shift, dtype=np.float64)
        return _rust_dense_transpose_square_dot_weights(x, weights, shift)
    elif len(args) == 5 or (len(args) == 4 and "dtype" in kwargs):
        # Sparse version - ignore dtype, ensure arrays are float64
        if len(args) == 5:
            data, indices, indptr, weights, dtype = args
        else:
            data, indices, indptr, weights = args[:4]
            kwargs.get("dtype")
        data = np.asarray(data, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        return _rust_sparse_transpose_square_dot_weights(data, indices, indptr, weights)
    elif len(args) == 4:
        # Sparse version without dtype - ensure arrays are float64
        data, indices, indptr, weights = args
        data = np.asarray(data, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        return _rust_sparse_transpose_square_dot_weights(data, indices, indptr, weights)
    else:
        raise TypeError(
            f"transpose_square_dot_weights() takes 3, 4, or 5 arguments but {len(args)} were given"  # noqa: E501
        )


# Sparse functions - need wrappers for matrix objects
if RUST_AVAILABLE:

    def sparse_sandwich(A, AT, d, rows, cols):
        """Wrapper to match Cython API: sparse_sandwich(A, AT, d, rows, cols)

        A is CSC format, AT is CSR format (transpose of A).
        """
        # Ensure d is float64
        d = np.asarray(d, dtype=np.float64)
        return _rust_sparse_sandwich(
            A.data, A.indices, A.indptr, AT.data, AT.indices, AT.indptr, d, rows, cols
        )

    def csr_matvec_unrestricted(X, v, out, X_indices):
        """Wrapper to match Cython API: csr_matvec_unrestricted(X, v, out, X_indices)"""  # noqa: E501
        if out is None:
            out = np.zeros(X.shape[0], dtype=X.dtype)
        v = np.asarray(v, dtype=np.float64)
        result = _rust_csr_matvec_unrestricted(
            X.data, X.indices, X.indptr, v, X.shape[0]
        )
        out += result  # ADD to out, don't replace
        return out

    def csr_matvec(X, v, rows, cols):
        """Wrapper to match Cython API: csr_matvec(X, v, rows, cols)"""  # noqa: E501
        v = np.asarray(v, dtype=np.float64)
        return _rust_csr_matvec(X.data, X.indices, X.indptr, v, rows, cols, X.shape[1])

    def csc_rmatvec_unrestricted(XT, v, out, XT_indices):
        """Wrapper to match Cython API: csc_rmatvec_unrestricted(XT, v, out, XT_indices)"""  # noqa: E501
        if out is None:
            out = np.zeros(XT.shape[1], dtype=XT.dtype)
        v = np.asarray(v, dtype=np.float64)
        result = _rust_csc_rmatvec_unrestricted(
            XT.data, XT.indices, XT.indptr, v, XT.shape[1]
        )
        out += result  # ADD to out, don't replace
        return out

    def csc_rmatvec(XT, v, rows, cols):
        """Wrapper to match Cython API: csc_rmatvec(XT, v, rows, cols)"""  # noqa: E501
        v = np.asarray(v, dtype=np.float64)
        return _rust_csc_rmatvec(
            XT.data, XT.indices, XT.indptr, v, rows, cols, XT.shape[0]
        )

    def csr_dense_sandwich(A, B, d, rows, A_cols, B_cols):
        """Wrapper to match Cython API: csr_dense_sandwich(A, B, d, rows, A_cols, B_cols)"""  # noqa: E501
        B = np.asarray(B, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        return _rust_csr_dense_sandwich(
            A.data, A.indices, A.indptr, B, d, rows, A_cols, B_cols
        )
else:
    # Use Cython versions directly
    sparse_sandwich = _rust_sparse_sandwich
    csr_matvec_unrestricted = _rust_csr_matvec_unrestricted
    csr_matvec = _rust_csr_matvec
    csc_rmatvec_unrestricted = _rust_csc_rmatvec_unrestricted
    csc_rmatvec = _rust_csc_rmatvec
    csr_dense_sandwich = _rust_csr_dense_sandwich

# Categorical functions - need wrappers for extended signatures
if RUST_AVAILABLE:

    def categorical_sandwich(indices, d, rows, n_cols):
        """Wrapper to match Cython API: categorical_sandwich(indices, d, rows, n_cols)"""  # noqa: E501
        d = np.asarray(d, dtype=np.float64)
        return _rust_categorical_sandwich(indices, d, rows, n_cols)

    def matvec_fast(indices, other, n_rows, cols, n_cols, out):
        """Wrapper to match Cython API: matvec_fast(indices, other, n_rows, cols, n_cols, out)"""  # noqa: E501
        other = np.asarray(other, dtype=np.float64)
        if cols is None:
            # Simple case: use all columns
            result = _rust_matvec_fast(indices, other, n_rows)
        else:
            # Restricted case: filter by cols
            result = np.zeros(n_rows, dtype=np.float64)
            col_included = np.zeros(n_cols, dtype=np.int32)
            for col in cols:
                col_included[col] = 1
            for i in range(n_rows):
                col_idx = indices[i]
                if col_included[col_idx]:
                    result[i] = other[col_idx]
        out += result  # ADD to out, don't replace

    def matvec_complex(indices, other, n_rows, cols, n_cols, out, drop_first):
        """Wrapper to match Cython API: matvec_complex(indices, other, n_rows, cols, n_cols, out, drop_first)"""  # noqa: E501
        other = np.asarray(other, dtype=np.float64)
        result = _rust_matvec_complex(indices, other, n_rows, drop_first)
        if cols is not None:
            # Apply column filtering
            col_included = np.zeros(n_cols, dtype=np.int32)
            for col in cols:
                col_included[col] = 1
            for i in range(n_rows):
                col_idx = indices[i] - (1 if drop_first else 0)
                if col_idx >= 0 and not col_included[col_idx]:
                    result[i] = 0
        out += result  # ADD to out, don't replace

    def transpose_matvec_fast(indices, other, n_cols, dtype, rows, cols, out):
        """Wrapper to match Cython API: transpose_matvec_fast(indices, other, n_cols, dtype, rows, cols, out)"""  # noqa: E501
        if rows is None and cols is None:
            # Simple case
            result = _rust_transpose_matvec_fast(indices, other, n_cols)
        elif cols is None:
            # Row restrictions only
            result = np.zeros(n_cols, dtype=dtype)
            for row_idx in rows:
                result[indices[row_idx]] += other[row_idx]
        else:
            # Column restrictions (with or without row restrictions)
            result = np.zeros(n_cols, dtype=dtype)
            col_included = np.zeros(n_cols, dtype=np.int32)
            for col in cols:
                col_included[col] = 1

            if rows is None:
                for row_idx in range(len(indices)):
                    col = indices[row_idx]
                    if col_included[col]:
                        result[col] += other[row_idx]
            else:
                for row_idx in rows:
                    col = indices[row_idx]
                    if col_included[col]:
                        result[col] += other[row_idx]
        out += result  # ADD to out, don't replace

    def transpose_matvec_complex(
        indices, other, n_cols, dtype, rows, cols, out, drop_first
    ):
        """Wrapper to match Cython API: transpose_matvec_complex(indices, other, n_cols, dtype, rows, cols, out, drop_first)"""  # noqa: E501
        if rows is None and cols is None:
            result = _rust_transpose_matvec_complex(indices, other, n_cols, drop_first)
        elif cols is None:
            # Row restrictions only
            result = np.zeros(n_cols, dtype=dtype)
            for row_idx in rows:
                col_idx = indices[row_idx] - (1 if drop_first else 0)
                if col_idx >= 0:
                    result[col_idx] += other[row_idx]
        else:
            # Column restrictions
            result = np.zeros(n_cols, dtype=dtype)
            col_included = np.zeros(n_cols, dtype=np.int32)
            for col in cols:
                col_included[col] = 1

            if rows is None:
                for row_idx in range(len(indices)):
                    col_idx = indices[row_idx] - (1 if drop_first else 0)
                    if col_idx >= 0 and col_included[col_idx]:
                        result[col_idx] += other[row_idx]
            else:
                for row_idx in rows:
                    col_idx = indices[row_idx] - (1 if drop_first else 0)
                    if col_idx >= 0 and col_included[col_idx]:
                        result[col_idx] += other[row_idx]
        out += result  # ADD to out, don't replace

    def sandwich_categorical_fast(indices, d, rows, dtype, n_cols):
        """Wrapper to match Cython API: sandwich_categorical_fast(indices, d, rows, dtype, n_cols)"""  # noqa: E501
        return _rust_sandwich_categorical_fast(indices, d, rows, n_cols)

    def sandwich_categorical_complex(indices, d, rows, dtype, n_cols, drop_first):
        """Wrapper to match Cython API: sandwich_categorical_complex(indices, d, rows, dtype, n_cols, drop_first)"""  # noqa: E501
        return _rust_sandwich_categorical_complex(indices, d, rows, n_cols, drop_first)

    def multiply_complex(indices, d, ncols, dtype, drop_first):
        """Wrapper to match Cython API: multiply_complex(indices, d, ncols, dtype, drop_first)"""  # noqa: E501
        # Ensure d is float64 for Rust function
        d = np.asarray(d, dtype=np.float64)
        result = _rust_multiply_complex(indices, d, drop_first)
        # result is (data, indices, indptr) - convert data to requested dtype
        data, new_indices, indptr = result
        if dtype != np.float64:
            data = np.asarray(data, dtype=dtype)
        return (data, new_indices, indptr)

    def subset_categorical_complex(indices, ncols, drop_first):
        """Wrapper to match Cython API: subset_categorical_complex(indices, ncols, drop_first)"""  # noqa: E501
        return _rust_subset_categorical_complex(indices, drop_first)

    def get_col_included(cols, n_cols):
        """Wrapper to match Cython API: get_col_included(cols, n_cols)"""  # noqa: E501
        return _rust_get_col_included(cols, n_cols)
else:
    # Use Cython versions directly
    categorical_sandwich = _rust_categorical_sandwich
    matvec_fast = _rust_matvec_fast
    matvec_complex = _rust_matvec_complex
    transpose_matvec_fast = _rust_transpose_matvec_fast
    transpose_matvec_complex = _rust_transpose_matvec_complex
    sandwich_categorical_fast = _rust_sandwich_categorical_fast
    sandwich_categorical_complex = _rust_sandwich_categorical_complex
    multiply_complex = _rust_multiply_complex
    subset_categorical_complex = _rust_subset_categorical_complex
    get_col_included = _rust_get_col_included

# Split functions
if RUST_AVAILABLE:

    def is_sorted(a):
        """Wrapper to match Cython API: is_sorted(a)"""  # noqa: E501
        return _rust_is_sorted(a)

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
        """Wrapper to match Cython API: sandwich_cat_cat(...)"""  # noqa: E501
        return _rust_sandwich_cat_cat(
            i_indices, j_indices, d, rows, i_ncol, j_ncol, i_drop_first, j_drop_first
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
        """Wrapper to match Cython API: sandwich_cat_dense(...)"""  # noqa: E501
        return _rust_sandwich_cat_dense(
            i_indices, d, mat_j, rows, j_cols, i_ncol, drop_first
        )

    def split_col_subsets(self, cols):
        """Wrapper to match Cython API: split_col_subsets(self, cols)"""  # noqa: E501
        return _rust_split_col_subsets(self.indices, cols)
else:
    # Use Cython versions directly
    is_sorted = _rust_is_sorted
    sandwich_cat_cat = _rust_sandwich_cat_cat
    sandwich_cat_dense = _rust_sandwich_cat_dense
    split_col_subsets = _rust_split_col_subsets

__all__ = [
    "RUST_AVAILABLE",
    "dense_sandwich",
    "dense_rmatvec",
    "dense_matvec",
    "transpose_square_dot_weights",
    "sparse_sandwich",
    "csr_matvec_unrestricted",
    "csr_matvec",
    "csc_rmatvec_unrestricted",
    "csc_rmatvec",
    "csr_dense_sandwich",
    "categorical_sandwich",
    "matvec_fast",
    "matvec_complex",
    "transpose_matvec_fast",
    "transpose_matvec_complex",
    "sandwich_categorical_fast",
    "sandwich_categorical_complex",
    "multiply_complex",
    "subset_categorical_complex",
    "get_col_included",
    "is_sorted",
    "sandwich_cat_cat",
    "sandwich_cat_dense",
    "split_col_subsets",
]
