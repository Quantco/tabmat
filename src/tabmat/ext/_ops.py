"""
Matrix operations for tabmat.

This module provides optimized matrix operations backed by Rust implementations.
"""

import numpy as np

# Import functions from Rust extension
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
    sparse_transpose_square_dot_weights as _rust_sparse_transpose_square_dot_weights,
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

# Dense functions with dtype conversion wrappers


def dense_sandwich(x, d, rows, cols):
    """Compute dense sandwich product X.T @ diag(d) @ X."""
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    return _rust_dense_sandwich(x, d, rows, cols)


def dense_rmatvec(x, v, rows, cols):
    """Compute X.T @ v for dense matrix."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return _rust_dense_rmatvec(x, v, rows, cols)


def dense_matvec(x, v, rows, cols):
    """Compute X @ v for dense matrix."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return _rust_dense_matvec(x, v, rows, cols)


def transpose_square_dot_weights(*args, **kwargs):
    """Dispatch to dense or sparse version based on arguments.

    Dense version: (X: ndarray, weights: array, shift: array)
    Sparse version: (data: array, indices: array, indptr: array, weights: array, dtype)
    """
    if len(args) == 3:
        # Dense version
        x, weights, shift = args
        x = np.asarray(x, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        shift = np.asarray(shift, dtype=np.float64)
        return _rust_dense_transpose_square_dot_weights(x, weights, shift)
    elif len(args) == 5 or (len(args) == 4 and "dtype" in kwargs):
        # Sparse version (dtype parameter is ignored)
        if len(args) == 5:
            data, indices, indptr, weights, _ = args
        else:
            data, indices, indptr, weights = args[:4]
        data = np.asarray(data, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        return _rust_sparse_transpose_square_dot_weights(data, indices, indptr, weights)
    elif len(args) == 4:
        # Sparse version without dtype
        data, indices, indptr, weights = args
        data = np.asarray(data, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        return _rust_sparse_transpose_square_dot_weights(data, indices, indptr, weights)
    else:
        raise TypeError(
            f"transpose_square_dot_weights() takes 3, 4, or 5 arguments "
            f"but {len(args)} were given"
        )


# Sparse functions with matrix object wrappers


def sparse_sandwich(A, AT, d, rows, cols):
    """Compute sparse sandwich product X.T @ diag(d) @ X.

    A is CSC format, AT is CSR format (transpose of A).
    """
    d = np.asarray(d, dtype=np.float64)
    a_data = np.asarray(A.data, dtype=np.float64)
    at_data = np.asarray(AT.data, dtype=np.float64)
    return _rust_sparse_sandwich(
        a_data, A.indices, A.indptr, at_data, AT.indices, AT.indptr, d, rows, cols
    )


def csr_matvec_unrestricted(X, v, out, X_indices):
    """Compute X @ v for CSR matrix (unrestricted)."""
    if out is None:
        out = np.zeros(X.shape[0], dtype=X.dtype)
    else:
        if out.shape[0] != X.shape[0]:
            raise ValueError(
                f"The first dimension of 'out' must be {X.shape[0]}, "
                f"but it is {out.shape[0]}."
            )
    v = np.asarray(v, dtype=np.float64)
    data = np.asarray(X.data, dtype=np.float64)
    result = _rust_csr_matvec_unrestricted(data, X.indices, X.indptr, v, X.shape[0])
    out += result
    return out


def csr_matvec(X, v, rows, cols):
    """Compute X @ v for CSR matrix with row/col restrictions."""
    v = np.asarray(v, dtype=np.float64)
    data = np.asarray(X.data, dtype=np.float64)
    return _rust_csr_matvec(data, X.indices, X.indptr, v, rows, cols, X.shape[1])


def csc_rmatvec_unrestricted(XT, v, out, XT_indices):
    """Compute X.T @ v for CSC matrix (unrestricted)."""
    if out is None:
        out = np.zeros(XT.shape[1], dtype=XT.dtype)
    else:
        if out.shape[0] != XT.shape[1]:
            raise ValueError(
                f"The first dimension of 'out' must be {XT.shape[1]}, "
                f"but it is {out.shape[0]}."
            )
    v = np.asarray(v, dtype=np.float64)
    data = np.asarray(XT.data, dtype=np.float64)
    result = _rust_csc_rmatvec_unrestricted(data, XT.indices, XT.indptr, v, XT.shape[1])
    out += result
    return out


def csc_rmatvec(XT, v, rows, cols):
    """Compute X.T @ v for CSC matrix with row/col restrictions."""
    v = np.asarray(v, dtype=np.float64)
    data = np.asarray(XT.data, dtype=np.float64)
    return _rust_csc_rmatvec(data, XT.indices, XT.indptr, v, rows, cols, XT.shape[0])


def csr_dense_sandwich(A, B, d, rows, A_cols, B_cols):
    """Compute cross sandwich A.T @ diag(d) @ B for CSR and dense matrices."""
    # Ensure B is contiguous and float64 - Rust requires contiguous arrays
    B = np.ascontiguousarray(B, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    a_data = np.asarray(A.data, dtype=np.float64)
    return _rust_csr_dense_sandwich(
        a_data, A.indices, A.indptr, B, d, rows, A_cols, B_cols
    )


# Categorical functions


def categorical_sandwich(indices, d, rows, n_cols):
    """Compute categorical sandwich product (diagonal)."""
    d = np.asarray(d, dtype=np.float64)
    return _rust_categorical_sandwich(indices, d, rows, n_cols)


def matvec_fast(indices, other, n_rows, cols, n_cols, out):
    """Compute categorical matrix-vector product (simple case)."""
    if out.shape[0] != n_rows:
        raise ValueError(
            f"The first dimension of 'out' must be {n_rows}, but it is {out.shape[0]}."
        )
    other = np.asarray(other, dtype=np.float64)
    if cols is None:
        result = _rust_matvec_fast(indices, other, n_rows)
    else:
        result = np.zeros(n_rows, dtype=np.float64)
        col_included = np.zeros(n_cols, dtype=np.int32)
        for col in cols:
            col_included[col] = 1
        for i in range(n_rows):
            col_idx = indices[i]
            if col_included[col_idx]:
                result[i] = other[col_idx]
    out += result


def matvec_complex(indices, other, n_rows, cols, n_cols, out, drop_first):
    """Compute categorical matrix-vector product (with drop_first)."""
    if out.shape[0] != n_rows:
        raise ValueError(
            f"The first dimension of 'out' must be {n_rows}, but it is {out.shape[0]}."
        )
    other = np.asarray(other, dtype=np.float64)
    result = _rust_matvec_complex(indices, other, n_rows, drop_first)
    if cols is not None:
        col_included = np.zeros(n_cols, dtype=np.int32)
        for col in cols:
            col_included[col] = 1
        for i in range(n_rows):
            col_idx = indices[i] - (1 if drop_first else 0)
            if col_idx >= 0 and not col_included[col_idx]:
                result[i] = 0
    out += result


def transpose_matvec_fast(indices, other, n_cols, dtype, rows, cols, out):
    """Compute categorical transpose-vector product (simple case)."""
    if rows is None and cols is None:
        result = _rust_transpose_matvec_fast(indices, other, n_cols)
    elif cols is None:
        result = np.zeros(n_cols, dtype=dtype)
        for row_idx in rows:
            result[indices[row_idx]] += other[row_idx]
    else:
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
    out += result


def transpose_matvec_complex(
    indices, other, n_cols, dtype, rows, cols, out, drop_first
):
    """Compute categorical transpose-vector product (with drop_first)."""
    if rows is None and cols is None:
        result = _rust_transpose_matvec_complex(indices, other, n_cols, drop_first)
    elif cols is None:
        result = np.zeros(n_cols, dtype=dtype)
        for row_idx in rows:
            col_idx = indices[row_idx] - (1 if drop_first else 0)
            if col_idx >= 0:
                result[col_idx] += other[row_idx]
    else:
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
    out += result


def sandwich_categorical_fast(indices, d, rows, dtype, n_cols):
    """Compute categorical sandwich product (simple case)."""
    return _rust_sandwich_categorical_fast(indices, d, rows, n_cols)


def sandwich_categorical_complex(indices, d, rows, dtype, n_cols, drop_first):
    """Compute categorical sandwich product (with drop_first)."""
    return _rust_sandwich_categorical_complex(indices, d, rows, n_cols, drop_first)


def multiply_complex(indices, d, ncols, dtype, drop_first):
    """Element-wise multiplication of categorical matrix by diagonal vector."""
    d = np.asarray(d, dtype=np.float64)
    result = _rust_multiply_complex(indices, d, drop_first)
    data, new_indices, indptr = result
    if dtype != np.float64:
        data = np.asarray(data, dtype=dtype)
    return (data, new_indices, indptr)


def subset_categorical_complex(indices, ncols, drop_first):
    """Convert categorical matrix to CSR format."""
    return _rust_subset_categorical_complex(indices, drop_first)


def get_col_included(cols, n_cols):
    """Create a column inclusion mask."""
    return _rust_get_col_included(cols, n_cols)


# Split functions


def is_sorted(a):
    """Check if array is sorted."""
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
    """Compute cross sandwich for two categorical matrices."""
    return _rust_sandwich_cat_cat(
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
    has_missings=False,
    drop_first=False,
):
    """Compute cross sandwich for categorical and dense matrices."""
    return _rust_sandwich_cat_dense(
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


def split_col_subsets(self, cols):
    """Split column subsets for split matrix operations."""
    return _rust_split_col_subsets(self.indices, cols)


__all__ = [
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
