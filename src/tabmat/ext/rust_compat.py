"""
Wrapper for Rust extensions with dtype conversion.

This module provides Python wrappers around Rust implementations
that ensure proper dtype conversion (primarily to float64).
"""

import numpy as np

# Import Rust functions
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


# Dense functions - add dtype conversion wrappers
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


# transpose_square_dot_weights dispatcher
def transpose_square_dot_weights(*args, **kwargs):
    """Dispatch to dense or sparse version based on arguments"""
    # Dense version: (X: ndarray, weights: array, shift: array)
    # Sparse version: (data, indices, indptr, weights, dtype)
    if len(args) == 3:
        # Dense version
        x, weights, shift = args
        x = np.asarray(x, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        shift = np.asarray(shift, dtype=np.float64)
        return _rust_dense_transpose_square_dot_weights(x, weights, shift)
    elif len(args) == 5 or (len(args) == 4 and "dtype" in kwargs):
        # Sparse version - ignore dtype parameter
        if len(args) == 5:
            data, indices, indptr, weights, _ = args  # dtype ignored
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


# Sparse functions - categorical matrix operations


def matvec_fast(indices, other, n_rows, cols, n_cols, out):
    """Wrapper with dtype conversion for matvec_fast."""  # noqa: E501
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
    """Wrapper with dtype conversion for matvec_complex."""  # noqa: E501
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
    """Wrapper with dtype conversion for transpose_matvec_fast."""  # noqa: E501
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
    """Wrapper with dtype conversion for transpose_matvec_complex."""  # noqa: E501
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
    """Wrapper with dtype conversion for sandwich_categorical_fast."""  # noqa: E501
    return _rust_sandwich_categorical_fast(indices, d, rows, n_cols)


def sandwich_categorical_complex(indices, d, rows, dtype, n_cols, drop_first):
    """Wrapper with dtype conversion for sandwich_categorical_complex."""  # noqa: E501
    return _rust_sandwich_categorical_complex(indices, d, rows, n_cols, drop_first)


def multiply_complex(indices, d, ncols, dtype, drop_first):
    """Wrapper with dtype conversion for multiply_complex."""  # noqa: E501
    # Ensure d is float64 for Rust function
    d = np.asarray(d, dtype=np.float64)
    result = _rust_multiply_complex(indices, d, drop_first)
    # result is (data, indices, indptr) - convert data to requested dtype
    data, new_indices, indptr = result
    if dtype != np.float64:
        data = np.asarray(data, dtype=dtype)
    return (data, new_indices, indptr)


def subset_categorical_complex(indices, ncols, drop_first):
    """Wrapper with dtype conversion for subset_categorical_complex."""  # noqa: E501
    return _rust_subset_categorical_complex(indices, drop_first)


def get_col_included(cols, n_cols):
    """Wrapper with dtype conversion for get_col_included."""  # noqa: E501
    return _rust_get_col_included(cols, n_cols)


# Sparse matrix wrapper functions
sparse_sandwich = _rust_sparse_sandwich
csr_matvec_unrestricted = _rust_csr_matvec_unrestricted
csr_matvec = _rust_csr_matvec
csc_rmatvec_unrestricted = _rust_csc_rmatvec_unrestricted
csc_rmatvec = _rust_csc_rmatvec
csr_dense_sandwich = _rust_csr_dense_sandwich
categorical_sandwich = _rust_categorical_sandwich


# Split functions


def is_sorted(a):
    """Wrapper with dtype conversion for is_sorted."""  # noqa: E501
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
    """Wrapper with dtype conversion for sandwich_cat_cat."""  # noqa: E501
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
    """Wrapper with dtype conversion for sandwich_cat_dense."""  # noqa: E501
    return _rust_sandwich_cat_dense(
        i_indices, d, mat_j, rows, j_cols, i_ncol, drop_first
    )


def split_col_subsets(self, cols):
    """Wrapper with dtype conversion for split_col_subsets."""  # noqa: E501
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
