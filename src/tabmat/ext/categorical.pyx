# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np

cimport cython
cimport numpy as np
from cython cimport floating

from cython.parallel import prange

ctypedef np.uint8_t uint8
ctypedef np.int8_t int8
from libcpp cimport bool


cdef extern from "cat_split_helpers.cpp":
    void _transpose_matvec_all_rows[F](int, int*, F*, F*, int)
    void _transpose_matvec_all_rows_drop_first[F](int, int*, F*, F*, int)


def transpose_matvec(
    int[:] indices,
    floating[:] other,
    int n_cols,
    dtype,
    rows,
    cols,
    floating[:] out
):
    cdef int row, row_idx, n_keep_rows, col
    cdef int n_rows = len(indices)
    cdef int[:] rows_view, cols_view, cols_included

    cdef bool no_row_restrictions = rows is None or len(rows) == n_rows
    cdef bool no_col_restrictions = cols is None or len(cols) == n_cols

    # Case 1: No row or col restrictions
    if no_row_restrictions and no_col_restrictions:
        _transpose_matvec_all_rows(n_rows, &indices[0], &other[0], &out[0], out.size)
    # Case 2: row restrictions but no col restrictions
    elif no_col_restrictions:
        rows_view = rows
        n_keep_rows = len(rows_view)
        for row_idx in range(n_keep_rows):
            row = rows_view[row_idx]
            out[indices[row]] += other[row]
    # Cases 3 and 4: col restrictions
    else:
        cols_view = cols
        cols_included = get_col_included(cols, n_cols)
        # Case 3: Col restrictions but no row restrictions
        if no_row_restrictions:
            for row_idx in range(n_rows):
                col = indices[row_idx]
                if cols_included[col]:
                    out[col] += other[row_idx]
        # Case 4: Both col restrictions and row restrictions
        else:
            rows_view = rows
            n_keep_rows = len(rows_view)
            for row_idx in range(n_keep_rows):
                row = rows_view[row_idx]
                col = indices[row]
                if cols_included[col]:
                    out[col] += other[row]


def transpose_matvec_drop_first(
    int[:] indices,
    floating[:] other,
    int n_cols,
    dtype,
    rows,
    cols,
    floating[:] out
):
    cdef int row, row_idx, n_keep_rows, col_idx
    cdef int n_rows = len(indices)
    cdef int[:] rows_view, cols_view, cols_included

    cdef bool no_row_restrictions = rows is None or len(rows) == n_rows
    cdef bool no_col_restrictions = cols is None or len(cols) == n_cols

    # Case 1: No row or col restrictions
    if no_row_restrictions and no_col_restrictions:
        _transpose_matvec_all_rows_drop_first(n_rows, &indices[0], &other[0], &out[0], out.size)
    # Case 2: row restrictions but no col restrictions
    elif no_col_restrictions:
        rows_view = rows
        n_keep_rows = len(rows_view)
        for row_idx in range(n_keep_rows):
            row = rows_view[row_idx]
            col_idx = indices[row] - 1
            if col_idx != -1:
                out[col_idx] += other[row]
    # Cases 3 and 4: col restrictions
    else:
        cols_view = cols
        cols_included = get_col_included(cols, n_cols)
        # Case 3: Col restrictions but no row restrictions
        if no_row_restrictions:
            for row_idx in range(n_rows):
                col_idx = indices[row_idx] - 1
                if (col_idx != -1) and (cols_included[col_idx]):
                    out[col_idx] += other[row_idx]
        # Case 4: Both col restrictions and row restrictions
        else:
            rows_view = rows
            n_keep_rows = len(rows_view)
            for row_idx in range(n_keep_rows):
                row = rows_view[row_idx]
                col_idx = indices[row] - 1
                if (col_idx != -1) and (cols_included[col_idx]):
                    out[col_idx] += other[row]


def get_col_included(int[:] cols, int n_cols):
    cdef int[:] col_included = np.zeros(n_cols, dtype=np.int32)
    cdef int n_cols_included = len(cols)
    for Ci in range(n_cols_included):
        col_included[cols[Ci]] = 1
    return col_included


def matvec(
    const int[:] indices,
    floating[:] other,
    int n_rows,
    int[:] cols,
    int n_cols,
    floating[:] out_vec
):
    """Matrix-vector multiplication. With one-hot-encoded data,
    this is equivalent to `other[cat_index]`.
    """
    cdef int i, col_idx, Ci, k
    cdef int[:] col_included

    if cols is None:
        for i in prange(n_rows, nogil=True):
            out_vec[i] += other[indices[i]]
    else:
        col_included = get_col_included(cols, n_cols)
        for i in prange(n_rows, nogil=True):
            col_idx = indices[i]
            if col_included[col_idx] == 1:
                out_vec[i] += other[col_idx]
    return


def matvec_drop_first(
    const int[:] indices, 
    floating[:] other, 
    int n_rows, 
    int[:] cols,
    int n_cols, 
    floating[:] out_vec
):
    """See `matvec`. Here we drop the first category of the
    CategoricalMatrix so the indices refer to the column index + 1.
    """
    cdef int i, col_idx, Ci, k
    cdef int[:] col_included

    if cols is None:
        for i in prange(n_rows, nogil=True):
            col_idx = indices[i] - 1  # reference category is always 0.
            if col_idx != -1:
                out_vec[i] += other[col_idx]
    else:
        col_included = get_col_included(cols, n_cols)
        for i in prange(n_rows, nogil=True):
            col_idx = indices[i] - 1
            if (col_idx != -1) and (col_included[col_idx] == 1):
                out_vec[i] += other[col_idx]
    return


def sandwich_categorical(
    const int[:] indices,
    floating[:] d,
    int[:] rows,
    dtype,
    int n_cols
):
    cdef floating[:] res = np.zeros(n_cols, dtype=dtype)
    cdef int col_idx, k, k_idx
    cdef int n_rows = len(rows)

    for k_idx in range(n_rows):
        k = rows[k_idx]
        col_idx = indices[k]
        res[col_idx] += d[k]
    return np.asarray(res)


def sandwich_categorical_drop_first(
    const int[:] indices,
    floating[:] d,
    int[:] rows,
    dtype,
    int n_cols
):
    cdef floating[:] res = np.zeros(n_cols, dtype=dtype)
    cdef int col_idx, k, k_idx
    cdef int n_rows = len(rows)

    for k_idx in range(n_rows):
        k = rows[k_idx]
        col_idx = indices[k] - 1  # reference category is always 0.
        if col_idx != -1:
            res[col_idx] += d[k]
    return np.asarray(res)
