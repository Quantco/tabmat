# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

from cython cimport floating
cimport cython
from cython.parallel import prange
ctypedef np.uint8_t uint8
ctypedef np.int8_t int8
from libcpp cimport bool


cdef extern from "cat_split_helpers.cpp":
    void _transpose_matvec_all_rows[F](int, int*, F*, F*, int)


# TODO: make this work with 'cols'
def transpose_matvec(int[:] indices, floating[:] other, int n_cols, dtype,
                  rows, cols, out):
    cdef floating[:] res = out
    cdef int row, row_idx, n_keep_rows, col, col_idx, j
    cdef int n_rows = len(indices)
    cdef int[:] rows_view, cols_view

    cdef bool no_row_restrictions = rows is None or len(rows) == n_rows
    cdef bool no_col_restrictions = cols is None or len(cols) == n_cols

    # Case 1: No row or col restrictions
    if no_row_restrictions and no_col_restrictions:
        _transpose_matvec_all_rows(n_rows, &indices[0], &other[0], &res[0], res.size)
    # Case 2: row restrictions but no col restrictions
    elif no_col_restrictions:
        rows_view = rows
        n_keep_rows = len(rows_view)
        for row_idx in range(n_keep_rows):
            row = rows_view[row_idx]
            res[indices[row]] += other[row]
    # Cases 3 and 4: col restrictions
    else:
        cols_view = cols
        # Case 3: Col restrictions but no row restrictions
        if no_col_restrictions:
            for row_idx in range(n_rows):
                col = indices[row_idx]
                # if col in cols_view:
                #    res[col] += other[row_idx]
                for col_idx in range(n_cols):
                    j = cols_view[col_idx]
                    if col == j:
                        res[col] += other[row_idx]
                        break
        # Case 4: Both col restrictions and row restrictions
        else:
            rows_view = rows
            n_keep_rows = len(rows_view)
            for row_idx in range(n_keep_rows):
                row = rows_view[row_idx]
                col = indices[row]
                if col in cols_view:
                    res[col] += other[row]
                # for c_idx in range(n_cols):
                #     c = cols_view[c_idx]
                #     if c == j:
                #         res[j] += other[i]
                #         break


def get_col_included(int[:] cols, int n_cols):
    cdef int[:] col_included = np.zeros(n_cols, dtype=np.int32)
    cdef int n_cols_included = len(cols)
    for Ci in range(n_cols_included):
        col_included[cols[Ci]] = 1
    return col_included


def matvec(const int[:] indices, floating[:] other, int n_rows, int[:] cols,
        int n_cols, floating[:] out_vec):
    cdef int i, col, Ci, k
    cdef int[:] col_included

    if cols is None:
        for i in prange(n_rows, nogil=True):
            out_vec[i] += other[indices[i]]
    else:
        col_included = get_col_included(cols, n_cols)
        for i in prange(n_rows, nogil=True):
            col = indices[i]
            if col_included[col] == 1:
                out_vec[i] += other[indices[i]]
    return



def sandwich_categorical(const int[:] indices, floating[:] d,
                        int[:] rows, dtype, int n_cols):
    cdef floating[:] res = np.zeros(n_cols, dtype=dtype)
    cdef int i, k, k_idx
    cdef int n_rows = len(rows)

    for k_idx in range(n_rows):
        k = rows[k_idx]
        i = indices[k]
        res[i] += d[k]
    return np.asarray(res)

