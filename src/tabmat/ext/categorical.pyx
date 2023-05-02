import numpy as np

cimport numpy as np
from cython cimport floating, numeric

from cython.parallel import prange

ctypedef np.uint8_t uint8
ctypedef np.int8_t int8
from libcpp cimport bool


cdef extern from "cat_split_helpers.cpp":
    void _transpose_matvec_all_rows[Int, F](Int, Int*, F*, F*, Int)
    void _transpose_matvec_all_rows_drop_first[Int, F](Int, Int*, F*, F*, Int)


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
    cdef int out_size = out.size
    cdef int[:] rows_view, cols_included

    cdef bool no_row_restrictions = rows is None or len(rows) == n_rows
    cdef bool no_col_restrictions = cols is None or len(cols) == n_cols

    # Case 1: No row or col restrictions
    if no_row_restrictions and no_col_restrictions:
        _transpose_matvec_all_rows(n_rows, &indices[0], &other[0], &out[0], out_size)
    # Case 2: row restrictions but no col restrictions
    elif no_col_restrictions:
        rows_view = rows
        n_keep_rows = len(rows_view)
        for row_idx in range(n_keep_rows):
            row = rows_view[row_idx]
            out[indices[row]] += other[row]
    # Cases 3 and 4: col restrictions
    else:
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
    cdef int out_size = out.size
    cdef int[:] rows_view, cols_included

    cdef bool no_row_restrictions = rows is None or len(rows) == n_rows
    cdef bool no_col_restrictions = cols is None or len(cols) == n_cols

    # Case 1: No row or col restrictions
    if no_row_restrictions and no_col_restrictions:
        _transpose_matvec_all_rows_drop_first(n_rows, &indices[0], &other[0], &out[0], out_size)
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
    cdef int i, col_idx
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
    cdef int i, col_idx
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


def multiply_drop_first(
    int[:] indices,
    numeric[:] d,
    int ncols,
    dtype,
):
    """Multiply a CategoricalMatrix by a vector d.

    The output cannot be a CategoricalMatrix anymore. Here
    we return the inputs to transform to a csr_matrix.

    Note that *_drop_first function assume the CategoricalMatrix
    has its first category dropped.

    Parameters
    ----------
    indices:
        The vector of categories
    d:
        The vector to multiply with
    ncols:
        The number of columns
    dtype:
        Data type of d

    Returns
    -------
    Tuple with:
        - new data
        - indices of nonzero elements
        - indptr
    """
    cdef:
        int nrows = len(indices)
        int nonref_cnt = 0
        Py_ssize_t i
        np.ndarray new_data = np.empty(nrows, dtype=dtype)
        np.ndarray new_indices = np.empty(nrows, dtype=np.int32)
        np.ndarray new_indptr = np.empty(nrows + 1, dtype=np.int32)
        numeric[:] vnew_data = new_data
        int[:] vnew_indices = new_indices
        int[:] vnew_indptr = new_indptr

    for i in range(nrows):
        vnew_indptr[i] = nonref_cnt
        if indices[i] != 0:
            vnew_data[nonref_cnt] = d[i]
            vnew_indices[nonref_cnt] = indices[i] - 1
            nonref_cnt += 1

    vnew_indptr[i+1] = nonref_cnt

    return new_data[:nonref_cnt], new_indices[:nonref_cnt], new_indptr


def subset_categorical_drop_first(
    int[:] indices,
    int ncols,
):
    """Construct the inputs to transform a CategoricalMatrix into a csr_matrix.

    Note that it is assumed here that we drop the first category of the
    CategoricalMatrix.

    Parameters
    ----------
    indices:
        The vector of categories
    ncols:
        Total number of columns (# of categories - 1)

    Returns
    -------
    Tuple with:
        - number of nonzero elements
        - indices of nonzero elements
        - indptr
    """
    cdef:
        int nrows = len(indices)
        int nonzero_cnt = 0
        Py_ssize_t i
        np.ndarray new_indices = np.empty(nrows, dtype=np.int32)
        np.ndarray new_indptr = np.empty(nrows + 1, dtype=np.int32)
        int[:] vnew_indices = new_indices
        int[:] vnew_indptr = new_indptr

    for i in range(nrows):
        vnew_indptr[i] = nonzero_cnt
        if indices[i] != 0:
            vnew_indices[nonzero_cnt] = indices[i] - 1
            nonzero_cnt += 1

    vnew_indptr[i+1] = nonzero_cnt

    return nonzero_cnt, new_indices[:nonzero_cnt], new_indptr
