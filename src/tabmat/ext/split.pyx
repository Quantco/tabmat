import numpy as np

cimport numpy as np
from cython cimport floating, integral
from libc.stdlib cimport free, malloc
from libcpp cimport bool
from libcpp.vector cimport vector


# This is necessary because it's quite difficult to have a dynamic array of
# memoryviews in Cython
# https://stackoverflow.com/questions/56915835/how-to-have-a-list-of-memory-views-in-cython
cdef struct ArrayableMemoryView:
    Py_ssize_t* data
    Py_ssize_t length
ctypedef np.uint8_t uint8
ctypedef np.int8_t int8
ctypedef fused win_integral:
    integral
    long long


cdef extern from "cat_split_helpers.cpp":
    void _sandwich_cat_denseC[Int, F](F*, Int*, Int*, Int, Int*, Int, F*, Int, F*, Int, Int) nogil
    void _sandwich_cat_denseF[Int, F](F*, Int*, Int*, Int, Int*, Int, F*, Int, F*, Int, Int) nogil
    void _sandwich_cat_cat[Int, F](F*, const Int*, const Int*, Int*, Int, F*, Int, Int, bool, bool)


def sandwich_cat_dense(
    int[:] i_indices,
    int i_ncol,
    floating[:] d,
    np.ndarray mat_j,
    int[:] rows,
    int[:] j_cols,
    bool is_c_contiguous
):
    cdef int nj_rows = mat_j.shape[0]
    cdef int nj_cols = mat_j.shape[1]
    cdef int n_active_rows = len(rows)
    cdef int nj_active_cols = len(j_cols)
    cdef floating[:, :] res
    res = np.zeros((i_ncol, nj_active_cols), dtype=mat_j.dtype)
    cdef int res_size = res.size

    if d.shape[0] == 0 or n_active_rows == 0 or nj_active_cols == 0 or i_ncol == 0:
        return np.asarray(res)

    cdef floating* d_p = &d[0]
    cdef int* i_indices_p = &i_indices[0]
    cdef int* rows_p = &rows[0]
    cdef int* j_cols_p = &j_cols[0]

    cdef floating* mat_j_p = <floating*>mat_j.data

    if is_c_contiguous:
        _sandwich_cat_denseC(d_p, i_indices_p, rows_p, n_active_rows, j_cols_p,
                            nj_active_cols, &res[0, 0], res_size, mat_j_p,
                            nj_rows, nj_cols)
    else:
        _sandwich_cat_denseF(d_p, i_indices_p, rows_p, n_active_rows, j_cols_p,
                            nj_active_cols, &res[0, 0], res_size, mat_j_p,
                            nj_rows, nj_cols)

    return np.asarray(res)


def sandwich_cat_cat(
    int[:] i_indices,
    int[:] j_indices,
    int i_ncol,
    int j_ncol,
    floating[:] d,
    int[:] rows,
    dtype,
    bint i_drop_first,
    bint j_drop_first
):
    """
    (X1.T @ diag(d) @ X2)[i, j] = sum_k X1[k, i] d[k] X2[k, j]
    """
    cdef floating[:, :] res = np.zeros((i_ncol, j_ncol), dtype=dtype)
    cdef int n_rows = len(rows)
    cdef int res_size = res.size

    _sandwich_cat_cat(&d[0], &i_indices[0], &j_indices[0], &rows[0], n_rows,
                        &res[0, 0], j_ncol, res_size, i_drop_first, j_drop_first)

    return np.asarray(res)


# This seems slower, so not using it for now
def _sandwich_cat_cat_limited_rows_cols(
    int[:] i_indices,
    int[:] j_indices,
    int i_ncol,
    int j_ncol,
    floating[:] d,
    int[:] rows,
    int[:] i_cols,
    int[:] j_cols
):
    """
    (X1.T @ diag(d) @ X2)[i, j] = sum_k X1[k, i] d[k] X2[k, j]
    """

    # TODO: support for single-precision d
    # TODO: this is writing an output of the wrong shape; filtering on rows
    # and cols still needs to happen after
    # TODO: Look into sparse matrix multiplication algorithms. Should one or both be csc?

    cdef floating[:, :] res
    res = np.zeros((i_ncol, j_ncol))
    cdef size_t k_idx, k, i, j

    cdef uint8[:] i_col_included = np.zeros(i_ncol, dtype=np.uint8)
    for Ci in range(i_ncol):
        i_col_included[i_cols[Ci]] = 1

    cdef uint8[:] j_col_included = np.zeros(j_ncol, dtype=np.uint8)
    for Ci in range(j_ncol):
        j_col_included[j_cols[Ci]] = 1

    for k_idx in range(len(rows)):
        k = rows[k_idx]
        i = i_indices[k]
        if i_col_included[i]:
            j = j_indices[k]
            if j_col_included[j]:
                res[i, j] += d[k]

    return np.asarray(res)


def split_col_subsets(self, int[:] cols):
    cdef int[:] next_subset_idx = np.zeros(len(self.indices), dtype=np.int32)
    cdef vector[vector[int]] subset_cols_indices
    cdef vector[vector[int]] subset_cols
    cdef vector[int] empty = []

    cdef int j
    cdef int n_matrices = len(self.indices)
    cdef ArrayableMemoryView* indices_arrs = <ArrayableMemoryView*> malloc(
        sizeof(ArrayableMemoryView) * n_matrices
    );
    cdef Py_ssize_t[:] this_idx_view
    for j in range(n_matrices):
        this_idx_view = self.indices[j]
        indices_arrs[j].length = len(this_idx_view)
        if indices_arrs[j].length > 0:
            indices_arrs[j].data = &(this_idx_view[0])
        subset_cols_indices.push_back(empty)
        subset_cols.push_back(empty)

    cdef int i
    cdef int n_cols = cols.shape[0]

    for i in range(n_cols):
        for j in range(n_matrices):

            while (
                next_subset_idx[j] < indices_arrs[j].length
                and indices_arrs[j].data[next_subset_idx[j]] < cols[i]
            ):
                next_subset_idx[j] += 1

            if (
                next_subset_idx[j] < indices_arrs[j].length
                and indices_arrs[j].data[next_subset_idx[j]] == cols[i]
            ):
                subset_cols_indices[j].push_back(i)
                subset_cols[j].push_back(next_subset_idx[j])
                next_subset_idx[j] += 1
                break

    free(indices_arrs)
    return (
        [
            np.array(subset_cols_indices[j], dtype=np.int32)
            for j in range(n_matrices)
        ],
        [
            np.array(subset_cols[j], dtype=np.int32)
            for j in range(n_matrices)
        ],
        n_cols
    )

def is_sorted(win_integral[:] a):
    cdef win_integral* a_ptr = &a[0]
    cdef win_integral i
    for i in range(a.size - 1):
        if a_ptr[i + 1] < a_ptr[i]:
            return False
    return True
