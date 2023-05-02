import numpy as np
cimport numpy as np

from cython cimport floating
from cython.parallel import prange
from libc.stdint cimport int64_t

cdef extern from "dense_helpers.cpp":
    void _denseC_sandwich[Int, F](Int*, Int*, F*, F*, F*, Int, Int, Int, Int, Int, Int, Int) nogil
    void _denseF_sandwich[Int, F](Int*, Int*, F*, F*, F*, Int, Int, Int, Int, Int, Int, Int) nogil
    void _denseC_rmatvec[Int, F](Int*, Int*, F*, F*, F*, Int, Int, Int, Int) nogil
    void _denseF_rmatvec[Int, F](Int*, Int*, F*, F*, F*, Int, Int, Int, Int) nogil
    void _denseC_matvec[Int, F](Int*, Int*, F*, F*, F*, Int, Int, Int, Int) nogil
    void _denseF_matvec[Int, F](Int*, Int*, F*, F*, F*, Int, Int, Int, Int) nogil

def dense_sandwich(np.ndarray X, floating[:] d, int[:] rows, int[:] cols, int thresh1d = 32, int kratio = 16, int innerblock = 128):
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int in_n = rows.shape[0]
    cdef int out_m = cols.shape[0]

    out = np.zeros((out_m,out_m), dtype=X.dtype)
    if in_n == 0 or out_m == 0:
        return out

    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef floating* Xp = <floating*>X.data
    cdef floating* dp = &d[0]

    cdef int* colsp = &cols[0]
    cdef int* rowsp = &rows[0]

    if X.flags["C_CONTIGUOUS"]:
        _denseC_sandwich(rowsp, colsp, Xp, dp, outp, in_n, out_m, m, n, thresh1d, kratio, innerblock)
    elif X.flags["F_CONTIGUOUS"]:
        _denseF_sandwich(rowsp, colsp, Xp, dp, outp, in_n, out_m, m, n, thresh1d, kratio, innerblock)
    else:
        raise Exception("The matrix X is not contiguous.")
    return out


# TODO: lots of duplicated code with dense_sandwich above
def dense_rmatvec(np.ndarray X, floating[:] v, int[:] rows, int[:] cols):
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int n_rows = rows.shape[0]
    cdef int n_cols = cols.shape[0]

    out = np.zeros(n_cols, dtype=X.dtype)
    if n_rows == 0 or n_cols == 0:
        return out

    cdef floating[:] out_view = out
    cdef floating* outp = &out_view[0]

    cdef floating* Xp = <floating*>X.data
    cdef floating* vp = &v[0]

    cdef int* colsp = &cols[0]
    cdef int* rowsp = &rows[0]

    if X.flags["C_CONTIGUOUS"]:
        _denseC_rmatvec(rowsp, colsp, Xp, vp, outp, n_rows, n_cols, m, n)
    elif X.flags["F_CONTIGUOUS"]:
        _denseF_rmatvec(rowsp, colsp, Xp, vp, outp, n_rows, n_cols, m, n)
    else:
        raise Exception("The matrix X is not contiguous.")
    return out

# TODO: lots of duplicated code with dense_sandwich above
def dense_matvec(np.ndarray X, floating[:] v, int[:] rows, int[:] cols):
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int n_rows = rows.shape[0]
    cdef int n_cols = cols.shape[0]

    out = np.zeros(n_rows, dtype=X.dtype)
    if n_rows == 0 or n_cols == 0:
        return out

    cdef floating[:] out_view = out
    cdef floating* outp = &out_view[0]

    cdef floating* Xp = <floating*>X.data
    cdef floating* vp = &v[0]

    cdef int* colsp = &cols[0]
    cdef int* rowsp = &rows[0]

    if X.flags["C_CONTIGUOUS"]:
        _denseC_matvec(rowsp, colsp, Xp, vp, outp, n_rows, n_cols, m, n)
    elif X.flags["F_CONTIGUOUS"]:
        _denseF_matvec(rowsp, colsp, Xp, vp, outp, n_rows, n_cols, m, n)
    else:
        raise Exception("The matrix X is not contiguous.")
    return out

def transpose_square_dot_weights(np.ndarray X, floating[:] weights):
    cdef floating* Xp = <floating*>X.data
    cdef int nrows = weights.shape[0]
    cdef int ncols = X.shape[1]
    cdef int64_t i, j

    cdef np.ndarray out = np.zeros(ncols, dtype=X.dtype)
    cdef floating* outp = <floating*>out.data

    if X.flags["C_CONTIGUOUS"]:
        for j in prange(ncols, nogil=True):
            for i in range(nrows):
                outp[j] = outp[j] + weights[i] * (Xp[i * ncols + j] ** 2)
    elif X.flags["F_CONTIGUOUS"]:
        for j in prange(ncols, nogil=True):
            for i in range(nrows):
                outp[j] = outp[j] + weights[i] * (Xp[j * nrows + i] ** 2)
    else:
        raise Exception("The matrix X is not contiguous.")
    return out
