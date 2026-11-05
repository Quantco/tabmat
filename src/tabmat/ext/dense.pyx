import numpy as np
cimport numpy as np

from cython cimport floating
from cython.parallel import prange
from libc.stdint cimport int64_t

np.import_array()


cdef extern from "dense_helpers.cpp":
    # The C++ kernels take the center as const F*. It is declared as F* here
    # because older Cython versions (e.g. 3.2) cannot deduce F from a const F*.
    void _denseC_sandwich[Int, F](Int*, Int*, F*, F*, F*, F*, Int, Int, Int, Int, Int, Int, Int) nogil
    void _denseF_sandwich[Int, F](Int*, Int*, F*, F*, F*, F*, Int, Int, Int, Int, Int, Int, Int) nogil
    void _denseC_rmatvec[Int, F](Int*, Int*, F*, F*, F*, F*, Int, Int, Int, Int) nogil
    void _denseF_rmatvec[Int, F](Int*, Int*, F*, F*, F*, F*, Int, Int, Int, Int) nogil
    void _denseC_matvec[Int, F](Int*, Int*, F*, F*, F*, F*, Int, Int, Int, Int) nogil
    void _denseF_matvec[Int, F](Int*, Int*, F*, F*, F*, F*, Int, Int, Int, Int) nogil

def _check_center(center, Py_ssize_t m, dtype, name="center", bint zeros=False):
    """Return ``center`` as a contiguous array of length ``m``.

    ``None`` stays ``None``, or becomes zeros if ``zeros``. The dense kernels
    always subtract a center, which costs nothing there, and ``x - 0 == x``
    exactly. The cross kernels skip the subtraction for a null center, because
    it would slow down their uncentered packing.
    """
    if center is None:
        return np.zeros(m, dtype=dtype) if zeros else None
    center = np.ascontiguousarray(center, dtype=dtype)
    if center.shape[0] != m:
        raise ValueError(f"{name} must have length {m}, got {center.shape[0]}.")
    return center


def dense_sandwich(np.ndarray X, floating[:] d, int[:] rows, int[:] cols, int thresh1d = 32, int kratio = 16, int innerblock = 128, center=None):
    """(X[rows, cols] - center[cols]).T @ diag(d[rows]) @ (X[rows, cols] - center[cols]).

    ``center`` is an optional length-``X.shape[1]`` vector (indexed by the
    original column index, like ``X``); ``None`` means no centering.
    """
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int in_n = rows.shape[0]
    cdef int out_m = cols.shape[0]

    # A const view, so that a read-only center works too. The kernel does not
    # write to it, so casting the const away for the call below is safe.
    cdef const floating[::1] mu = _check_center(center, m, X.dtype, zeros=True)

    out = np.zeros((out_m,out_m), dtype=X.dtype)
    if in_n == 0 or out_m == 0:
        return out

    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef floating* Xp = <floating*>X.data
    cdef floating* dp = &d[0]

    cdef int* colsp = &cols[0]
    cdef int* rowsp = &rows[0]
    cdef floating* mup = <floating*>&mu[0]

    if X.flags["C_CONTIGUOUS"]:
        _denseC_sandwich(rowsp, colsp, Xp, dp, mup, outp, in_n, out_m, m, n, thresh1d, kratio, innerblock)
    elif X.flags["F_CONTIGUOUS"]:
        _denseF_sandwich(rowsp, colsp, Xp, dp, mup, outp, in_n, out_m, m, n, thresh1d, kratio, innerblock)
    else:
        raise Exception("The matrix X is not contiguous.")
    return out


def dense_rmatvec(
    np.ndarray X, const floating[::1] v, int[:] rows=None, int[:] cols=None, center=None
):
    """(X[rows, cols] - center[cols]).T @ v[rows].

    ``rows=None`` and ``cols=None`` select all rows and columns, which lets the
    kernel vectorize along them; ``center=None`` means no centering.
    """
    return _dense_matvec_impl(X, v, rows, cols, center, True)


def dense_matvec(
    np.ndarray X, const floating[::1] v, int[:] rows=None, int[:] cols=None, center=None
):
    """(X[rows, cols] - center[cols]) @ v[cols], with the same conventions as
    :func:`dense_rmatvec`.
    """
    return _dense_matvec_impl(X, v, rows, cols, center, False)


cdef _dense_matvec_impl(
    np.ndarray X, const floating[::1] v, int[:] rows, int[:] cols, center, bint transpose
):
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int n_rows = n if rows is None else rows.shape[0]
    cdef int n_cols = m if cols is None else cols.shape[0]
    cdef const floating[::1] mu = _check_center(center, m, X.dtype, zeros=True)

    out = np.zeros(n_cols if transpose else n_rows, dtype=X.dtype)
    if n_rows == 0 or n_cols == 0:
        return out

    cdef floating[:] out_view = out
    cdef floating* outp = &out_view[0]

    cdef floating* Xp = <floating*>X.data
    cdef floating* vp = <floating*>&v[0]
    cdef floating* mup = <floating*>&mu[0]

    cdef int* rowsp = NULL
    cdef int* colsp = NULL
    if rows is not None:
        rowsp = &rows[0]
    if cols is not None:
        colsp = &cols[0]

    if X.flags["C_CONTIGUOUS"]:
        if transpose:
            _denseC_rmatvec(rowsp, colsp, Xp, vp, mup, outp, n_rows, n_cols, m, n)
        else:
            _denseC_matvec(rowsp, colsp, Xp, vp, mup, outp, n_rows, n_cols, m, n)
    elif X.flags["F_CONTIGUOUS"]:
        if transpose:
            _denseF_rmatvec(rowsp, colsp, Xp, vp, mup, outp, n_rows, n_cols, m, n)
        else:
            _denseF_matvec(rowsp, colsp, Xp, vp, mup, outp, n_rows, n_cols, m, n)
    else:
        raise Exception("The matrix X is not contiguous.")
    return out

def transpose_square_dot_weights(np.ndarray X, floating[:] weights, floating[:] shift):
    cdef floating* Xp = <floating*>X.data
    cdef int nrows = weights.shape[0]
    cdef int ncols = X.shape[1]
    cdef int64_t i, j

    cdef np.ndarray out = np.zeros(ncols, dtype=X.dtype)
    cdef floating* outp = <floating*>out.data

    if X.flags["C_CONTIGUOUS"]:
        for j in prange(ncols, nogil=True):
            for i in range(nrows):
                outp[j] = outp[j] + weights[i] * ((Xp[i * ncols + j] - shift[j]) ** 2)
    elif X.flags["F_CONTIGUOUS"]:
        for j in prange(ncols, nogil=True):
            for i in range(nrows):
                outp[j] = outp[j] + weights[i] * ((Xp[j * nrows + i] - shift[j]) ** 2)
    else:
        raise Exception("The matrix X is not contiguous.")
    return out
