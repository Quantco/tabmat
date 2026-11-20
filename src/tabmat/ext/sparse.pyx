import numpy as np
cimport numpy as np

from cython cimport floating, integral
from cython.parallel import prange
from libc.stdint cimport int64_t

from .dense import _check_center


np.import_array()

ctypedef np.uint8_t uint8

ctypedef fused win_integral:
    integral
    long long

def sparse_sandwich(A, AT, const floating[::1] d, const win_integral[::1] rows, const win_integral[::1] cols):
    # AT is CSC
    # A is CSC
    # Computes AT @ diag(d) @ A

    cdef const floating[::1] Adata = A.data
    cdef const win_integral[::1] Aindices = A.indices
    cdef const win_integral[::1] Aindptr = A.indptr

    cdef const floating[::1] ATdata = AT.data
    cdef const win_integral[::1] ATindices = AT.indices
    cdef const win_integral[::1] ATindptr = AT.indptr

    cdef floating* Adatap = <floating*>&Adata[0]
    cdef win_integral* Aindicesp = <win_integral*>&Aindices[0]
    cdef floating* ATdatap = <floating*>&ATdata[0]
    cdef win_integral* ATindicesp = <win_integral*>&ATindices[0]
    cdef win_integral* ATindptrp = <win_integral*>&ATindptr[0]

    cdef floating* dp = <floating*>&d[0]

    cdef int64_t m = cols.shape[0]
    out = np.zeros((m, m), dtype=A.dtype)
    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef int64_t AT_idx, A_idx, Ci, i, Cj, j, k
    cdef floating A_val, AT_val

    cdef uint8[:] row_included = np.zeros(d.shape[0], dtype=np.uint8)
    for Ci in range(rows.shape[0]):
        row_included[rows[Ci]] = True

    cdef int[:] col_map = np.full(A.shape[1], -1, dtype=np.int32)
    for Cj in range(m):
        col_map[cols[Cj]] = Cj

    #TODO: see what happens when we swap to having k as the outer loop here?
    for Cj in prange(m, nogil=True):
        j = cols[Cj]
        for A_idx in range(Aindptr[j], Aindptr[j+1]):
            k = Aindicesp[A_idx]
            if not row_included[k]:
                continue

            A_val = Adatap[A_idx] * dp[k]
            Ci = 0
            for AT_idx in range(ATindptrp[k], ATindptrp[k+1]):
                i = ATindicesp[AT_idx]
                if i > j:
                    break

                Ci = col_map[i]
                if Ci == -1:
                    continue

                AT_val = ATdatap[AT_idx]
                outp[Cj * m + Ci] = outp[Cj * m + Ci] + AT_val * A_val

    out += np.tril(out, -1).T
    return out

def csr_matvec_unrestricted(X, const floating[::1] v, out, const win_integral[::1] X_indices):
    cdef const floating[::1] Xdata = X.data
    cdef const win_integral[::1] Xindices = X.indices
    cdef const win_integral[::1] Xindptr = X.indptr

    if out is None:
        out = np.zeros(X.shape[0], dtype=X.dtype)
    cdef floating[::1] out_view = out;

    cdef floating* Xdatap = <floating*>&Xdata[0];
    cdef win_integral* Xindicesp = <win_integral*>&Xindices[0];
    cdef win_integral* Xindptrp = <win_integral*>&Xindptr[0];
    cdef floating* outp = &out_view[0];

    cdef int64_t i, X_idx, j
    cdef int64_t n = out.shape[0]
    cdef floating Xval, vval

    for i in prange(n, nogil=True):
        for X_idx in range(Xindptrp[i], Xindptrp[i+1]):
            j = Xindicesp[X_idx]
            Xval = Xdatap[X_idx]
            vval = v[j]
            outp[i] = outp[i] + Xval * vval;
    return out

def csr_matvec(
        X,
        const floating[::1] v,
        const win_integral[::1] rows,
        const win_integral[::1] cols
    ):
    cdef const floating[::1] Xdata = X.data
    cdef const win_integral[::1] Xindices = X.indices
    cdef const win_integral[::1] Xindptr = X.indptr

    cdef int64_t n = rows.shape[0]
    out = np.zeros(n, dtype=X.dtype)
    cdef floating[::1] out_view = out;

    cdef floating* Xdatap = <floating*>&Xdata[0];
    cdef win_integral* Xindicesp = <win_integral*>&Xindices[0];
    cdef win_integral* Xindptrp = <win_integral*>&Xindptr[0];
    cdef floating* outp = &out_view[0];

    cdef int64_t Ci, i, Cj, X_idx, j
    cdef floating Xval, vval

    cdef uint8[:] col_included = np.zeros(X.shape[1], dtype=np.uint8)
    for Cj in range(cols.shape[0]):
        col_included[cols[Cj]] = True

    for Ci in prange(n, nogil=True):
        i = rows[Ci]
        for X_idx in range(Xindptrp[i], Xindptrp[i+1]):
            j = Xindicesp[X_idx]
            if not col_included[j]:
                continue
            Xval = Xdatap[X_idx]
            vval = v[j]
            outp[Ci] = outp[Ci] + Xval * vval;
    return out

def csc_rmatvec_unrestricted(XT, const floating[::1] v, out, const win_integral[::1] XT_indices):
    cdef const floating[::1] XTdata = XT.data
    cdef const win_integral[::1] XTindices = XT.indices
    cdef const win_integral[::1] XTindptr = XT.indptr

    cdef int m = XT.shape[1]
    if out is None:
        out = np.zeros(m, dtype=XT.dtype)
    cdef floating[::1] out_view = out;

    cdef floating* XTdatap = <floating*>&XTdata[0];
    cdef win_integral* XTindicesp = <win_integral*>&XTindices[0];
    cdef win_integral* XTindptrp = <win_integral*>&XTindptr[0];
    cdef floating* outp = &out_view[0];

    cdef int64_t i, XT_idx, j
    cdef floating XTval, vval

    for j in prange(m, nogil=True):
        for XT_idx in range(XTindptrp[j], XTindptrp[j+1]):
            i = XTindicesp[XT_idx]
            XTval = XTdatap[XT_idx];
            vval = v[i]
            outp[j] = outp[j] + XTval * vval;
    return out

def csc_rmatvec(XT, const floating[::1] v, const win_integral[::1] rows, const win_integral[::1] cols):
    cdef const floating[::1] XTdata = XT.data
    cdef const win_integral[::1] XTindices = XT.indices
    cdef const win_integral[::1] XTindptr = XT.indptr

    cdef int64_t n = rows.shape[0]
    cdef int64_t m = cols.shape[0]
    out = np.zeros(m, dtype=XT.dtype)
    cdef floating[::1] out_view = out;

    cdef floating* XTdatap = <floating*>&XTdata[0];
    cdef win_integral* XTindicesp = <win_integral*>&XTindices[0];
    cdef win_integral* XTindptrp = <win_integral*>&XTindptr[0];
    cdef floating* outp = &out_view[0];

    cdef int64_t Ci, i, Cj, XT_idx, j
    cdef floating XTval, vval

    cdef uint8[:] row_included = np.zeros(XT.shape[0], dtype=np.uint8)
    for Ci in range(n):
        row_included[rows[Ci]] = True

    for Cj in prange(m, nogil=True):
        j = cols[Cj]
        for XT_idx in range(XTindptrp[j], XTindptrp[j+1]):
            i = XTindicesp[XT_idx]
            if not row_included[i]:
                continue
            XTval = XTdatap[XT_idx];
            vval = v[i]
            outp[Cj] = outp[Cj] + XTval * vval;
    return out

cdef extern from "sparse_helpers.hpp":
    # The C++ kernels take the center of B as const F*. It is declared as F* here
    # because older Cython versions (e.g. 3.2) cannot deduce F from a const F*.
    void _csr_denseC_sandwich[I, F](
        F*, I*, I*, F*, F*, F*, F*, I, I, I,
        I*, I*, I*, I, I, I
    ) nogil
    void _csr_denseF_sandwich[I, F](
        F*, I*, I*, F*, F*, F*, F*, I, I, I,
        I*, I*, I*, I, I, I
    ) nogil

def csr_dense_sandwich(
        A,
        np.ndarray B,
        const floating[::1] d,
        const win_integral[::1] rows,
        const win_integral[::1] A_cols,
        const win_integral[::1] B_cols,
        B_center=None
    ):
    # computes where (A.T * d) @ (B - B_center)
    # assumes that A is in csr form; B_center=None means no centering
    cdef const floating[::1] B_mu = _check_center(B_center, B.shape[1], B.dtype, "B_center")

    cdef const floating[::1] Adata = A.data
    cdef const win_integral[::1] Aindices = A.indices
    cdef const win_integral[::1] Aindptr = A.indptr

    # A has shape (n, m)
    # B has shape (n, r)
    cdef win_integral m = A.shape[1]
    cdef win_integral n = d.shape[0]
    cdef win_integral r = B.shape[1]

    cdef win_integral nr = rows.shape[0]
    cdef win_integral nAc = A_cols.shape[0]
    cdef win_integral nBc = B_cols.shape[0]

    out = np.zeros((nAc, nBc), dtype=A.dtype)
    if nr == 0 or nAc == 0 or nBc == 0 or (Aindptr[A.indptr.shape[0] - 1] - Aindptr[0]) == 0:
        return out

    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef floating* Bp = <floating*>B.data

    cdef win_integral* rowsp = <win_integral*>&rows[0];
    cdef win_integral* A_colsp = <win_integral*>&A_cols[0];
    cdef win_integral* B_colsp = <win_integral*>&B_cols[0];
    cdef floating* B_mup = NULL
    if B_mu is not None:
        B_mup = <floating*>&B_mu[0]

    if B.flags["C_CONTIGUOUS"]:
        _csr_denseC_sandwich(
            <floating*>&Adata[0], <win_integral*>&Aindices[0], <win_integral*>&Aindptr[0], Bp, B_mup, <floating*>&d[0], outp, m, n, r,
            rowsp, A_colsp, B_colsp, nr, nAc, nBc
        )
    elif B.flags["F_CONTIGUOUS"]:
        _csr_denseF_sandwich(
            <floating*>&Adata[0], <win_integral*>&Aindices[0], <win_integral*>&Aindptr[0], Bp, B_mup, <floating*>&d[0], outp, m, n, r,
            rowsp, A_colsp, B_colsp, nr, nAc, nBc
        )
    else:
        raise Exception()
    return out

def transpose_square_dot_weights(
        const floating[:] data,
        const win_integral[:] indices,
        const win_integral[:] indptr,
        const floating[:] weights,
        dtype):

    cdef int ncols = indptr.shape[0] - 1

    cdef int i, j, k

    cdef np.ndarray out = np.zeros(ncols, dtype=dtype)
    cdef floating* outp = <floating*>out.data

    cdef floating v
    for j in prange(ncols, nogil=True):
        for k in range(indptr[j], indptr[j+1]):
            i = indices[k]
            v = data[k]
            outp[j] = outp[j] + weights[i] * (v ** 2)
    return out
