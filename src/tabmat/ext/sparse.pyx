# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

import cython
from cython cimport floating, integral
from cython.parallel import prange
from libc.stdint cimport int64_t

cimport openmp
from scipy import sparse

ctypedef np.uint8_t uint8

ctypedef fused win_integral:
    integral
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sandwich(A, AT, floating[:] d, win_integral[:] rows, win_integral[:] cols):
    # AT is CSC
    # A is CSC
    # Computes AT @ diag(d) @ A

    cdef floating[:] Adata = A.data
    cdef win_integral[:] Aindices = A.indices
    cdef win_integral[:] Aindptr = A.indptr

    cdef floating[:] ATdata = AT.data
    cdef win_integral[:] ATindices = AT.indices
    cdef win_integral[:] ATindptr = AT.indptr

    cdef floating* Adatap = &Adata[0]
    cdef win_integral* Aindicesp = &Aindices[0]
    cdef floating* ATdatap = &ATdata[0]
    cdef win_integral* ATindicesp = &ATindices[0]
    cdef win_integral* ATindptrp = &ATindptr[0]

    cdef floating* dp = &d[0]

    cdef int64_t m = cols.shape[0]
    out = np.zeros((m, m), dtype=A.dtype)
    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef int64_t AT_idx, A_idx, AT_row, A_col, Ci, i, Cj, j, Ck, k
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
        Ck = 0 # A2: this value is never used; consider removing
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

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sandwich_coo_serial(A, AT, floating[:] d, win_integral[:] rows, win_integral[:] cols):
    # Does same calculation as sparse_sandwich(), but returns the result in COO form
    # (i.e. three arrays: data, rows, columns)

    # indices is array of row indices
    # data is array of corresponding nonzero values
    # indptr points to column starts in indices and data
    cdef floating[:] Adata = A.data
    cdef win_integral[:] Aindices = A.indices
    cdef win_integral[:] Aindptr = A.indptr

    cdef floating[:] ATdata = AT.data
    cdef win_integral[:] ATindices = AT.indices
    cdef win_integral[:] ATindptr = AT.indptr

    cdef floating* Adatap = &Adata[0]
    cdef win_integral* Aindicesp = &Aindices[0]
    cdef floating* ATdatap = &ATdata[0]
    cdef win_integral* ATindicesp = &ATindices[0]
    cdef win_integral* ATindptrp = &ATindptr[0]

    cdef floating* dp = &d[0]

    cdef int64_t m = cols.shape[0]

    # Is there a way to adaptively determine how long the COO arrays should be?
    # For wide or dense enough problems, the COO arrays will not fit in memory
    cdef int64_t out_len = 100000000

    out_data = np.zeros(out_len, dtype=A.dtype)  
    cdef floating[:] out_data_view = out_data
    cdef floating* out_data_p = &out_data_view[0]
    out_rows = np.zeros(out_len, dtype=np.int32)
    cdef win_integral[:] out_rows_view = out_rows
    cdef win_integral* out_rows_p = &out_rows_view[0]
    out_cols = np.zeros(out_len, dtype=np.int32)
    cdef win_integral[:] out_cols_view = out_cols
    cdef win_integral* out_cols_p = &out_cols_view[0]

    cdef int64_t AT_idx, A_idx, AT_row, A_col, Ci, i, Cj, j, Ck, k
    cdef floating A_val, AT_val

    cdef uint8[:] row_included = np.zeros(d.shape[0], dtype=np.uint8)
    for Ci in range(rows.shape[0]):
        row_included[rows[Ci]] = True

    cdef int[:] col_map = np.full(A.shape[1], -1, dtype=np.int32)
    for Cj in range(m):
        col_map[cols[Cj]] = Cj

    cdef int64_t out_idx = 0

    for Cj in range(m):  # had to switch to serial due to out_idx incrementing
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

                out_data_p[out_idx] = AT_val * A_val
                out_rows_p[out_idx] = Cj
                out_cols_p[out_idx] = Ci

                out_idx = out_idx + 1

    return out_data, out_rows, out_cols

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sandwich_coo_parallel(A, AT, floating[:] d, win_integral[:] rows, win_integral[:] cols):
    # Does same calculation as sparse_sandwich_coo_serial(), but in parallel
    # For some reason, it runs slower than the serial version

    cdef floating[:] Adata = A.data
    cdef win_integral[:] Aindices = A.indices
    cdef win_integral[:] Aindptr = A.indptr

    cdef floating[:] ATdata = AT.data
    cdef win_integral[:] ATindices = AT.indices
    cdef win_integral[:] ATindptr = AT.indptr

    cdef floating* Adatap = &Adata[0]
    cdef win_integral* Aindicesp = &Aindices[0]
    cdef floating* ATdatap = &ATdata[0]
    cdef win_integral* ATindicesp = &ATindices[0]
    cdef win_integral* ATindptrp = &ATindptr[0]

    cdef floating* dp = &d[0]

    cdef int64_t m = cols.shape[0]

    cdef int64_t out_len = 100000000

    out_data = np.zeros(out_len, dtype=A.dtype)  
    cdef floating[:] out_data_view = out_data
    cdef floating* out_data_p = &out_data_view[0]
    out_rows = np.zeros(out_len, dtype=np.int32)
    cdef win_integral[:] out_rows_view = out_rows
    cdef win_integral* out_rows_p = &out_rows_view[0]
    out_cols = np.zeros(out_len, dtype=np.int32)
    cdef win_integral[:] out_cols_view = out_cols
    cdef win_integral* out_cols_p = &out_cols_view[0]

    cdef int64_t AT_idx, A_idx, AT_row, A_col, Ci, i, Cj, j, Ck, k
    cdef floating A_val, AT_val

    cdef uint8[:] row_included = np.zeros(d.shape[0], dtype=np.uint8)
    for Ci in range(rows.shape[0]):
        row_included[rows[Ci]] = True

    cdef int[:] col_map = np.full(A.shape[1], -1, dtype=np.int32)
    for Cj in range(m):
        col_map[cols[Cj]] = Cj

    # each thread will have its own index counter, so no conflicts
    cdef int64_t idx_start
    out_idxs = np.zeros(openmp.omp_get_max_threads(), dtype=np.int32)  
    cdef win_integral[:] out_idxs_view = out_idxs
    cdef win_integral* out_idxs_p = &out_idxs_view[0]

    cdef int64_t cur_thread_num

    for Cj in prange(m, nogil=True):
        cur_thread_num = openmp.omp_get_thread_num()
        idx_start = cur_thread_num * out_len // openmp.omp_get_num_threads()

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

                out_data_p[idx_start + out_idxs_p[cur_thread_num]] = AT_val * A_val
                out_rows_p[idx_start + out_idxs_p[cur_thread_num]] = Cj
                out_cols_p[idx_start + out_idxs_p[cur_thread_num]] = Ci

                out_idxs_p[cur_thread_num] = out_idxs_p[cur_thread_num] + 1

    return out_data, out_rows, out_cols

@cython.boundscheck(False)
@cython.wraparound(False)
def csr_matvec_unrestricted(X, floating[:] v, out, win_integral[:] X_indices):
    cdef floating[:] Xdata = X.data
    cdef win_integral[:] Xindices = X.indices
    cdef win_integral[:] Xindptr = X.indptr

    if out is None:
        out = np.zeros(X.shape[0], dtype=X.dtype)
    cdef floating[:] out_view = out;

    cdef floating* Xdatap = &Xdata[0];
    cdef win_integral* Xindicesp = &Xindices[0];
    cdef win_integral* Xindptrp = &Xindptr[0];
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

@cython.boundscheck(False)
@cython.wraparound(False)
def csr_matvec(
        X,
        floating[:] v,
        win_integral[:] rows,
        win_integral[:] cols
    ):
    cdef floating[:] Xdata = X.data
    cdef win_integral[:] Xindices = X.indices
    cdef win_integral[:] Xindptr = X.indptr

    cdef int64_t n = rows.shape[0]
    cdef int64_t m = cols.shape[0]
    out = np.zeros(n, dtype=X.dtype)
    cdef floating[:] out_view = out;

    cdef floating* Xdatap = &Xdata[0];
    cdef win_integral* Xindicesp = &Xindices[0];
    cdef win_integral* Xindptrp = &Xindptr[0];
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

@cython.boundscheck(False)
@cython.wraparound(False)
def csc_rmatvec_unrestricted(XT, floating[:] v, out, win_integral[:] XT_indices):
    cdef floating[:] XTdata = XT.data
    cdef win_integral[:] XTindices = XT.indices
    cdef win_integral[:] XTindptr = XT.indptr

    cdef int m = XT.shape[1]
    if out is None:
        out = np.zeros(m, dtype=XT.dtype)
    cdef floating[:] out_view = out;

    cdef floating* XTdatap = &XTdata[0];
    cdef win_integral* XTindicesp = &XTindices[0];
    cdef win_integral* XTindptrp = &XTindptr[0];
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

@cython.boundscheck(False)
@cython.wraparound(False)
def csc_rmatvec(XT, floating[:] v, win_integral[:] rows, win_integral[:] cols):
    cdef floating[:] XTdata = XT.data
    cdef win_integral[:] XTindices = XT.indices
    cdef win_integral[:] XTindptr = XT.indptr

    cdef int64_t n = rows.shape[0]
    cdef int64_t m = cols.shape[0]
    out = np.zeros(m, dtype=XT.dtype)
    cdef floating[:] out_view = out;

    cdef floating* XTdatap = &XTdata[0];
    cdef win_integral* XTindicesp = &XTindices[0];
    cdef win_integral* XTindptrp = &XTindptr[0];
    cdef floating* outp = &out_view[0];
    cdef win_integral* rowsp
    cdef win_integral* colsp

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

cdef extern from "sparse_helpers.cpp":
    void _csr_denseC_sandwich[I, F](
        F*, I*, I*, F*, F*, F*, I, I, I,
        I*, I*, I*, I, I, I
    ) nogil
    void _csr_denseF_sandwich[I, F](
        F*, I*, I*, F*, F*, F*, I, I, I,
        I*, I*, I*, I, I, I
    ) nogil

def csr_dense_sandwich(
        A,
        np.ndarray B,
        floating[:] d,
        win_integral[:] rows,
        win_integral[:] A_cols,
        win_integral[:] B_cols
    ):
    # computes where (A.T * d) @ B
    # assumes that A is in csr form
    cdef floating[:] Adata = A.data
    cdef win_integral[:] Aindices = A.indices
    cdef win_integral[:] Aindptr = A.indptr

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

    cdef win_integral* rowsp = &rows[0];
    cdef win_integral* A_colsp = &A_cols[0];
    cdef win_integral* B_colsp = &B_cols[0];

    if B.flags['C_CONTIGUOUS']:
        _csr_denseC_sandwich(
            &Adata[0], &Aindices[0], &Aindptr[0], Bp, &d[0], outp, m, n, r,
            rowsp, A_colsp, B_colsp, nr, nAc, nBc
        )
    elif B.flags['F_CONTIGUOUS']:
        _csr_denseF_sandwich(
            &Adata[0], &Aindices[0], &Aindptr[0], Bp, &d[0], outp, m, n, r,
            rowsp, A_colsp, B_colsp, nr, nAc, nBc
        )
    else:
        raise Exception()
    return out

def transpose_square_dot_weights(
        floating[:] data,
        win_integral[:] indices,
        win_integral[:] indptr,
        floating[:] weights,
        dtype):

    cdef int nrows = weights.shape[0]
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
