#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>

#include <xsimd/xsimd.hpp>

#include "alloc.h"
#include "order.h"

#ifdef _WIN32
    #define SIZE_T long long
#else
    #define SIZE_T size_t
#endif

#if XSIMD_VERSION_MAJOR >= 8
    #define XSIMD_BROADCAST broadcast
#else
    #define XSIMD_BROADCAST set_simd
#endif

namespace xs = xsimd;

// out[A_cols, B_cols] = A[rows, A_cols].T @ diag(d[rows]) @ (B - Bmu)[rows, B_cols]
//
// A is a CSR matrix with m columns, B is a dense n x r matrix in the given
// order, out is nA_cols x nB_cols, row-major, zeroed. The optional per-column
// center Bmu (nullptr means none) is subtracted while packing B.
//
// The rows are processed in kblock-sized blocks and the dense columns in
// jblock-sized blocks. For each block pair the d-scaled dense block is packed
// into a per-thread row-major panel R, then every nonzero A[k, i] of the
// block adds Q * R[k, :] into the thread's private copy of out row i (an axpy
// that vectorizes over the dense columns). The private copies are summed
// into out at the end.
template <Order order, typename Int, typename F>
void csr_dense_sandwich(
    const F* Adata, const Int* Aindices, const Int* Aindptr,
    const F* B, const F* Bmu, const F* d, F* out,
    Int m, Int n, Int r,
    const Int* rows, const Int* A_cols, const Int* B_cols,
    Int nrows, Int nA_cols, Int nB_cols
    )
{
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr Int simd_size = 1;
#else
    constexpr Int simd_size = xsimd::simd_type<F>::size;
#endif

    constexpr auto alignment = simd_size * sizeof(F);

    int kblock = 128;
    int jblock = 128;
    auto Rglobal = make_aligned_unique<F>(
        omp_get_max_threads() * kblock * jblock,
        alignment
    );

    std::vector<Int> Acol_map(m, -1);
    // Don't parallelize because the number of columns is small
    for (Py_ssize_t Ci = 0; Ci < nA_cols; Ci++) {
        Acol_map[A_cols[Ci]] = Ci;
    }

    #pragma omp parallel
    {
        Py_ssize_t nB_cols_rounded = ceil(((float)nB_cols) / ((float)simd_size)) * simd_size;
        auto outtemp = make_aligned_unique<F>(
            nA_cols * nB_cols_rounded,
            alignment
        );
        for (Py_ssize_t Ci = 0; Ci < nA_cols; Ci++) {
            for (Py_ssize_t Cj = 0; Cj < nB_cols; Cj++) {
                outtemp.get()[Ci * nB_cols_rounded + Cj] = 0.0;
            }
        }

        #pragma omp for
        for (Py_ssize_t Ckk = 0; Ckk < nrows; Ckk += kblock) {
            Py_ssize_t Ckmax = std::min<Py_ssize_t>(Ckk + kblock, nrows);
            for (Py_ssize_t Cjj = 0; Cjj < nB_cols; Cjj += jblock) {
                Py_ssize_t Cjmax = std::min<Py_ssize_t>(Cjj + jblock, nB_cols);

                // pack the d-scaled dense block, row-major
                F* R = &Rglobal.get()[omp_get_thread_num() * kblock * jblock];
                for (Py_ssize_t Ck = Ckk; Ck < Ckmax; Ck++) {
                    Int k = rows[Ck];
                    for (Py_ssize_t Cj = Cjj; Cj < Cjmax; Cj++) {
                        Int j = B_cols[Cj];
                        F Bv = load_X<order>(B, k, j, n, r);
                        if (Bmu != nullptr) {
                            Bv -= Bmu[j];
                        }
                        R[(Ck - Ckk) * jblock + (Cj - Cjj)] = d[k] * Bv;
                    }
                }

                for (Py_ssize_t Ck = Ckk; Ck < Ckmax; Ck++) {
                    Int k = rows[Ck];
                    for (Int A_idx = Aindptr[k]; A_idx < Aindptr[k + 1]; A_idx++) {
                        Int i = Aindices[A_idx];
                        Py_ssize_t Ci = Acol_map[i];
                        if (Ci == -1) {
                            continue;
                        }

                        F Q = Adata[A_idx];
                        F* outrow = &outtemp.get()[Ci * nB_cols_rounded];
                        const F* Rrow = &R[(Ck - Ckk) * jblock];
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
                        auto Qsimd = Q;
#else
                        auto Qsimd = xs::XSIMD_BROADCAST(Q);
#endif

                        Py_ssize_t Cj = Cjj;
                        Py_ssize_t Cjmax2 = Cjj + ((Cjmax - Cjj) / simd_size) * simd_size;
                        for (; Cj < Cjmax2; Cj += simd_size) {
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
                            auto Bsimd = Rrow[Cj - Cjj];
                            auto outsimd = outrow[Cj];
                            outrow[Cj] = xs::fma(Qsimd, Bsimd, outsimd);
#else
                            auto Bsimd = xs::load_aligned(&Rrow[Cj - Cjj]);
                            auto outsimd = xs::load_aligned(&outrow[Cj]);
                            outsimd = xs::fma(Qsimd, Bsimd, outsimd);
                            outsimd.store_aligned(&outrow[Cj]);
#endif
                        }

                        for (; Cj < Cjmax; Cj++) {
                            outrow[Cj] += Q * Rrow[Cj - Cjj];
                        }
                    }
                }
            }
        }

        for (Py_ssize_t Ci = 0; Ci < nA_cols; Ci++) {
            for (Py_ssize_t Cj = 0; Cj < nB_cols; Cj++) {
                #pragma omp atomic
                out[(Py_ssize_t) Ci * nB_cols + Cj] += outtemp.get()[Ci * nB_cols_rounded + Cj];
            }
        }
    }
}

template <typename Int, typename F>
void _csr_denseC_sandwich(
    F* Adata, Int* Aindices, Int* Aindptr,
    F* B, const F* Bmu, F* d, F* out,
    Int m, Int n, Int r,
    Int* rows, Int* A_cols, Int* B_cols,
    Int nrows, Int nA_cols, Int nB_cols
    )
{
    csr_dense_sandwich<Order::C>(
        Adata, Aindices, Aindptr, B, Bmu, d, out, m, n, r,
        rows, A_cols, B_cols, nrows, nA_cols, nB_cols
    );
}

template <typename Int, typename F>
void _csr_denseF_sandwich(
    F* Adata, Int* Aindices, Int* Aindptr,
    F* B, const F* Bmu, F* d, F* out,
    Int m, Int n, Int r,
    Int* rows, Int* A_cols, Int* B_cols,
    Int nrows, Int nA_cols, Int nB_cols
    )
{
    csr_dense_sandwich<Order::F>(
        Adata, Aindices, Aindptr, B, Bmu, d, out, m, n, r,
        rows, A_cols, B_cols, nrows, nA_cols, nB_cols
    );
}
