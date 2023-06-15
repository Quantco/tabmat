#include <iostream>
#include <vector>
#include <omp.h>

#include <xsimd/xsimd.hpp>

#include "alloc.h"

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

<%def name="csr_dense_sandwich_tmpl(order)">
template <typename Int, typename F>
void _csr_dense${order}_sandwich(
    F* Adata, Int* Aindices, Int* Aindptr,
    F* B, F* d, F* out,
    Int m, Int n, Int r,
    Int* rows, Int* A_cols, Int* B_cols,
    Int nrows, Int nA_cols, Int nB_cols
    ) 
{
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr Int simd_size = 1;
#else
    constexpr Int simd_size = xsimd::simd_type<F>::size;
#endif

    constexpr auto alignment = simd_size*sizeof(F);

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
        for (Py_ssize_t Ckk = 0; Ckk < nrows; Ckk+=kblock) {
            Py_ssize_t Ckmax = Ckk + kblock;
            if (Ckmax > nrows) {
                Ckmax = nrows;
            }
            for (Py_ssize_t Cjj = 0; Cjj < nB_cols; Cjj+=jblock) {
                Py_ssize_t Cjmax = Cjj + jblock;
                if (Cjmax > nB_cols) {
                    Cjmax = nB_cols;
                }

                F* R = &Rglobal.get()[omp_get_thread_num()*kblock*jblock];
                for (Py_ssize_t Ck = Ckk; Ck < Ckmax; Ck++) {
                    Int k = rows[Ck];
                    for (Py_ssize_t Cj = Cjj; Cj < Cjmax; Cj++) {
                        Int j = B_cols[Cj];
                        %if order == 'C':
                            F Bv = B[(Py_ssize_t) k * r + j];
                        % else:
                            F Bv = B[(Py_ssize_t) j * n + k];
                        % endif
                        R[(Py_ssize_t) (Ck-Ckk) * jblock + (Cj-Cjj)] = d[k] * Bv;
                    }
                }

                for (Py_ssize_t Ck = Ckk; Ck < Ckmax; Ck++) {
                    Int k = rows[Ck];
                    for (Int A_idx = Aindptr[k]; A_idx < Aindptr[k+1]; A_idx++) {
                        Int i = Aindices[A_idx];
                        Py_ssize_t Ci = Acol_map[i];
                        if (Ci == -1) {
                            continue;
                        }

                        F Q = Adata[A_idx];
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
			auto Qsimd = Q;
#else
                        auto Qsimd = xs::XSIMD_BROADCAST(Q);
#endif

                        Py_ssize_t Cj = Cjj;
                        Py_ssize_t Cjmax2 = Cjj + ((Cjmax - Cjj) / simd_size) * simd_size;
                        for (; Cj < Cjmax2; Cj+=simd_size) {
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
                            auto Bsimd = R[(Py_ssize_t) (Ck-Ckk) * jblock + (Cj-Cjj)];
                            auto outsimd = outtemp.get()[Ci * nB_cols_rounded + Cj];
#else
                            auto Bsimd = xs::load_aligned(&R[(Py_ssize_t) (Ck-Ckk) * jblock + (Cj-Cjj)]);
                            auto outsimd = xs::load_aligned(&outtemp.get()[Ci * nB_cols_rounded + Cj]);
#endif
                            outsimd = xs::fma(Qsimd, Bsimd, outsimd);
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
                            outtemp.get()[Ci * nB_cols_rounded + Cj] = outsimd;
#else
                            outsimd.store_aligned(&outtemp.get()[Ci * nB_cols_rounded + Cj]);
#endif
                        }

                        for (; Cj < Cjmax; Cj++) {
                            outtemp.get()[Ci * nB_cols_rounded + Cj] += Q * R[(Py_ssize_t) (Ck-Ckk) * jblock + (Cj-Cjj)];
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
</%def>

${csr_dense_sandwich_tmpl('C')}
${csr_dense_sandwich_tmpl('F')}
