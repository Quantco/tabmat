#include <iostream>
#include <vector>
#include <omp.h>

#include <xsimd/xsimd.hpp>

#include "alloc.h"

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
    constexpr Int simd_size = xsimd::simd_type<F>::size;
    constexpr auto alignment = simd_size*sizeof(F);

    Int kblock = 128;
    Int jblock = 128;
    auto Rglobal = make_aligned_unique<F>(
        omp_get_max_threads() * kblock * jblock,
        alignment
    );

    std::vector<Int> Acol_map(m, -1);
    // Don't parallelize because the number of columns is small
    for (Int Ci = 0; Ci < nA_cols; Ci++) {
        Int i = A_cols[Ci];
        Acol_map[i] = Ci;
    }

    #pragma omp parallel
    {
        Int nB_cols_rounded = ceil(((float)nB_cols) / ((float)simd_size)) * simd_size;
        auto outtemp = make_aligned_unique<F>(
            nA_cols * nB_cols_rounded,
            alignment
        );
        for (Int Ci = 0; Ci < nA_cols; Ci++) {
            for (Int Cj = 0; Cj < nB_cols; Cj++) {
                outtemp.get()[Ci*nB_cols_rounded+Cj] = 0.0;
            }
        }

        #pragma omp for
        for (Int Ckk = 0; Ckk < nrows; Ckk+=kblock) {
            Int Ckmax = Ckk + kblock;
            if (Ckmax > nrows) {
                Ckmax = nrows;
            }
            for (Int Cjj = 0; Cjj < nB_cols; Cjj+=jblock) {
                Int Cjmax = Cjj + jblock;
                if (Cjmax > nB_cols) {
                    Cjmax = nB_cols;
                }

                F* R = &Rglobal.get()[omp_get_thread_num()*kblock*jblock];
                for (Int Ck = Ckk; Ck < Ckmax; Ck++) {
                    Int k = rows[Ck];
                    for (Int Cj = Cjj; Cj < Cjmax; Cj++) {
                        Int j = B_cols[Cj];
                        %if order == 'C':
                            F Bv = B[k * r + j];
                        % else:
                            F Bv = B[j * n + k];
                        % endif
                        R[(Ck-Ckk) * jblock + (Cj-Cjj)] = d[k] * Bv;
                    }
                }

                for (Int Ck = Ckk; Ck < Ckmax; Ck++) {
                    Int k = rows[Ck];
                    for (Int A_idx = Aindptr[k]; A_idx < Aindptr[k+1]; A_idx++) {
                        Int i = Aindices[A_idx];
                        Int Ci = Acol_map[i];
                        if (Ci == -1) {
                            continue;
                        }

                        F Q = Adata[A_idx];
                        auto Qsimd = xs::XSIMD_BROADCAST(Q);

                        Int Cj = Cjj;
                        Int Cjmax2 = Cjj + ((Cjmax - Cjj) / simd_size) * simd_size;
                        for (; Cj < Cjmax2; Cj+=simd_size) {
                            auto Bsimd = xs::load_aligned(&R[(Ck-Ckk)*jblock+(Cj-Cjj)]);
                            auto outsimd = xs::load_aligned(&outtemp.get()[Ci*nB_cols_rounded+Cj]);
                            outsimd = xs::fma(Qsimd, Bsimd, outsimd);
                            outsimd.store_aligned(&outtemp.get()[Ci*nB_cols_rounded+Cj]);
                        }

                        for (; Cj < Cjmax; Cj++) {
                            outtemp.get()[Ci*nB_cols_rounded+Cj] += Q * R[(Ck-Ckk)*jblock+(Cj-Cjj)];
                        }
                    }
                }
            }
        }

        for (Int Ci = 0; Ci < nA_cols; Ci++) {
            for (Int Cj = 0; Cj < nB_cols; Cj++) {
                #pragma omp atomic
                out[Ci*nB_cols+Cj] += outtemp.get()[Ci*nB_cols_rounded+Cj];
            }
        }
    }
}
</%def>

${csr_dense_sandwich_tmpl('C')}
${csr_dense_sandwich_tmpl('F')}
