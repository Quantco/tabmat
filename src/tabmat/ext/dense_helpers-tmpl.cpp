// The dense_sandwich function below implement a BLIS/GotoBLAS-like sandwich
// product for computing A.T @ diag(d) @ A
// If a per-column center mu is passed, it computes (A - mu).T @ diag(d) @ (A - mu)
// instead; the centering is fused into the panel packing, so it costs ~nothing.
// It works for both C-ordered and Fortran-ordered matrices.
// It is parallelized to be fast for both narrow and square matrices
//
// A good intro to thinking about matrix-multiply optimization is here:
// https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-172-performance-engineering-of-software-systems-fall-2018/lecture-slides/MIT6_172F18_lec1.pdf
//
// For more reading, it'd be good to dig into the GotoBLAS and BLIS implementation. 
// page 3 here has a good summary of the ordered of blocking/loops:
// http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
//
// The innermost simd loop is parallelized using xsimd and should
// use the largest vector instructions available on any given machine.
//
// There's a bit of added complexity here from the use of Mako templates.
// It looks scary, but it makes the loop unrolling and generalization across
// matrix orderings and parallelization schemes much simpler than it would be
// if implemented directly.

#include <xsimd/xsimd.hpp>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

#include "alloc.h"

#if XSIMD_VERSION_MAJOR >= 8
    #define XSIMD_BROADCAST broadcast
#else
    #define XSIMD_BROADCAST set_simd
#endif

#if XSIMD_VERSION_MAJOR >= 9
    #define XSIMD_REDUCE_ADD reduce_add
#else
    #define XSIMD_REDUCE_ADD hadd
#endif

namespace xs = xsimd;

<%def name="middle_j(kparallel, IBLOCK, JBLOCK)">
    int jmaxblock = jmin + ((jmaxinner - jmin) / ${JBLOCK}) * ${JBLOCK};
    for (; j < jmaxblock; j += ${JBLOCK}) {

        // setup simd accumulators
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
		auto accumsimd${ir}_${jr} = (F)0.0;
#else
                auto accumsimd${ir}_${jr} = xs::XSIMD_BROADCAST(((F)0.0));
#endif
            % endfor
        % endfor

        % for ir in range(IBLOCK):
            int basei${ir} = (i - imin2 + ${ir}) * kstep;
        % endfor
        % for jr in range(JBLOCK):
            int basej${jr} = (j - jmin2 + ${jr}) * kstep;
        % endfor

        // main simd inner loop
        % for ir in range(IBLOCK):
            F* Lptr${ir} = &L[basei${ir}];
        % endfor
        % for jr in range(JBLOCK):
            F* Rptr${jr} = &R[basej${jr}];
        % endfor
        int kblocksize = ((kmax - kmin) / simd_size) * simd_size;
        F* Rptr0end = Rptr0 + kblocksize;
        for(; Rptr0 < Rptr0end; 
            % for jr in range(JBLOCK):
                Rptr${jr}+=simd_size,
            % endfor
            % for ir in range(IBLOCK):
                % if ir == IBLOCK - 1:
                    Lptr${ir} += simd_size
                % else:
                    Lptr${ir} += simd_size,
                % endif
            % endfor
            ) {
            % for ir in range(IBLOCK):
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
		auto Xtd${ir} = *Lptr${ir};
#else
                auto Xtd${ir} = xs::load_aligned(Lptr${ir});
#endif
                % for jr in range(JBLOCK):
                {
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
		    auto Xsimd = *Rptr${jr};
#else
                    auto Xsimd = xs::load_aligned(Rptr${jr});
#endif
                    accumsimd${ir}_${jr} = xs::fma(Xtd${ir}, Xsimd, accumsimd${ir}_${jr});
                }
                % endfor
            % endfor
        }

        // horizontal sum of the simd blocks
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
		F accum${ir}_${jr} = accumsimd${ir}_${jr};
#else
                F accum${ir}_${jr} = xs::XSIMD_REDUCE_ADD(accumsimd${ir}_${jr});
#endif
            % endfor
        % endfor

        // remainder loop handling the entries that can't be handled in a
        // simd_size stride
        for (int k = kblocksize; k < kmax - kmin; k++) {
            % for ir in range(IBLOCK):
                F Xtd${ir} = L[basei${ir} + k];
            % endfor
            % for jr in range(JBLOCK):
                F Xv${jr} = R[basej${jr} + k];
            % endfor
            % for ir in range(IBLOCK):
                % for jr in range(JBLOCK):
                    accum${ir}_${jr} += Xtd${ir} * Xv${jr};
                % endfor
            % endfor
        }

        // add to the output array
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
                % if kparallel:
                    // we only need to be careful about parallelism when we're
                    // parallelizing the k loop. if we're just parallelizing i
                    // and j, the sum here is safe
                    #pragma omp atomic
                % endif
                out[(i + ${ir}) * out_m + (j + ${jr})] += accum${ir}_${jr};
            % endfor
        % endfor
    }
</%def>

<%def name="outer_i(kparallel, IBLOCK, JBLOCKS)">
    int imaxblock = imin + ((imax - imin) / ${IBLOCK}) * ${IBLOCK};
    for (; i < imaxblock; i += ${IBLOCK}) {
        int jmaxinner = jmax;
        if (jmaxinner > i + ${IBLOCK}) {
            jmaxinner = i + ${IBLOCK};
        }
        int j = jmin;
        % for JBLOCK in JBLOCKS:
        {
            ${middle_j(kparallel, IBLOCK, JBLOCK)}
        }
        % endfor
    }
</%def>

<%def name="dense_base_tmpl(kparallel)">
template <typename Int, typename F>
void dense_base${kparallel}(F* R, F* L, F* d, F* out,
                Py_ssize_t out_m,
                Py_ssize_t imin2, Py_ssize_t imax2,
                Py_ssize_t jmin2, Py_ssize_t jmax2, 
                Py_ssize_t kmin, Py_ssize_t kmax, Int innerblock, Int kstep) 
{
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr std::size_t simd_size = 1;
#else
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
#endif
    for (Py_ssize_t imin = imin2; imin < imax2; imin+=innerblock) {
        Py_ssize_t imax = imin + innerblock; 
        if (imax > imax2) {
            imax = imax2; 
        }
        for (Py_ssize_t jmin = jmin2; jmin < jmax2; jmin+=innerblock) {
            Py_ssize_t jmax = jmin + innerblock; 
            if (jmax > jmax2) {
                jmax = jmax2; 
            }
            Py_ssize_t i = imin;
            % for IBLOCK in [4, 2, 1]:
            {
                ${outer_i(kparallel, IBLOCK, [4, 2, 1])}
            }
            % endfor
        }
    }
}
</%def>

${dense_base_tmpl(True)}
${dense_base_tmpl(False)}

<%def name="k_loop(kparallel, order)">
% if kparallel:
    #pragma omp parallel for
    for (Py_ssize_t Rk = 0; Rk < in_n; Rk+=kratio*thresh1d) {
% else:
    for (Py_ssize_t Rk = 0; Rk < in_n; Rk+=kratio*thresh1d) {
% endif
    int Rkmax2 = Rk + kratio * thresh1d; 
    if (Rkmax2 > in_n) {
        Rkmax2 = in_n; 
    }

    F* R = Rglobal.get();
    % if kparallel:
    R += omp_get_thread_num()*thresh1d*thresh1d*kratio*kratio;
    for (Py_ssize_t Cjj = Cj; Cjj < Cjmax2; Cjj++) {
    % else:
    #pragma omp parallel for
    for (Py_ssize_t Cjj = Cj; Cjj < Cjmax2; Cjj++) {
    % endif
        {
            Int jj = cols[Cjj];
            // Optional per-column center: pack d * (X - mu) instead of d * X.
            // x - 0 == x exactly, so mu == nullptr reproduces the old result.
            F mujj = (mu == nullptr) ? (F)0.0 : mu[jj];
            %if order == 'F':
                //TODO: this could use some pointer logic for the R assignment?
                for (Py_ssize_t Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    Int kk = rows[Rkk];
                    R[(Cjj-Cj) * kratio * thresh1d + (Rkk-Rk)] = d[kk] * (X[(Py_ssize_t) jj * n + kk] - mujj);
                }
            % else:
                for (Py_ssize_t Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    Int kk = rows[Rkk];
                    R[(Cjj-Cj) * kratio * thresh1d + (Rkk-Rk)] = d[kk] * (X[(Py_ssize_t) kk * m + jj] - mujj);
                }
            % endif
        }
    }

    % if kparallel:
        for (Py_ssize_t Ci = Cj; Ci < out_m; Ci+=thresh1d) {
    % else:
        #pragma omp parallel for
        for (Py_ssize_t Ci = Cj; Ci < out_m; Ci+=thresh1d) {
    % endif
        Py_ssize_t Cimax2 = Ci + thresh1d; 
        if (Cimax2 > out_m) {
            Cimax2 = out_m; 
        }
        F* L = &Lglobal.get()[omp_get_thread_num()*thresh1d*thresh1d*kratio];
        for (Py_ssize_t Cii = Ci; Cii < Cimax2; Cii++) {
            Int ii = cols[Cii];
            F muii = (mu == nullptr) ? (F)0.0 : mu[ii];
            %if order == 'F':
                for (Py_ssize_t Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    Int kk = rows[Rkk];
                    L[(Py_ssize_t) (Cii-Ci) * kratio * thresh1d + (Rkk-Rk)] = X[(Py_ssize_t) ii * n + kk] - muii;
                }
            % else:
                for (Py_ssize_t Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    Int kk = rows[Rkk];
                    L[(Py_ssize_t) (Cii-Ci) * kratio * thresh1d + (Rkk-Rk)] = X[(Py_ssize_t) kk * m + ii] - muii;
                }
            % endif
        }
        dense_base${kparallel}(R, L, d, out, out_m, Ci, Cimax2, Cj, Cjmax2, Rk, Rkmax2, innerblock, kratio*thresh1d);
    }
}
</%def>


<%def name="dense_sandwich_tmpl(order)">
template <typename Int, typename F>
void _dense${order}_sandwich(Int* rows, Int* cols, F* X, F* d, const F* mu, F* out,
        Int in_n, Int out_m, Int m, Int n, Int thresh1d, Int kratio, Int innerblock) 
{
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr std::size_t simd_size = 1;
#else
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
#endif
    constexpr auto alignment = simd_size * sizeof(F);

    bool kparallel = (in_n / (kratio*thresh1d)) > (out_m / thresh1d);
    Py_ssize_t Rsize = thresh1d*thresh1d*kratio*kratio;
    if (kparallel) {
        Rsize *= omp_get_max_threads();
    }

    auto Rglobal = make_aligned_unique<F>(Rsize, alignment);
    auto Lglobal = make_aligned_unique<F>(
        omp_get_max_threads() * thresh1d * thresh1d * kratio, 
        alignment
    );
    for (Py_ssize_t Cj = 0; Cj < out_m; Cj+=kratio*thresh1d) {
        Py_ssize_t Cjmax2 = Cj + kratio*thresh1d; 
        if (Cjmax2 > out_m) {
            Cjmax2 = out_m; 
        }
        if (kparallel) {
            ${k_loop(True, order)}
        } else {
            ${k_loop(False, order)}
        }
    }

    #pragma omp parallel for if(out_m > 100)
    for (Py_ssize_t Ci = 0; Ci < out_m; Ci++) {
        for (Py_ssize_t Cj = 0; Cj <= Ci; Cj++) {
            out[Cj * out_m + Ci] = out[Ci * out_m + Cj];
        }
    }
}
</%def>

${dense_sandwich_tmpl('C')}
${dense_sandwich_tmpl('F')}


// The matrix-vector kernels below compute with (X - mu) for a per-column
// center mu, subtracting it from each entry as it is read. Expanding into
// X @ v - mu @ v instead would cancel when mu is large relative to the spread
// of X. Passing nullptr for mu means no centering, and nullptr for rows or cols
// selects all of them. The inner loops run along the contiguous direction of X
// (rows for F order, columns for C order) and are vectorized if that direction
// is not restricted; otherwise they fall back to indexed loads.

// sum_k (x[k] - c[k]) * y[k], or with c[0] for all k if not VC.
template <bool VC, typename F>
inline F centered_dot(const F* x, const F* c, const F* y, Py_ssize_t len) {
    Py_ssize_t k = 0;
    F total = 0.0;
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr Py_ssize_t simd_size = xsimd::simd_type<F>::size;
    auto c0 = xs::XSIMD_BROADCAST(c[0]);
    auto acc0 = xs::XSIMD_BROADCAST((F)0.0);
    auto acc1 = acc0;
    for (; k + 2 * simd_size <= len; k += 2 * simd_size) {
        auto ca = c0;
        auto cb = c0;
        if constexpr (VC) {
            ca = xs::load_unaligned(c + k);
            cb = xs::load_unaligned(c + k + simd_size);
        }
        acc0 = xs::fma(xs::load_unaligned(x + k) - ca, xs::load_unaligned(y + k), acc0);
        acc1 = xs::fma(
            xs::load_unaligned(x + k + simd_size) - cb,
            xs::load_unaligned(y + k + simd_size),
            acc1
        );
    }
    total = xs::XSIMD_REDUCE_ADD(acc0 + acc1);
#endif
    for (; k < len; k++) {
        total += (x[k] - (VC ? c[k] : c[0])) * y[k];
    }
    return total;
}

// y[k] += (x[k] - c[k]) * a, or with c[0] for all k if not VC.
template <bool VC, typename F>
inline void centered_axpy(const F* x, const F* c, F a, F* y, Py_ssize_t len) {
    Py_ssize_t k = 0;
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr Py_ssize_t simd_size = xsimd::simd_type<F>::size;
    auto c0 = xs::XSIMD_BROADCAST(c[0]);
    auto asimd = xs::XSIMD_BROADCAST(a);
    for (; k + simd_size <= len; k += simd_size) {
        auto ck = c0;
        if constexpr (VC) {
            ck = xs::load_unaligned(c + k);
        }
        auto yk = xs::fma(xs::load_unaligned(x + k) - ck, asimd, xs::load_unaligned(y + k));
        xs::store_unaligned(y + k, yk);
    }
#endif
    for (; k < len; k++) {
        y[k] += (x[k] - (VC ? c[k] : c[0])) * a;
    }
}

// y[k] += sum_t (x[t][k] - c[t]) * a[t] for t < 4, which reads and writes y
// once for four columns.
template <typename F>
inline void centered_axpy4(const F* const* x, const F* c, const F* a, F* y, Py_ssize_t len) {
    Py_ssize_t k = 0;
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr Py_ssize_t simd_size = xsimd::simd_type<F>::size;
    % for t in range(4):
        auto c${t} = xs::XSIMD_BROADCAST(c[${t}]);
        auto a${t} = xs::XSIMD_BROADCAST(a[${t}]);
    % endfor
    for (; k + simd_size <= len; k += simd_size) {
        auto yk = xs::load_unaligned(y + k);
        % for t in range(4):
            yk = xs::fma(xs::load_unaligned(x[${t}] + k) - c${t}, a${t}, yk);
        % endfor
        xs::store_unaligned(y + k, yk);
    }
#endif
    for (; k < len; k++) {
        F yk = y[k];
        for (int t = 0; t < 4; t++) {
            yk += (x[t][k] - c[t]) * a[t];
        }
        y[k] = yk;
    }
}

<%def name="dense_rmatvec_tmpl(order)">
// Computes (X - mu)[rows, cols].T @ v[rows].
template <typename Int, typename F>
void _dense${order}_rmatvec(Int* rows, Int* cols, F* X, F* v, const F* mu, F* out,
        Int n_rows, Int n_cols, Int m, Int n)
{
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
    constexpr std::size_t simd_size = 1;
#else
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
#endif
    constexpr std::size_t alignment = simd_size * sizeof(F);

    std::vector<F> zeros;
    if (mu == nullptr) {
        zeros.assign(m, (F)0.0);
        mu = zeros.data();
    }

    auto outglobal = make_aligned_unique<F>(omp_get_max_threads()*n_cols, alignment);

    // Longer blocks for F order mean fewer column switches per block.
    constexpr Py_ssize_t rowblocksize = ${'4096' if order == 'F' else '1024'};

    // Starting threads costs more than a small product.
    #pragma omp parallel for if((Py_ssize_t) n_rows * n_cols > 100000)
    for (Py_ssize_t Ci = 0; Ci < n_rows; Ci += rowblocksize) {
        Py_ssize_t Cimax = std::min(Ci + rowblocksize, (Py_ssize_t) n_rows);

        F* outlocal = &outglobal.get()[omp_get_thread_num()*n_cols];

        % if order == 'F':
            for (Py_ssize_t Cjj = 0; Cjj < n_cols; Cjj++) {
                Py_ssize_t j = (cols == nullptr) ? Cjj : cols[Cjj];
                const F* Xj = &X[j * n];
                F muj = mu[j];
                if (rows == nullptr) {
                    outlocal[Cjj] = centered_dot<false>(&Xj[Ci], &muj, &v[Ci], Cimax - Ci);
                } else {
                    F out_entry = 0.0;
                    for (Py_ssize_t Cii = Ci; Cii < Cimax; Cii++) {
                        Int i = rows[Cii];
                        out_entry += (Xj[i] - muj) * v[i];
                    }
                    outlocal[Cjj] = out_entry;
                }
            }
        % else:
            for (Py_ssize_t Cjj = 0; Cjj < n_cols; Cjj++) {
                outlocal[Cjj] = 0.0;
            }
            for (Py_ssize_t Cii = Ci; Cii < Cimax; Cii++) {
                Py_ssize_t i = (rows == nullptr) ? Cii : rows[Cii];
                const F* Xi = &X[i * m];
                F vi = v[i];
                if (cols == nullptr) {
                    centered_axpy<true>(Xi, mu, vi, outlocal, n_cols);
                } else {
                    for (Py_ssize_t Cjj = 0; Cjj < n_cols; Cjj++) {
                        Int j = cols[Cjj];
                        outlocal[Cjj] += (Xi[j] - mu[j]) * vi;
                    }
                }
            }
        % endif

        for (Py_ssize_t Cj = 0; Cj < n_cols; Cj++) {
            #pragma omp atomic
            out[Cj] += outlocal[Cj];
        }
    }
}
</%def>
${dense_rmatvec_tmpl('C')}
${dense_rmatvec_tmpl('F')}

<%def name="dense_matvec_tmpl(order)">
// Computes (X - mu)[rows, cols] @ v[cols].
template <typename Int, typename F>
void _dense${order}_matvec(Int* rows, Int* cols, F* X, F* v, const F* mu, F* out,
        Int n_rows, Int n_cols, Int m, Int n)
{
    std::vector<F> zeros;
    if (mu == nullptr) {
        zeros.assign(m, (F)0.0);
        mu = zeros.data();
    }

    constexpr Py_ssize_t rowblocksize = ${'4096' if order == 'F' else '1024'};

    // Starting threads costs more than a small product.
    #pragma omp parallel for if((Py_ssize_t) n_rows * n_cols > 100000)
    for (Py_ssize_t Ci = 0; Ci < n_rows; Ci += rowblocksize) {
        Py_ssize_t Cimax = std::min(Ci + rowblocksize, (Py_ssize_t) n_rows);
        % if order == 'F':
            // Each thread owns out[Ci:Cimax] and adds the columns to it, four
            // at a time if all rows are used.
            for (Py_ssize_t Cii = Ci; Cii < Cimax; Cii++) {
                out[Cii] = 0.0;
            }
            Py_ssize_t Cjj = 0;
            if (rows == nullptr) {
                for (; Cjj + 4 <= n_cols; Cjj += 4) {
                    const F* Xs[4];
                    F mus[4];
                    F vs[4];
                    for (int t = 0; t < 4; t++) {
                        Py_ssize_t j = (cols == nullptr) ? Cjj + t : cols[Cjj + t];
                        Xs[t] = &X[j * n + Ci];
                        mus[t] = mu[j];
                        vs[t] = v[j];
                    }
                    centered_axpy4(Xs, mus, vs, &out[Ci], Cimax - Ci);
                }
            }
            for (; Cjj < n_cols; Cjj++) {
                Py_ssize_t j = (cols == nullptr) ? Cjj : cols[Cjj];
                const F* Xj = &X[j * n];
                F muj = mu[j];
                F vj = v[j];
                if (rows == nullptr) {
                    centered_axpy<false>(&Xj[Ci], &muj, vj, &out[Ci], Cimax - Ci);
                } else {
                    for (Py_ssize_t Cii = Ci; Cii < Cimax; Cii++) {
                        out[Cii] += (Xj[rows[Cii]] - muj) * vj;
                    }
                }
            }
        % else:
            for (Py_ssize_t Cii = Ci; Cii < Cimax; Cii++) {
                Py_ssize_t i = (rows == nullptr) ? Cii : rows[Cii];
                const F* Xi = &X[i * m];
                if (cols == nullptr) {
                    out[Cii] = centered_dot<true>(Xi, mu, v, n_cols);
                } else {
                    F out_entry = 0.0;
                    for (Py_ssize_t Cjj = 0; Cjj < n_cols; Cjj++) {
                        Int j = cols[Cjj];
                        out_entry += (Xi[j] - mu[j]) * v[j];
                    }
                    out[Cii] = out_entry;
                }
            }
        % endif
    }
}
</%def>
${dense_matvec_tmpl('C')}
${dense_matvec_tmpl('F')}
