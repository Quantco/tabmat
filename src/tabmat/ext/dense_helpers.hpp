// The dense_sandwich function below implements a BLIS/GotoBLAS-like sandwich
// product for computing A.T @ diag(d) @ A
// With a per-column center mu, it computes (A - mu).T @ diag(d) @ (A - mu); the
// centering is fused into the panel packing, so it costs ~nothing. A zero center
// gives the uncentered product exactly.
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
// The variants of each kernel (memory order, k-parallel vs column-parallel,
// register tile shape) are template instantiations: the compiler unrolls the
// constant-bound loops over the IBLOCK x JBLOCK tile and discards the
// branches that `if constexpr` rules out.

#pragma once

#include <algorithm>
#include <xsimd/xsimd.hpp>
#include <omp.h>

#include "alloc.h"
#include "order.h"

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

// Thin wrappers so the kernels read the same whether xsimd found a vector
// instruction set or has to fall back to scalar code.
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <typename F> using simd_t = F;
template <typename F> constexpr std::size_t simd_size_v = 1;
template <typename F> inline F simd_zero() { return (F)0.0; }
template <typename F> inline F simd_load(const F* ptr) { return *ptr; }
template <typename F> inline F simd_reduce_add(F x) { return x; }
#else
template <typename F> using simd_t = xs::simd_type<F>;
template <typename F> constexpr std::size_t simd_size_v = xs::simd_type<F>::size;
template <typename F> inline simd_t<F> simd_zero() { return xs::XSIMD_BROADCAST((F)0.0); }
template <typename F> inline simd_t<F> simd_load(const F* ptr) { return xs::load_aligned(ptr); }
template <typename F> inline F simd_reduce_add(const simd_t<F>& x) { return xs::XSIMD_REDUCE_ADD(x); }
#endif

// ---------------------------------------------------------------------------
// Micro-kernel: accumulate the IBLOCK x JBLOCK tile of
// out[i:i+IBLOCK, j:j+JBLOCK] over k in [kmin, kmax) from the packed panels.
//
// L holds rows of out (plain X columns), R holds columns of out (d-scaled X
// columns). Both are packed k-contiguous with a stride of kstep between
// columns, and the panels start at out row imin2 / out column jmin2.
// ---------------------------------------------------------------------------
template <bool kparallel, int IBLOCK, int JBLOCK, typename Int, typename F>
inline void dense_tile(const F* R, const F* L, F* out, Py_ssize_t out_m,
                       Py_ssize_t i, Py_ssize_t j,
                       Py_ssize_t imin2, Py_ssize_t jmin2,
                       Py_ssize_t kmin, Py_ssize_t kmax, Int kstep)
{
    constexpr std::size_t simd_size = simd_size_v<F>;

    // setup simd accumulators
    simd_t<F> accumsimd[IBLOCK][JBLOCK];
    for (int ir = 0; ir < IBLOCK; ir++) {
        for (int jr = 0; jr < JBLOCK; jr++) {
            accumsimd[ir][jr] = simd_zero<F>();
        }
    }

    const F* Lptr[IBLOCK];
    for (int ir = 0; ir < IBLOCK; ir++) {
        Lptr[ir] = &L[(i - imin2 + ir) * kstep];
    }
    const F* Rptr[JBLOCK];
    for (int jr = 0; jr < JBLOCK; jr++) {
        Rptr[jr] = &R[(j - jmin2 + jr) * kstep];
    }

    // main simd inner loop
    const Py_ssize_t klen = kmax - kmin;
    const Py_ssize_t kblocksize = (klen / simd_size) * simd_size;
    for (Py_ssize_t k = 0; k < kblocksize; k += simd_size) {
        for (int ir = 0; ir < IBLOCK; ir++) {
            auto Xtd = simd_load<F>(Lptr[ir] + k);
            for (int jr = 0; jr < JBLOCK; jr++) {
                auto Xsimd = simd_load<F>(Rptr[jr] + k);
                accumsimd[ir][jr] = xs::fma(Xtd, Xsimd, accumsimd[ir][jr]);
            }
        }
    }

    // horizontal sum of the simd blocks
    F accum[IBLOCK][JBLOCK];
    for (int ir = 0; ir < IBLOCK; ir++) {
        for (int jr = 0; jr < JBLOCK; jr++) {
            accum[ir][jr] = simd_reduce_add<F>(accumsimd[ir][jr]);
        }
    }

    // remainder loop handling the entries that can't be handled in a
    // simd_size stride
    for (Py_ssize_t k = kblocksize; k < klen; k++) {
        for (int ir = 0; ir < IBLOCK; ir++) {
            F Xtd = Lptr[ir][k];
            for (int jr = 0; jr < JBLOCK; jr++) {
                accum[ir][jr] += Xtd * Rptr[jr][k];
            }
        }
    }

    // add to the output array
    for (int ir = 0; ir < IBLOCK; ir++) {
        for (int jr = 0; jr < JBLOCK; jr++) {
            if constexpr (kparallel) {
                // we only need to be careful about parallelism when we're
                // parallelizing the k loop. if we're just parallelizing i
                // and j, the sum here is safe
                #pragma omp atomic
                out[(i + ir) * out_m + (j + jr)] += accum[ir][jr];
            } else {
                out[(i + ir) * out_m + (j + jr)] += accum[ir][jr];
            }
        }
    }
}

// Sweep j over [jmin, jmaxinner) in steps of JBLOCK for the rows i..i+IBLOCK,
// starting from the current j and advancing it. The remainder that doesn't
// fit a JBLOCK-wide tile is left for the caller's next, narrower JBLOCK.
template <bool kparallel, int IBLOCK, int JBLOCK, typename Int, typename F>
inline void dense_tile_cols(const F* R, const F* L, F* out, Py_ssize_t out_m,
                            Py_ssize_t i, Py_ssize_t& j,
                            Py_ssize_t jmin, Py_ssize_t jmaxinner,
                            Py_ssize_t imin2, Py_ssize_t jmin2,
                            Py_ssize_t kmin, Py_ssize_t kmax, Int kstep)
{
    Py_ssize_t jmaxblock = jmin + ((jmaxinner - jmin) / JBLOCK) * JBLOCK;
    for (; j < jmaxblock; j += JBLOCK) {
        dense_tile<kparallel, IBLOCK, JBLOCK>(
            R, L, out, out_m, i, j, imin2, jmin2, kmin, kmax, kstep
        );
    }
}

// Sweep i over [imin, imax) in steps of IBLOCK, starting from the current i
// and advancing it. Only the lower triangle (j <= i) of out is computed, so
// the j sweep is clipped at i + IBLOCK.
template <bool kparallel, int IBLOCK, typename Int, typename F>
inline void dense_tile_rows(const F* R, const F* L, F* out, Py_ssize_t out_m,
                            Py_ssize_t& i, Py_ssize_t imin, Py_ssize_t imax,
                            Py_ssize_t jmin, Py_ssize_t jmax,
                            Py_ssize_t imin2, Py_ssize_t jmin2,
                            Py_ssize_t kmin, Py_ssize_t kmax, Int kstep)
{
    Py_ssize_t imaxblock = imin + ((imax - imin) / IBLOCK) * IBLOCK;
    for (; i < imaxblock; i += IBLOCK) {
        Py_ssize_t jmaxinner = std::min<Py_ssize_t>(jmax, i + IBLOCK);
        Py_ssize_t j = jmin;
        dense_tile_cols<kparallel, IBLOCK, 4>(R, L, out, out_m, i, j, jmin, jmaxinner, imin2, jmin2, kmin, kmax, kstep);
        dense_tile_cols<kparallel, IBLOCK, 2>(R, L, out, out_m, i, j, jmin, jmaxinner, imin2, jmin2, kmin, kmax, kstep);
        dense_tile_cols<kparallel, IBLOCK, 1>(R, L, out, out_m, i, j, jmin, jmaxinner, imin2, jmin2, kmin, kmax, kstep);
    }
}

// Compute the lower-triangular part of the out block
// [imin2, imax2) x [jmin2, jmax2) from the packed panels, in innerblock-sized
// cache blocks, each tiled with the widest register tile that fits.
template <bool kparallel, typename Int, typename F>
void dense_base(const F* R, const F* L, F* out, Py_ssize_t out_m,
                Py_ssize_t imin2, Py_ssize_t imax2,
                Py_ssize_t jmin2, Py_ssize_t jmax2,
                Py_ssize_t kmin, Py_ssize_t kmax, Int innerblock, Int kstep)
{
    for (Py_ssize_t imin = imin2; imin < imax2; imin += innerblock) {
        Py_ssize_t imax = std::min<Py_ssize_t>(imin + innerblock, imax2);
        for (Py_ssize_t jmin = jmin2; jmin < jmax2; jmin += innerblock) {
            Py_ssize_t jmax = std::min<Py_ssize_t>(jmin + innerblock, jmax2);
            Py_ssize_t i = imin;
            dense_tile_rows<kparallel, 4>(R, L, out, out_m, i, imin, imax, jmin, jmax, imin2, jmin2, kmin, kmax, kstep);
            dense_tile_rows<kparallel, 2>(R, L, out, out_m, i, imin, imax, jmin, jmax, imin2, jmin2, kmin, kmax, kstep);
            dense_tile_rows<kparallel, 1>(R, L, out, out_m, i, imin, imax, jmin, jmax, imin2, jmin2, kmin, kmax, kstep);
        }
    }
}

// ---------------------------------------------------------------------------
// Packing: copy the rows [Rk, Rkmax2) of one X column into a k-contiguous
// panel column, subtracting the column's center. R gets the d-scaled values,
// L the plain ones.
// ---------------------------------------------------------------------------
template <Order order, typename Int, typename F>
inline void pack_R_column(const Int* rows, const Int* cols, const F* X, const F* d,
                          const F* mu, F* R, Py_ssize_t Cj, Py_ssize_t Cjj,
                          Py_ssize_t Rk, Py_ssize_t Rkmax2,
                          Int m, Int n, Int kstep)
{
    Int jj = cols[Cjj];
    // Pack d * (X - mu) instead of d * X.
    F mujj = mu[jj];
    F* Rcol = &R[(Cjj - Cj) * kstep];
    for (Py_ssize_t Rkk = Rk; Rkk < Rkmax2; Rkk++) {
        Int kk = rows[Rkk];
        Rcol[Rkk - Rk] = d[kk] * (load_X<order>(X, kk, jj, n, m) - mujj);
    }
}

template <Order order, typename Int, typename F>
inline void pack_L_column(const Int* rows, const Int* cols, const F* X, const F* mu,
                          F* L, Py_ssize_t Ci, Py_ssize_t Cii,
                          Py_ssize_t Rk, Py_ssize_t Rkmax2,
                          Int m, Int n, Int kstep)
{
    Int ii = cols[Cii];
    F muii = mu[ii];
    F* Lcol = &L[(Cii - Ci) * kstep];
    for (Py_ssize_t Rkk = Rk; Rkk < Rkmax2; Rkk++) {
        Int kk = rows[Rkk];
        Lcol[Rkk - Rk] = load_X<order>(X, kk, ii, n, m) - muii;
    }
}

// Pack the L panel for out rows [Ci, Ci + thresh1d) into this thread's L
// buffer and multiply it against the R panel.
template <bool kparallel, Order order, typename Int, typename F>
inline void dense_row_block(const Int* rows, const Int* cols, const F* X, const F* mu,
                            F* out, const F* R, F* Lglobal,
                            Py_ssize_t Ci, Py_ssize_t Cj, Py_ssize_t Cjmax2,
                            Py_ssize_t Rk, Py_ssize_t Rkmax2,
                            Int out_m, Int m, Int n,
                            Int thresh1d, Int innerblock, Int kstep)
{
    Py_ssize_t Cimax2 = std::min<Py_ssize_t>(Ci + thresh1d, out_m);
    F* L = &Lglobal[(Py_ssize_t) omp_get_thread_num() * thresh1d * kstep];
    for (Py_ssize_t Cii = Ci; Cii < Cimax2; Cii++) {
        pack_L_column<order>(rows, cols, X, mu, L, Ci, Cii, Rk, Rkmax2, m, n, kstep);
    }
    dense_base<kparallel>(
        R, L, out, out_m, Ci, Cimax2, Cj, Cjmax2, Rk, Rkmax2, innerblock, kstep
    );
}

// All the work for the out columns [Cj, Cjmax2): for each k block, pack the
// R panel, then sweep the out rows from Cj downwards (lower triangle only).
//
// kparallel chooses which loop the threads split:
//  - true:  the k loop. Each thread packs its own R panel and the partial
//           sums of different k blocks meet in out, hence the atomics in the
//           micro-kernel. Chosen when X is tall and narrow.
//  - false: the pack loop over columns and the row-block loop, with one
//           shared R panel. Chosen when out is large enough to split.
template <bool kparallel, Order order, typename Int, typename F>
void dense_column_block(const Int* rows, const Int* cols, const F* X, const F* d,
                        const F* mu, F* out, F* Rglobal, F* Lglobal,
                        Py_ssize_t Cj, Py_ssize_t Cjmax2,
                        Int in_n, Int out_m, Int m, Int n,
                        Int thresh1d, Int innerblock, Int kstep)
{
    if constexpr (kparallel) {
        #pragma omp parallel for
        for (Py_ssize_t Rk = 0; Rk < in_n; Rk += kstep) {
            Py_ssize_t Rkmax2 = std::min<Py_ssize_t>(Rk + kstep, in_n);
            F* R = &Rglobal[(Py_ssize_t) omp_get_thread_num() * kstep * kstep];
            for (Py_ssize_t Cjj = Cj; Cjj < Cjmax2; Cjj++) {
                pack_R_column<order>(rows, cols, X, d, mu, R, Cj, Cjj, Rk, Rkmax2, m, n, kstep);
            }
            for (Py_ssize_t Ci = Cj; Ci < out_m; Ci += thresh1d) {
                dense_row_block<kparallel, order>(
                    rows, cols, X, mu, out, R, Lglobal, Ci, Cj, Cjmax2, Rk, Rkmax2,
                    out_m, m, n, thresh1d, innerblock, kstep
                );
            }
        }
    } else {
        for (Py_ssize_t Rk = 0; Rk < in_n; Rk += kstep) {
            Py_ssize_t Rkmax2 = std::min<Py_ssize_t>(Rk + kstep, in_n);
            F* R = Rglobal;
            #pragma omp parallel for
            for (Py_ssize_t Cjj = Cj; Cjj < Cjmax2; Cjj++) {
                pack_R_column<order>(rows, cols, X, d, mu, R, Cj, Cjj, Rk, Rkmax2, m, n, kstep);
            }
            #pragma omp parallel for
            for (Py_ssize_t Ci = Cj; Ci < out_m; Ci += thresh1d) {
                dense_row_block<kparallel, order>(
                    rows, cols, X, mu, out, R, Lglobal, Ci, Cj, Cjmax2, Rk, Rkmax2,
                    out_m, m, n, thresh1d, innerblock, kstep
                );
            }
        }
    }
}

// out[cols, cols] = (X - mu)[rows, cols].T @ diag(d[rows]) @ (X - mu)[rows, cols]
// X is n x m in the given order, mu is a per-column center of length m; out is
// out_m x out_m, row-major, zeroed.
template <Order order, typename Int, typename F>
void dense_sandwich(const Int* rows, const Int* cols, const F* X, const F* d,
                    const F* mu, F* out, Int in_n, Int out_m, Int m, Int n,
                    Int thresh1d, Int kratio, Int innerblock)
{
    constexpr auto alignment = simd_size_v<F> * sizeof(F);
    const Int kstep = kratio * thresh1d;

    bool kparallel = (in_n / kstep) > (out_m / thresh1d);
    Py_ssize_t Rsize = (Py_ssize_t) kstep * kstep;
    if (kparallel) {
        Rsize *= omp_get_max_threads();
    }

    auto Rglobal = make_aligned_unique<F>(Rsize, alignment);
    auto Lglobal = make_aligned_unique<F>(
        (Py_ssize_t) omp_get_max_threads() * thresh1d * kstep,
        alignment
    );
    for (Py_ssize_t Cj = 0; Cj < out_m; Cj += kstep) {
        Py_ssize_t Cjmax2 = std::min<Py_ssize_t>(Cj + kstep, out_m);
        if (kparallel) {
            dense_column_block<true, order>(
                rows, cols, X, d, mu, out, Rglobal.get(), Lglobal.get(), Cj, Cjmax2,
                in_n, out_m, m, n, thresh1d, innerblock, kstep
            );
        } else {
            dense_column_block<false, order>(
                rows, cols, X, d, mu, out, Rglobal.get(), Lglobal.get(), Cj, Cjmax2,
                in_n, out_m, m, n, thresh1d, innerblock, kstep
            );
        }
    }

    // mirror the lower triangle into the upper one
    #pragma omp parallel for if(out_m > 100)
    for (Py_ssize_t Ci = 0; Ci < out_m; Ci++) {
        for (Py_ssize_t Cj = 0; Cj <= Ci; Cj++) {
            out[Cj * out_m + Ci] = out[Ci * out_m + Cj];
        }
    }
}

template <typename Int, typename F>
void _denseC_sandwich(Int* rows, Int* cols, F* X, F* d, const F* mu, F* out,
        Int in_n, Int out_m, Int m, Int n, Int thresh1d, Int kratio, Int innerblock)
{
    dense_sandwich<Order::C>(rows, cols, X, d, mu, out, in_n, out_m, m, n, thresh1d, kratio, innerblock);
}

template <typename Int, typename F>
void _denseF_sandwich(Int* rows, Int* cols, F* X, F* d, const F* mu, F* out,
        Int in_n, Int out_m, Int m, Int n, Int thresh1d, Int kratio, Int innerblock)
{
    dense_sandwich<Order::F>(rows, cols, X, d, mu, out, in_n, out_m, m, n, thresh1d, kratio, innerblock);
}

// ---------------------------------------------------------------------------
// Matrix-vector products.
//
// The kernels below compute with (X - mu) for a per-column center mu,
// subtracting it from each entry as it is read. Expanding into X @ v - mu @ v
// instead would cancel when mu is large relative to the spread of X. Passing
// nullptr for rows or cols selects all of them. The inner loops run along the
// contiguous direction of X (rows for F order, columns for C order) and are
// vectorized if that direction is not restricted; otherwise they fall back to
// indexed loads.
// ---------------------------------------------------------------------------

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
    simd_t<F> csimd[4];
    simd_t<F> asimd[4];
    for (int t = 0; t < 4; t++) {
        csimd[t] = xs::XSIMD_BROADCAST(c[t]);
        asimd[t] = xs::XSIMD_BROADCAST(a[t]);
    }
    for (; k + simd_size <= len; k += simd_size) {
        auto yk = xs::load_unaligned(y + k);
        for (int t = 0; t < 4; t++) {
            yk = xs::fma(xs::load_unaligned(x[t] + k) - csimd[t], asimd[t], yk);
        }
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

// Longer row blocks for F order mean fewer column switches per block.
template <Order order>
constexpr Py_ssize_t matvec_rowblocksize = (order == Order::F) ? 4096 : 1024;

// out[cols] = (X - mu)[rows, cols].T @ v[rows]
template <Order order, typename Int, typename F>
void dense_rmatvec(const Int* rows, const Int* cols, const F* X, const F* v,
                   const F* mu, F* out, Int n_rows, Int n_cols, Int m, Int n)
{
    constexpr std::size_t alignment = simd_size_v<F> * sizeof(F);

    auto outglobal = make_aligned_unique<F>(omp_get_max_threads() * n_cols, alignment);

    constexpr Py_ssize_t rowblocksize = matvec_rowblocksize<order>;

    // Starting threads costs more than a small product.
    #pragma omp parallel for if((Py_ssize_t) n_rows * n_cols > 100000)
    for (Py_ssize_t Ci = 0; Ci < n_rows; Ci += rowblocksize) {
        Py_ssize_t Cimax = std::min<Py_ssize_t>(Ci + rowblocksize, n_rows);

        F* outlocal = &outglobal.get()[omp_get_thread_num() * n_cols];

        if constexpr (order == Order::F) {
            // each column is contiguous: one dot product per column
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
        } else {
            // each row is contiguous: scale the row block by v and sum
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
        }

        for (Py_ssize_t Cj = 0; Cj < n_cols; Cj++) {
            #pragma omp atomic
            out[Cj] += outlocal[Cj];
        }
    }
}

template <typename Int, typename F>
void _denseC_rmatvec(Int* rows, Int* cols, F* X, F* v, const F* mu, F* out,
        Int n_rows, Int n_cols, Int m, Int n)
{
    dense_rmatvec<Order::C>(rows, cols, X, v, mu, out, n_rows, n_cols, m, n);
}

template <typename Int, typename F>
void _denseF_rmatvec(Int* rows, Int* cols, F* X, F* v, const F* mu, F* out,
        Int n_rows, Int n_cols, Int m, Int n)
{
    dense_rmatvec<Order::F>(rows, cols, X, v, mu, out, n_rows, n_cols, m, n);
}

// out[rows] = (X - mu)[rows, cols] @ v[cols]
template <Order order, typename Int, typename F>
void dense_matvec(const Int* rows, const Int* cols, const F* X, const F* v,
                  const F* mu, F* out, Int n_rows, Int n_cols, Int m, Int n)
{
    constexpr Py_ssize_t rowblocksize = matvec_rowblocksize<order>;

    // Starting threads costs more than a small product.
    #pragma omp parallel for if((Py_ssize_t) n_rows * n_cols > 100000)
    for (Py_ssize_t Ci = 0; Ci < n_rows; Ci += rowblocksize) {
        Py_ssize_t Cimax = std::min<Py_ssize_t>(Ci + rowblocksize, n_rows);
        if constexpr (order == Order::F) {
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
        } else {
            // each row is contiguous: one dot product per row
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
        }
    }
}

template <typename Int, typename F>
void _denseC_matvec(Int* rows, Int* cols, F* X, F* v, const F* mu, F* out,
        Int n_rows, Int n_cols, Int m, Int n)
{
    dense_matvec<Order::C>(rows, cols, X, v, mu, out, n_rows, n_cols, m, n);
}

template <typename Int, typename F>
void _denseF_matvec(Int* rows, Int* cols, F* X, F* v, const F* mu, F* out,
        Int n_rows, Int n_cols, Int m, Int n)
{
    dense_matvec<Order::F>(rows, cols, X, v, mu, out, n_rows, n_cols, m, n);
}
