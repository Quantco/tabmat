#pragma once

#include <vector>
#include <omp.h>

#include "order.h"

// Each kernel comes in a "fast" and a "complex" flavour. The complex one
// supports dropping the first category (indices shift by one and index -1 is
// skipped); the fast one is the same kernel with that check compiled out.

// ---------------------------------------------------------------------------
// res[indices[i]] += other[i] for every row, i.e. res = cat.T @ other.
// Each thread scatters into its own copy of res, then the copies are summed.
// ---------------------------------------------------------------------------
template <bool drop, typename Int, typename F>
void transpose_matvec_all_rows(
    Int n_rows,
    const Int* indices,
    const F* other,
    F* res,
    Int res_size,
    bool drop_first
) {
    int num_threads = omp_get_max_threads();
    std::vector<F> all_res(num_threads * res_size, 0.0);
    #pragma omp parallel shared(all_res)
    {
        int tid = omp_get_thread_num();
        F* res_slice = &all_res[tid * res_size];
        #pragma omp for
        for (Py_ssize_t i = 0; i < n_rows; i++) {
            if constexpr (drop) {
                Py_ssize_t col_idx = indices[i] - drop_first;
                if (col_idx >= 0) {
                    res_slice[col_idx] += other[i];
                }
            } else {
                res_slice[indices[i]] += other[i];
            }
        }
        #pragma omp for
        for (Py_ssize_t i = 0; i < res_size; ++i) {
            for (int tid = 0; tid < num_threads; ++tid) {
                res[i] += all_res[tid * res_size + i];
            }
        }
    }
}

template <typename Int, typename F>
void _transpose_matvec_all_rows_fast(
    Int n_rows, const Int* indices, F* other, F* res, Int res_size
) {
    transpose_matvec_all_rows<false>(n_rows, indices, other, res, res_size, false);
}

template <typename Int, typename F>
void _transpose_matvec_all_rows_complex(
    Int n_rows, const Int* indices, F* other, F* res, Int res_size, bool drop_first
) {
    transpose_matvec_all_rows<true>(n_rows, indices, other, res, res_size, drop_first);
}

// ---------------------------------------------------------------------------
// res[i, j] += d[k] for every row k with i_indices[k] == i, j_indices[k] == j,
// i.e. res = cat_i.T @ diag(d) @ cat_j, res being res_n_col wide, row-major.
// ---------------------------------------------------------------------------
template <bool drop, typename Int, typename F>
void sandwich_cat_cat(
    const F* d,
    const Int* i_indices,
    const Int* j_indices,
    const Int* rows,
    Int len_rows,
    F* res,
    Int res_n_col,
    Int res_size,
    bool i_drop_first,
    bool j_drop_first
)
{
    #pragma omp parallel
    {
        std::vector<F> restemp(res_size, 0.0);
        #pragma omp for
        for (Py_ssize_t k_idx = 0; k_idx < len_rows; k_idx++) {
            Int k = rows[k_idx];

            Int i = i_indices[k];
            Int j = j_indices[k];
            if constexpr (drop) {
                i -= i_drop_first;
                if (i < 0) {
                    continue;
                }
                j -= j_drop_first;
                if (j < 0) {
                    continue;
                }
            }

            restemp[(Py_ssize_t) i * res_n_col + j] += d[k];
        }
        for (Py_ssize_t i = 0; i < res_size; i++) {
            #pragma omp atomic
            res[i] += restemp[i];
        }
    }
}

template <typename Int, typename F>
void _sandwich_cat_cat_fast(
    F* d, const Int* i_indices, const Int* j_indices, Int* rows, Int len_rows,
    F* res, Int res_n_col, Int res_size
)
{
    sandwich_cat_cat<false>(
        d, i_indices, j_indices, rows, len_rows, res, res_n_col, res_size, false, false
    );
}

template <typename Int, typename F>
void _sandwich_cat_cat_complex(
    F* d, const Int* i_indices, const Int* j_indices, Int* rows, Int len_rows,
    F* res, Int res_n_col, Int res_size, bool i_drop_first, bool j_drop_first
)
{
    sandwich_cat_cat<true>(
        d, i_indices, j_indices, rows, len_rows, res, res_n_col, res_size,
        i_drop_first, j_drop_first
    );
}

// ---------------------------------------------------------------------------
// res[i, :] += d[k] * (mat_j - mat_j_mu)[k, j_cols] for every row k with
// indices[k] == i, i.e. res = cat.T @ diag(d) @ (mat_j - mat_j_mu)[:, j_cols],
// res being row-major with len_j_cols columns. mat_j is mat_j_nrow x
// mat_j_ncol in the given order. The optional per-column center mat_j_mu
// (nullptr means none) is subtracted from mat_j as it is read.
// ---------------------------------------------------------------------------
template <Order order, bool drop, typename Int, typename F>
void sandwich_cat_dense(
    const F* d,
    const Int* indices,
    const Int* rows,
    Int len_rows,
    const Int* j_cols,
    Int len_j_cols,
    F* res,
    Int res_size,
    const F* mat_j,
    const F* mat_j_mu,
    Int mat_j_nrow,
    Int mat_j_ncol,
    bool drop_first
    )
{
    #pragma omp parallel
    {
        std::vector<F> restemp(res_size, 0.0);
        #pragma omp for
        for (Py_ssize_t k_idx = 0; k_idx < len_rows; k_idx++) {
            Py_ssize_t k = rows[k_idx];
            // MAYBE TODO: explore whether the column restriction slows things down a
            // lot, particularly if not restricting the columns allows using SIMD
            // instructions
            // MAYBE TODO: explore whether swapping the loop order for F-ordered mat_j
            // is useful.
            Py_ssize_t i = indices[k];
            if constexpr (drop) {
                i -= drop_first;
                if (i < 0) {
                    continue;
                }
            }
            for (Py_ssize_t j_idx = 0; j_idx < len_j_cols; j_idx++) {
                Py_ssize_t j = j_cols[j_idx];
                F value = load_X<order>(mat_j, k, j, mat_j_nrow, mat_j_ncol);
                if (mat_j_mu != nullptr) {
                    value -= mat_j_mu[j];
                }
                restemp[i * len_j_cols + j_idx] += d[k] * value;
            }
        }
        for (Py_ssize_t i = 0; i < res_size; i++) {
            #pragma omp atomic
            res[i] += restemp[i];
        }
    }
}

template <typename Int, typename F>
void _sandwich_cat_denseC_fast(
    F* d, const Int* indices, Int* rows, Int len_rows, Int* j_cols, Int len_j_cols,
    F* res, Int res_size, F* mat_j, const F* mat_j_mu, Int mat_j_nrow, Int mat_j_ncol
    )
{
    sandwich_cat_dense<Order::C, false>(
        d, indices, rows, len_rows, j_cols, len_j_cols, res, res_size,
        mat_j, mat_j_mu, mat_j_nrow, mat_j_ncol, false
    );
}

template <typename Int, typename F>
void _sandwich_cat_denseF_fast(
    F* d, const Int* indices, Int* rows, Int len_rows, Int* j_cols, Int len_j_cols,
    F* res, Int res_size, F* mat_j, const F* mat_j_mu, Int mat_j_nrow, Int mat_j_ncol
    )
{
    sandwich_cat_dense<Order::F, false>(
        d, indices, rows, len_rows, j_cols, len_j_cols, res, res_size,
        mat_j, mat_j_mu, mat_j_nrow, mat_j_ncol, false
    );
}

template <typename Int, typename F>
void _sandwich_cat_denseC_complex(
    F* d, const Int* indices, Int* rows, Int len_rows, Int* j_cols, Int len_j_cols,
    F* res, Int res_size, F* mat_j, const F* mat_j_mu, Int mat_j_nrow, Int mat_j_ncol, bool drop_first
    )
{
    sandwich_cat_dense<Order::C, true>(
        d, indices, rows, len_rows, j_cols, len_j_cols, res, res_size,
        mat_j, mat_j_mu, mat_j_nrow, mat_j_ncol, drop_first
    );
}

template <typename Int, typename F>
void _sandwich_cat_denseF_complex(
    F* d, const Int* indices, Int* rows, Int len_rows, Int* j_cols, Int len_j_cols,
    F* res, Int res_size, F* mat_j, const F* mat_j_mu, Int mat_j_nrow, Int mat_j_ncol, bool drop_first
    )
{
    sandwich_cat_dense<Order::F, true>(
        d, indices, rows, len_rows, j_cols, len_j_cols, res, res_size,
        mat_j, mat_j_mu, mat_j_nrow, mat_j_ncol, drop_first
    );
}
