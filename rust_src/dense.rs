//! Dense matrix operations for tabmat.
//!
//! This module provides high-performance implementations of dense matrix operations
//! commonly used in generalized linear model (GLM) fitting. The operations are
//! optimized for both row-major (C-order) and column-major (F-order) memory layouts.
//!
//! # Key Operations
//!
//! - [`dense_sandwich`]: Computes `X.T @ diag(d) @ X` - the core Hessian computation
//! - [`dense_matvec`]: Computes `X @ v` - forward matrix-vector product
//! - [`dense_rmatvec`]: Computes `X.T @ v` - transpose matrix-vector product
//! - [`transpose_square_dot_weights`]: Computes column-wise weighted squared sums
//!
//! # Performance Strategy
//!
//! The sandwich product uses a hybrid approach:
//! - **BLAS dsyrk** for square-ish matrices (exploits symmetry, highly optimized)
//! - **BLIS-style blocking** with true k-parallelism for tall matrices
//!
//! The blocking parameters follow BLIS conventions:
//! - `THRESH1D`: Block size for output dimensions (32)
//! - `KRATIO`: Multiplier for k-dimension blocking (16)
//! - `INNERBLOCK`: Register blocking size (128)
//!
//! SIMD vectorization using `f64x4` (4-wide double precision) is applied to
//! inner loops where applicable.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use wide::f64x4;

/// Computes the dense sandwich product: `X.T @ diag(d) @ X`.
///
/// This is the core operation for computing the Hessian matrix in GLM fitting.
/// The result is symmetric, so only the upper triangle is computed and then
/// mirrored to the lower triangle.
///
/// # Arguments
///
/// * `x` - Input matrix X (n_total_rows × n_total_cols)
/// * `d` - Diagonal weight vector (length n_total_rows)
/// * `rows` - Row indices to include in the computation
/// * `cols` - Column indices to include in the computation
///
/// # Returns
///
/// A symmetric matrix of shape (len(cols), len(cols)) containing the sandwich product.
///
/// # Algorithm Selection
///
/// - For square-ish matrices (n_rows ≥ 500, n_cols ≥ 10): Uses BLAS dsyrk
/// - For very tall matrices (n_rows > 100 × n_cols): Uses k-parallel BLIS blocking
#[pyfunction]
#[pyo3(signature = (x, d, rows, cols))]
pub fn dense_sandwich<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x_arr = x.as_array();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();

    let out_m = cols_slice.len();
    let in_n = rows_slice.len();

    if in_n == 0 || out_m == 0 {
        return PyArray2::zeros_bound(py, [out_m, out_m], false);
    }

    // Get matrix layout info
    let strides = x_arr.strides();
    let is_c_order = strides[1] == 1;
    let (n_total_rows, n_total_cols) = (x_arr.nrows(), x_arr.ncols());

    // Decision logic for tall vs square-ish matrices:
    // - Tall matrices (large in_n, small out_m): use BLIS with true k-parallelism
    // - Square-ish matrices: use BLAS dsyrk
    //
    // For tall matrices, the output is small enough (out_m^2) that we can afford
    // one output buffer per k-block, enabling true parallel accumulation.
    // Threshold: if in_n / out_m > 100 (very tall), use BLIS k-parallel
    let is_tall = in_n > 100 * out_m && out_m <= 200;

    if !is_tall && in_n >= 500 && out_m >= 10 {
        return dense_sandwich_blas(py, &x_arr, d_slice, rows_slice, cols_slice);
    }

    // Use BLIS-style approach with true k-parallelism for tall matrices
    let mut out = vec![0.0f64; out_m * out_m];

    if is_c_order {
        if let Some(x_slice) = x_arr.as_slice() {
            if is_tall {
                dense_sandwich_tall_c(
                    x_slice,
                    d_slice,
                    rows_slice,
                    cols_slice,
                    &mut out,
                    n_total_cols,
                    in_n,
                    out_m,
                );
            } else {
                dense_sandwich_blis_c(
                    x_slice,
                    d_slice,
                    rows_slice,
                    cols_slice,
                    &mut out,
                    n_total_cols,
                    in_n,
                    out_m,
                );
            }
        }
    } else if let Some(x_slice) = x_arr.as_slice_memory_order() {
        // F-order: use as_slice_memory_order() which gives column-major data
        if is_tall {
            dense_sandwich_tall_f(
                x_slice,
                d_slice,
                rows_slice,
                cols_slice,
                &mut out,
                n_total_rows,
                in_n,
                out_m,
            );
        } else {
            dense_sandwich_blis_f(
                x_slice,
                d_slice,
                rows_slice,
                cols_slice,
                &mut out,
                n_total_rows,
                in_n,
                out_m,
            );
        }
    }

    // Fill lower triangle by symmetry
    for i in 0..out_m {
        for j in (i + 1)..out_m {
            out[j * out_m + i] = out[i * out_m + j];
        }
    }

    let out_2d: Vec<Vec<f64>> = (0..out_m)
        .map(|i| out[i * out_m..(i + 1) * out_m].to_vec())
        .collect();

    PyArray2::from_vec2_bound(py, &out_2d).unwrap()
}

/// BLAS-accelerated dense sandwich for square-ish matrices.
///
/// Uses BLAS `dsyrk` (symmetric rank-k update) which is highly optimized for
/// computing `A.T @ A` style products. The input is pre-scaled by `sqrt(d)`
/// so that `dsyrk(X_scaled)` computes `X.T @ diag(d) @ X`.
fn dense_sandwich_blas<'py>(
    py: Python<'py>,
    x_arr: &ndarray::ArrayView2<f64>,
    d_slice: &[f64],
    rows_slice: &[i32],
    cols_slice: &[i32],
) -> Bound<'py, PyArray2<f64>> {
    let n_rows = rows_slice.len();
    let out_m = cols_slice.len();

    let mut x_sub = vec![0.0; n_rows * out_m];
    let cols: Vec<usize> = cols_slice.iter().map(|&c| c as usize).collect();

    if let Some(x_slice) = x_arr.as_slice() {
        let n_total_cols = x_arr.ncols();
        x_sub
            .par_chunks_mut(out_m)
            .enumerate()
            .for_each(|(i, out_row)| {
                let row_idx = rows_slice[i] as usize;
                let d_sqrt = d_slice[row_idx].sqrt();
                let row_offset = row_idx * n_total_cols;
                for (j, &col_j) in cols.iter().enumerate() {
                    out_row[j] = x_slice[row_offset + col_j] * d_sqrt;
                }
            });
    } else {
        for (i, &row_i) in rows_slice.iter().enumerate() {
            let row_idx = row_i as usize;
            let d_sqrt = d_slice[row_idx].sqrt();
            let out_offset = i * out_m;
            for (j, &col_j) in cols.iter().enumerate() {
                x_sub[out_offset + j] = x_arr[[row_idx, col_j]] * d_sqrt;
            }
        }
    }

    let mut result = vec![0.0; out_m * out_m];

    unsafe {
        cblas_sys::cblas_dsyrk(
            cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
            cblas_sys::CBLAS_UPLO::CblasUpper,
            cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
            out_m as i32,
            n_rows as i32,
            1.0,
            x_sub.as_ptr(),
            out_m as i32,
            0.0,
            result.as_mut_ptr(),
            out_m as i32,
        );
    }

    // Fill lower triangle from upper
    for i in 0..out_m {
        for j in (i + 1)..out_m {
            result[j * out_m + i] = result[i * out_m + j];
        }
    }

    let out_2d: Vec<Vec<f64>> = (0..out_m)
        .map(|i| result[i * out_m..(i + 1) * out_m].to_vec())
        .collect();

    PyArray2::from_vec2_bound(py, &out_2d).unwrap()
}

// =============================================================================
// BLIS-style blocking parameters
// =============================================================================
// These parameters control cache-blocking behavior and match the C++ defaults.
// See: "BLIS: A Framework for Rapidly Instantiating BLAS Functionality"

/// Block size for output matrix dimensions (i and j loops).
/// Small enough to fit L1 cache, large enough to amortize loop overhead.
const THRESH1D: usize = 32;

/// Ratio of k-dimension block size to output block size.
/// k-blocks are KRATIO × THRESH1D = 512 rows.
const KRATIO: usize = 16;

/// Register blocking size for innermost loops.
/// Controls micro-kernel tile size for better instruction-level parallelism.
const INNERBLOCK: usize = 128;

/// SIMD vector width for f64 operations (f64x4 = 4 doubles = 256 bits).
const SIMD_WIDTH: usize = 4;

/// Optimized dense sandwich for tall matrices (C-order) with true k-parallelism.
/// Uses BLAS dsyrk for each k-block, with parallel fold/reduce.
fn dense_sandwich_tall_c(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    stride: usize, // total columns in X (stride for C-order)
    in_n: usize,
    out_m: usize,
) {
    let kblock = KRATIO * THRESH1D; // 512 rows per k-block
    let n_kblocks = (in_n + kblock - 1) / kblock;

    // Pre-convert cols to usize
    let cols_usize: Vec<usize> = cols.iter().map(|&c| c as usize).collect();
    let out_size = out_m * out_m;

    // Thread-local struct to hold pre-allocated buffers
    struct ThreadLocal {
        out_buf: Vec<f64>,
        x_sub: Vec<f64>, // k_size x out_m, scaled by sqrt(d)
    }

    let x_sub_size = kblock * out_m;

    // Use parallel fold/reduce with BLAS dsyrk for each k-block
    let result = (0..n_kblocks)
        .into_par_iter()
        .fold(
            || ThreadLocal {
                out_buf: vec![0.0f64; out_size],
                x_sub: vec![0.0f64; x_sub_size],
            },
            |mut tl, kb| {
                let rk = kb * kblock;
                let rkmax = (rk + kblock).min(in_n);
                let k_size = rkmax - rk;

                // Build x_sub: k_size x out_m matrix with rows scaled by sqrt(d)
                // x_sub[k][j] = sqrt(d[row_k]) * X[row_k, col_j]
                for (k_local, rkk) in (rk..rkmax).enumerate() {
                    let kk = rows[rkk] as usize;
                    let d_sqrt = d[kk].sqrt();
                    for (j_local, &col_j) in cols_usize.iter().enumerate() {
                        tl.x_sub[k_local * out_m + j_local] = d_sqrt * x[kk * stride + col_j];
                    }
                }

                // Use BLAS dsyrk: C += A.T @ A where A is k_size x out_m
                unsafe {
                    cblas_sys::cblas_dsyrk(
                        cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                        cblas_sys::CBLAS_UPLO::CblasUpper,
                        cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
                        out_m as i32,  // N: size of C
                        k_size as i32, // K: number of rows in A
                        1.0,           // alpha
                        tl.x_sub.as_ptr(),
                        out_m as i32,  // lda
                        1.0,           // beta (accumulate)
                        tl.out_buf.as_mut_ptr(),
                        out_m as i32,  // ldc
                    );
                }

                tl
            },
        )
        .map(|tl| tl.out_buf)
        .reduce(
            || vec![0.0f64; out_size],
            |mut a, b| {
                for i in 0..out_size {
                    a[i] += b[i];
                }
                a
            },
        );

    // Copy result to output (upper triangle only, lower will be filled by caller)
    out[..out_size].copy_from_slice(&result);
}

/// Optimized dense sandwich for tall matrices (F-order) with true k-parallelism.
/// Uses BLAS dsyrk for each k-block, with parallel fold/reduce.
fn dense_sandwich_tall_f(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    stride: usize, // total rows in X (stride for F-order)
    in_n: usize,
    out_m: usize,
) {
    let kblock = KRATIO * THRESH1D;
    let n_kblocks = (in_n + kblock - 1) / kblock;

    let cols_usize: Vec<usize> = cols.iter().map(|&c| c as usize).collect();
    let out_size = out_m * out_m;

    struct ThreadLocal {
        out_buf: Vec<f64>,
        x_sub: Vec<f64>,
    }

    let x_sub_size = kblock * out_m;

    let result = (0..n_kblocks)
        .into_par_iter()
        .fold(
            || ThreadLocal {
                out_buf: vec![0.0f64; out_size],
                x_sub: vec![0.0f64; x_sub_size],
            },
            |mut tl, kb| {
                let rk = kb * kblock;
                let rkmax = (rk + kblock).min(in_n);
                let k_size = rkmax - rk;

                // Build x_sub: k_size x out_m matrix with rows scaled by sqrt(d)
                // F-order: X[row, col] = x[col * stride + row]
                for (k_local, rkk) in (rk..rkmax).enumerate() {
                    let kk = rows[rkk] as usize;
                    let d_sqrt = d[kk].sqrt();
                    for (j_local, &col_j) in cols_usize.iter().enumerate() {
                        tl.x_sub[k_local * out_m + j_local] = d_sqrt * x[col_j * stride + kk];
                    }
                }

                // Use BLAS dsyrk: C += A.T @ A where A is k_size x out_m
                unsafe {
                    cblas_sys::cblas_dsyrk(
                        cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                        cblas_sys::CBLAS_UPLO::CblasUpper,
                        cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
                        out_m as i32,
                        k_size as i32,
                        1.0,
                        tl.x_sub.as_ptr(),
                        out_m as i32,
                        1.0,
                        tl.out_buf.as_mut_ptr(),
                        out_m as i32,
                    );
                }

                tl
            },
        )
        .map(|tl| tl.out_buf)
        .reduce(
            || vec![0.0f64; out_size],
            |mut a, b| {
                for i in 0..out_size {
                    a[i] += b[i];
                }
                a
            },
        );

    out[..out_size].copy_from_slice(&result);
}

/// BLIS-style dense sandwich for C-ordered (row-major) matrices.
///
/// Dynamically chooses between k-parallel and i-parallel strategies based on
/// matrix shape. For tall matrices (many rows, few columns), k-parallelism
/// is more efficient. For square-ish matrices, i-parallelism is preferred.
fn dense_sandwich_blis_c(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    m: usize, // total columns in X (stride)
    in_n: usize,
    out_m: usize,
) {
    let kblock = KRATIO * THRESH1D;

    // Decide parallelization strategy based on matrix shape
    // For tall matrices (many rows, few cols): parallelize k dimension
    // For wide matrices: parallelize i dimension
    let kparallel = (in_n / kblock) > (out_m / THRESH1D);

    // Pre-convert cols to usize
    let cols_usize: Vec<usize> = cols.iter().map(|&c| c as usize).collect();

    // Process j-blocks (outer loop)
    for cj in (0..out_m).step_by(kblock) {
        let cjmax = (cj + kblock).min(out_m);

        if kparallel {
            // K-parallel: parallelize over k-blocks, need atomic updates or per-thread buffers
            dense_sandwich_k_parallel_c(
                x, d, rows, &cols_usize, out, m, in_n, out_m, cj, cjmax,
            );
        } else {
            // I-parallel: parallelize over i-blocks
            dense_sandwich_i_parallel_c(
                x, d, rows, &cols_usize, out, m, in_n, out_m, cj, cjmax,
            );
        }
    }
}

/// K-parallel version for tall matrices (C-order)
/// Processes k-blocks sequentially but parallelizes inner i-blocks
fn dense_sandwich_k_parallel_c(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[usize],
    out: &mut [f64],
    m: usize,
    in_n: usize,
    out_m: usize,
    cj: usize,
    cjmax: usize,
) {
    let kblock = KRATIO * THRESH1D;
    let jblock_size = cjmax - cj;

    // Process k-blocks sequentially
    for rk in (0..in_n).step_by(kblock) {
        let rkmax = (rk + kblock).min(in_n);
        let k_size = rkmax - rk;

        // Allocate R buffer: [j_local][k_local]
        let mut r_buf = vec![0.0f64; jblock_size * kblock];

        // Fill R: R[j][k] = d[row] * X[row, col_j]
        for cjj in cj..cjmax {
            let jj = cols[cjj];
            let r_col = &mut r_buf[(cjj - cj) * kblock..];
            for (k_local, rkk) in (rk..rkmax).enumerate() {
                let kk = rows[rkk] as usize;
                r_col[k_local] = d[kk] * x[kk * m + jj];
            }
        }

        // Parallel over i-blocks, collect partial results
        let i_block_results: Vec<(usize, usize, Vec<f64>)> = (cj..out_m)
            .into_par_iter()
            .step_by(THRESH1D)
            .map(|ci| {
                let cimax = (ci + THRESH1D).min(out_m);
                let i_size = cimax - ci;

                // Allocate L buffer: [i_local][k_local]
                let mut l_buf = vec![0.0f64; THRESH1D * kblock];

                // Fill L: L[i][k] = X[row, col_i]
                for cii in ci..cimax {
                    let ii = cols[cii];
                    let l_row = &mut l_buf[(cii - ci) * kblock..];
                    for (k_local, rkk) in (rk..rkmax).enumerate() {
                        let kk = rows[rkk] as usize;
                        l_row[k_local] = x[kk * m + ii];
                    }
                }

                // Compute local block result
                let mut local_out = vec![0.0f64; i_size * jblock_size];
                dense_base_kernel_local(
                    &r_buf,
                    &l_buf,
                    &mut local_out,
                    jblock_size,
                    0,
                    i_size,
                    0,
                    jblock_size,
                    k_size,
                    kblock,
                );

                (ci, cimax, local_out)
            })
            .collect();

        // Accumulate results into output
        for (ci, cimax, local_out) in i_block_results {
            for (i_local, cii) in (ci..cimax).enumerate() {
                for (j_local, cjj) in (cj..cjmax).enumerate() {
                    out[cii * out_m + cjj] += local_out[i_local * jblock_size + j_local];
                }
            }
        }
    }
}

/// I-parallel version for wide matrices (C-order)
fn dense_sandwich_i_parallel_c(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[usize],
    out: &mut [f64],
    m: usize,
    in_n: usize,
    out_m: usize,
    cj: usize,
    cjmax: usize,
) {
    let kblock = KRATIO * THRESH1D;
    let jblock_size = cjmax - cj;

    // Process k-blocks sequentially
    for rk in (0..in_n).step_by(kblock) {
        let rkmax = (rk + kblock).min(in_n);
        let k_size = rkmax - rk;

        // Shared R buffer for all threads: [j_local][k_local]
        let mut r_buf = vec![0.0f64; jblock_size * kblock];

        // Fill R in parallel over j
        r_buf
            .par_chunks_mut(kblock)
            .enumerate()
            .for_each(|(j_local, r_col)| {
                let cjj = cj + j_local;
                if cjj < cjmax {
                    let jj = cols[cjj];
                    for (k_local, rkk) in (rk..rkmax).enumerate() {
                        let kk = rows[rkk] as usize;
                        r_col[k_local] = d[kk] * x[kk * m + jj];
                    }
                }
            });

        // Parallel over i-blocks
        let i_block_results: Vec<(usize, usize, Vec<f64>)> = (cj..out_m)
            .into_par_iter()
            .step_by(THRESH1D)
            .map(|ci| {
                let cimax = (ci + THRESH1D).min(out_m);
                let i_size = cimax - ci;

                // Local L buffer for this i-block
                let mut l_buf = vec![0.0f64; THRESH1D * kblock];

                // Fill L
                for cii in ci..cimax {
                    let ii = cols[cii];
                    let l_row = &mut l_buf[(cii - ci) * kblock..];
                    for (k_local, rkk) in (rk..rkmax).enumerate() {
                        let kk = rows[rkk] as usize;
                        l_row[k_local] = x[kk * m + ii];
                    }
                }

                // Compute local block result
                let mut local_out = vec![0.0f64; i_size * jblock_size];
                dense_base_kernel_local(
                    &r_buf,
                    &l_buf,
                    &mut local_out,
                    jblock_size,
                    0,
                    i_size,
                    0,
                    jblock_size,
                    k_size,
                    kblock,
                );

                (ci, cimax, local_out)
            })
            .collect();

        // Accumulate results into output
        for (ci, cimax, local_out) in i_block_results {
            for (i_local, cii) in (ci..cimax).enumerate() {
                for (j_local, cjj) in (cj..cjmax).enumerate() {
                    out[cii * out_m + cjj] += local_out[i_local * jblock_size + j_local];
                }
            }
        }
    }
}

/// BLIS-style dense sandwich for F-ordered (column-major) matrices
fn dense_sandwich_blis_f(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    n: usize, // total rows in X (stride for F-order)
    in_n: usize,
    out_m: usize,
) {
    let kblock = KRATIO * THRESH1D;
    let kparallel = (in_n / kblock) > (out_m / THRESH1D);
    let cols_usize: Vec<usize> = cols.iter().map(|&c| c as usize).collect();

    for cj in (0..out_m).step_by(kblock) {
        let cjmax = (cj + kblock).min(out_m);

        if kparallel {
            dense_sandwich_k_parallel_f(x, d, rows, &cols_usize, out, n, in_n, out_m, cj, cjmax);
        } else {
            dense_sandwich_i_parallel_f(x, d, rows, &cols_usize, out, n, in_n, out_m, cj, cjmax);
        }
    }
}

/// K-parallel version for tall matrices (F-order)
/// Processes k-blocks sequentially but parallelizes inner i-blocks
fn dense_sandwich_k_parallel_f(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[usize],
    out: &mut [f64],
    n: usize,
    in_n: usize,
    out_m: usize,
    cj: usize,
    cjmax: usize,
) {
    let kblock = KRATIO * THRESH1D;
    let jblock_size = cjmax - cj;

    for rk in (0..in_n).step_by(kblock) {
        let rkmax = (rk + kblock).min(in_n);
        let k_size = rkmax - rk;

        let mut r_buf = vec![0.0f64; jblock_size * kblock];

        // Fill R for F-order: X[row, col] = x[col * n + row]
        for cjj in cj..cjmax {
            let jj = cols[cjj];
            let r_col = &mut r_buf[(cjj - cj) * kblock..];
            for (k_local, rkk) in (rk..rkmax).enumerate() {
                let kk = rows[rkk] as usize;
                r_col[k_local] = d[kk] * x[jj * n + kk];
            }
        }

        let i_block_results: Vec<(usize, usize, Vec<f64>)> = (cj..out_m)
            .into_par_iter()
            .step_by(THRESH1D)
            .map(|ci| {
                let cimax = (ci + THRESH1D).min(out_m);
                let i_size = cimax - ci;

                let mut l_buf = vec![0.0f64; THRESH1D * kblock];

                for cii in ci..cimax {
                    let ii = cols[cii];
                    let l_row = &mut l_buf[(cii - ci) * kblock..];
                    for (k_local, rkk) in (rk..rkmax).enumerate() {
                        let kk = rows[rkk] as usize;
                        l_row[k_local] = x[ii * n + kk];
                    }
                }

                let mut local_out = vec![0.0f64; i_size * jblock_size];
                dense_base_kernel_local(
                    &r_buf,
                    &l_buf,
                    &mut local_out,
                    jblock_size,
                    0,
                    i_size,
                    0,
                    jblock_size,
                    k_size,
                    kblock,
                );

                (ci, cimax, local_out)
            })
            .collect();

        for (ci, cimax, local_out) in i_block_results {
            for (i_local, cii) in (ci..cimax).enumerate() {
                for (j_local, cjj) in (cj..cjmax).enumerate() {
                    out[cii * out_m + cjj] += local_out[i_local * jblock_size + j_local];
                }
            }
        }
    }
}

/// I-parallel version for wide matrices (F-order)
fn dense_sandwich_i_parallel_f(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[usize],
    out: &mut [f64],
    n: usize,
    in_n: usize,
    out_m: usize,
    cj: usize,
    cjmax: usize,
) {
    let kblock = KRATIO * THRESH1D;
    let jblock_size = cjmax - cj;

    for rk in (0..in_n).step_by(kblock) {
        let rkmax = (rk + kblock).min(in_n);
        let k_size = rkmax - rk;

        let mut r_buf = vec![0.0f64; jblock_size * kblock];

        r_buf
            .par_chunks_mut(kblock)
            .enumerate()
            .for_each(|(j_local, r_col)| {
                let cjj = cj + j_local;
                if cjj < cjmax {
                    let jj = cols[cjj];
                    for (k_local, rkk) in (rk..rkmax).enumerate() {
                        let kk = rows[rkk] as usize;
                        r_col[k_local] = d[kk] * x[jj * n + kk];
                    }
                }
            });

        let i_block_results: Vec<(usize, usize, Vec<f64>)> = (cj..out_m)
            .into_par_iter()
            .step_by(THRESH1D)
            .map(|ci| {
                let cimax = (ci + THRESH1D).min(out_m);
                let i_size = cimax - ci;

                let mut l_buf = vec![0.0f64; THRESH1D * kblock];

                for cii in ci..cimax {
                    let ii = cols[cii];
                    let l_row = &mut l_buf[(cii - ci) * kblock..];
                    for (k_local, rkk) in (rk..rkmax).enumerate() {
                        let kk = rows[rkk] as usize;
                        l_row[k_local] = x[ii * n + kk];
                    }
                }

                let mut local_out = vec![0.0f64; i_size * jblock_size];
                dense_base_kernel_local(
                    &r_buf,
                    &l_buf,
                    &mut local_out,
                    jblock_size,
                    0,
                    i_size,
                    0,
                    jblock_size,
                    k_size,
                    kblock,
                );

                (ci, cimax, local_out)
            })
            .collect();

        for (ci, cimax, local_out) in i_block_results {
            for (i_local, cii) in (ci..cimax).enumerate() {
                for (j_local, cjj) in (cj..cjmax).enumerate() {
                    out[cii * out_m + cjj] += local_out[i_local * jblock_size + j_local];
                }
            }
        }
    }
}

/// Inner kernel with local output (for i-parallel).
///
/// Computes `L @ R.T` where L has shape [i_size × k_size] and R has shape [j_size × k_size].
/// Uses 4×4 micro-kernels with SIMD vectorization for the inner dot product.
#[inline]
fn dense_base_kernel_local(
    r: &[f64],
    l: &[f64],
    out: &mut [f64],
    out_stride: usize,
    imin: usize,
    imax: usize,
    jmin: usize,
    jmax: usize,
    k_size: usize,
    kstep: usize,
) {
    for iblock in (imin..imax).step_by(INNERBLOCK) {
        let iblock_max = (iblock + INNERBLOCK).min(imax);
        for jblock in (jmin..jmax).step_by(INNERBLOCK) {
            let jblock_max = (jblock + INNERBLOCK).min(jmax);

            // Process 4x4 blocks
            let mut i = iblock;
            while i + 4 <= iblock_max {
                let mut j = jblock;
                while j + 4 <= jblock_max {
                    let mut accum = [[0.0f64; 4]; 4];

                    let mut k = 0;
                    let k_simd_end = (k_size / SIMD_WIDTH) * SIMD_WIDTH;

                    while k < k_simd_end {
                        let l0 = f64x4::from(&l[i * kstep + k..i * kstep + k + 4]);
                        let l1 = f64x4::from(&l[(i + 1) * kstep + k..(i + 1) * kstep + k + 4]);
                        let l2 = f64x4::from(&l[(i + 2) * kstep + k..(i + 2) * kstep + k + 4]);
                        let l3 = f64x4::from(&l[(i + 3) * kstep + k..(i + 3) * kstep + k + 4]);

                        let r0 = f64x4::from(&r[j * kstep + k..j * kstep + k + 4]);
                        let r1 = f64x4::from(&r[(j + 1) * kstep + k..(j + 1) * kstep + k + 4]);
                        let r2 = f64x4::from(&r[(j + 2) * kstep + k..(j + 2) * kstep + k + 4]);
                        let r3 = f64x4::from(&r[(j + 3) * kstep + k..(j + 3) * kstep + k + 4]);

                        for (ir, lv) in [l0, l1, l2, l3].iter().enumerate() {
                            for (jr, rv) in [r0, r1, r2, r3].iter().enumerate() {
                                let prod = *lv * *rv;
                                let arr = prod.to_array();
                                accum[ir][jr] += arr[0] + arr[1] + arr[2] + arr[3];
                            }
                        }

                        k += SIMD_WIDTH;
                    }

                    while k < k_size {
                        for ir in 0..4 {
                            let lv = l[(i + ir) * kstep + k];
                            for jr in 0..4 {
                                let rv = r[(j + jr) * kstep + k];
                                accum[ir][jr] += lv * rv;
                            }
                        }
                        k += 1;
                    }

                    for ir in 0..4 {
                        for jr in 0..4 {
                            out[(i + ir) * out_stride + (j + jr)] += accum[ir][jr];
                        }
                    }

                    j += 4;
                }

                while j < jblock_max {
                    let mut accum = [0.0f64; 4];
                    for k in 0..k_size {
                        let rv = r[j * kstep + k];
                        for ir in 0..4 {
                            accum[ir] += l[(i + ir) * kstep + k] * rv;
                        }
                    }
                    for ir in 0..4 {
                        out[(i + ir) * out_stride + j] += accum[ir];
                    }
                    j += 1;
                }

                i += 4;
            }

            while i < iblock_max {
                for j in jblock..jblock_max {
                    let mut accum = 0.0f64;
                    for k in 0..k_size {
                        accum += l[i * kstep + k] * r[j * kstep + k];
                    }
                    out[i * out_stride + j] += accum;
                }
                i += 1;
            }
        }
    }
}

/// Computes the dense transpose-vector product: `X.T @ v`.
///
/// Returns a row vector (1 × n_cols) representing the sum of rows of X weighted by v.
///
/// # Arguments
///
/// * `x` - Input matrix X
/// * `v` - Weight vector (length matches len(rows))
/// * `rows` - Row indices to include
/// * `cols` - Column indices to include
///
/// # Performance
///
/// - C-order matrices: parallelizes over row blocks (256 rows per block)
/// - F-order matrices: parallelizes over columns (contiguous column access)
#[pyfunction]
#[pyo3(signature = (x, v, rows, cols))]
pub fn dense_rmatvec<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    v: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x_arr = x.as_array();
    let v_slice = v.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();

    let n_rows = rows_slice.len();
    let n_cols = cols_slice.len();

    if n_rows == 0 || n_cols == 0 {
        let out = vec![0.0; n_cols];
        let result_2d: Vec<Vec<f64>> = vec![out];
        return PyArray2::from_vec2_bound(py, &result_2d).unwrap();
    }

    // Get matrix info
    let strides = x_arr.strides();
    let is_c_order = strides[1] == 1;
    let x_ncols = x_arr.ncols();
    let x_nrows = x_arr.nrows();

    // Pre-convert cols to usize
    let cols_usize: Vec<usize> = cols_slice.iter().map(|&c| c as usize).collect();

    let result = if is_c_order {
        // C-order: parallelize over row blocks (better cache locality)
        const ROW_BLOCK: usize = 256;
        let n_row_blocks = (n_rows + ROW_BLOCK - 1) / ROW_BLOCK;

        let x_slice = x_arr.as_slice().unwrap();

        (0..n_row_blocks)
            .into_par_iter()
            .fold(
                || vec![0.0f64; n_cols],
                |mut out_local, rb| {
                    let r_start = rb * ROW_BLOCK;
                    let r_end = (r_start + ROW_BLOCK).min(n_rows);

                    for r_idx in r_start..r_end {
                        let row = rows_slice[r_idx] as usize;
                        let v_val = v_slice[r_idx];
                        let row_start = row * x_ncols;

                        for (j, &col) in cols_usize.iter().enumerate() {
                            out_local[j] += x_slice[row_start + col] * v_val;
                        }
                    }

                    out_local
                },
            )
            .reduce(
                || vec![0.0f64; n_cols],
                |mut a, b| {
                    for j in 0..n_cols {
                        a[j] += b[j];
                    }
                    a
                },
            )
    } else {
        // F-order: parallelize over columns (contiguous column access)
        let x_slice = x_arr.as_slice_memory_order().unwrap();

        cols_usize
            .par_iter()
            .map(|&col| {
                let col_start = col * x_nrows;
                let mut accum = 0.0;

                for (r_idx, &row_idx) in rows_slice.iter().enumerate() {
                    let row = row_idx as usize;
                    accum += x_slice[col_start + row] * v_slice[r_idx];
                }

                accum
            })
            .collect()
    };

    let result_2d: Vec<Vec<f64>> = vec![result];
    PyArray2::from_vec2_bound(py, &result_2d).unwrap()
}

/// Computes the dense matrix-vector product: `X @ v`.
///
/// Returns a row vector (1 × n_rows) where each element is the dot product
/// of a row of X with v.
///
/// # Arguments
///
/// * `x` - Input matrix X
/// * `v` - Input vector (length matches len(cols))
/// * `rows` - Row indices to include
/// * `cols` - Column indices to include
///
/// # Performance
///
/// Uses row block parallelization (256 rows per block) with direct writes
/// to output buffer using `par_chunks_mut` for zero-copy parallel output.
#[pyfunction]
#[pyo3(signature = (x, v, rows, cols))]
pub fn dense_matvec<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    v: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x_arr = x.as_array();
    let v_slice = v.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();

    let n_rows = rows_slice.len();
    let n_cols = cols_slice.len();

    if n_rows == 0 || n_cols == 0 {
        let out = vec![0.0; n_rows];
        let result_2d: Vec<Vec<f64>> = vec![out];
        return PyArray2::from_vec2_bound(py, &result_2d).unwrap();
    }

    // Get matrix info
    let strides = x_arr.strides();
    let is_c_order = strides[1] == 1;
    let x_ncols = x_arr.ncols();
    let x_nrows = x_arr.nrows();

    // Pre-convert cols to usize
    let cols_usize: Vec<usize> = cols_slice.iter().map(|&c| c as usize).collect();

    // Pre-allocate output and use par_chunks_mut for zero-copy parallel writes
    let mut out = vec![0.0f64; n_rows];
    const ROW_BLOCK: usize = 256;

    if is_c_order {
        if let Some(x_slice) = x_arr.as_slice() {
            out.par_chunks_mut(ROW_BLOCK)
                .enumerate()
                .for_each(|(block_idx, out_chunk)| {
                    let r_start = block_idx * ROW_BLOCK;
                    for (local_idx, out_val) in out_chunk.iter_mut().enumerate() {
                        let r_idx = r_start + local_idx;
                        let row = unsafe { *rows_slice.get_unchecked(r_idx) } as usize;
                        let row_start = row * x_ncols;
                        let mut accum = 0.0;
                        // v has length n_cols (pre-selected by Python wrapper)
                        for j in 0..n_cols {
                            unsafe {
                                let col = *cols_usize.get_unchecked(j);
                                accum += x_slice.get_unchecked(row_start + col)
                                    * v_slice.get_unchecked(j);
                            }
                        }
                        *out_val = accum;
                    }
                });
        } else {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(r_idx, out_val)| {
                    let row = rows_slice[r_idx] as usize;
                    let mut accum = 0.0;
                    for (j, &col) in cols_usize.iter().enumerate() {
                        accum += x_arr[[row, col]] * v_slice[j];
                    }
                    *out_val = accum;
                });
        }
    } else if let Some(x_slice) = x_arr.as_slice_memory_order() {
        out.par_chunks_mut(ROW_BLOCK)
            .enumerate()
            .for_each(|(block_idx, out_chunk)| {
                let r_start = block_idx * ROW_BLOCK;
                for (local_idx, out_val) in out_chunk.iter_mut().enumerate() {
                    let r_idx = r_start + local_idx;
                    let row = unsafe { *rows_slice.get_unchecked(r_idx) } as usize;
                    let mut accum = 0.0;
                    for j in 0..n_cols {
                        unsafe {
                            let col = *cols_usize.get_unchecked(j);
                            accum += x_slice.get_unchecked(col * x_nrows + row)
                                * v_slice.get_unchecked(j);
                        }
                    }
                    *out_val = accum;
                }
            });
    } else {
        out.par_iter_mut()
            .enumerate()
            .for_each(|(r_idx, out_val)| {
                let row = rows_slice[r_idx] as usize;
                let mut accum = 0.0;
                for (j, &col) in cols_usize.iter().enumerate() {
                    accum += x_arr[[row, col]] * v_slice[j];
                }
                *out_val = accum;
            });
    }

    let result_2d: Vec<Vec<f64>> = vec![out];
    PyArray2::from_vec2_bound(py, &result_2d).unwrap()
}

/// Computes column-wise weighted squared deviations from shift values.
///
/// For each column j, computes: `sum_i weights[i] * (X[i,j] - shift[j])^2`
///
/// This operation is used for computing variance-like statistics in GLM fitting.
///
/// # Arguments
///
/// * `x` - Input matrix X (n_rows × n_cols)
/// * `weights` - Row weights (length n_rows)
/// * `shift` - Shift values per column (length n_cols)
///
/// # Returns
///
/// A 1D array of length n_cols with the weighted squared sums.
#[pyfunction(name = "dense_transpose_square_dot_weights")]
#[pyo3(signature = (x, weights, shift))]
pub fn transpose_square_dot_weights<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    shift: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let weights_slice = weights.as_slice().unwrap();
    let shift_slice = shift.as_slice().unwrap();
    
    let nrows = weights_slice.len();
    let ncols = x.shape()[1];
    
    let mut out = vec![0.0; ncols];
    
    // Parallel over columns
    out.par_iter_mut().enumerate().for_each(|(j, out_val)| {
        let mut accum = 0.0;
        for i in 0..nrows {
            let diff = x[[i, j]] - shift_slice[j];
            accum += weights_slice[i] * diff * diff;
        }
        *out_val = accum;
    });
    
    PyArray1::from_vec_bound(py, out)
}
