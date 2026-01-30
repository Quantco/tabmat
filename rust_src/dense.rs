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
//! The sandwich product uses k-parallel BLAS dsyrk with blocking:
//! - Rows are divided into k-blocks of size KRATIO × THRESH1D (512 rows)
//! - Each k-block builds a scaled submatrix and uses BLAS dsyrk
//! - Results are accumulated using parallel fold/reduce
//!
//! The blocking parameters follow BLIS conventions:
//! - `THRESH1D`: Block size for output dimensions (32)
//! - `KRATIO`: Multiplier for k-dimension blocking (16)

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Computes the dense sandwich product: `X.T @ diag(d) @ X`.
///
/// This is the core operation for computing the Hessian matrix in GLM fitting.
/// The result is symmetric, so only the upper triangle is computed and then
/// mirrored to the lower triangle.
///
/// # Arguments
///
/// * `x` - Input matrix X (n_total_rows × n_total_cols)
/// * `d` - Diagonal weight vector (length n_total_rows), may contain negative values
/// * `rows` - Row indices to include in the computation
/// * `cols` - Column indices to include in the computation
///
/// # Returns
///
/// A symmetric matrix of shape (len(cols), len(cols)) containing the sandwich product.
///
/// # Algorithm
///
/// Uses k-parallel BLAS dsyrk with blocking. Rows are divided into k-blocks,
/// each block builds a scaled submatrix and uses BLAS dsyrk, then results
/// are accumulated using parallel fold/reduce. Negative weights are handled
/// by partitioning into positive/negative contributions.
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

    // Check if any weights are negative
    let has_negative_weights = rows_slice.iter().any(|&r| d_slice[r as usize] < 0.0);

    let mut out = vec![0.0f64; out_m * out_m];

    if is_c_order {
        if let Some(x_slice) = x_arr.as_slice() {
            if has_negative_weights {
                dense_sandwich_tall_signed_c(
                    x_slice,
                    d_slice,
                    rows_slice,
                    cols_slice,
                    &mut out,
                    n_total_cols,
                    out_m,
                );
            } else {
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
            }
        }
    } else if let Some(x_slice) = x_arr.as_slice_memory_order() {
        // F-order: use as_slice_memory_order() which gives column-major data
        if has_negative_weights {
            dense_sandwich_tall_signed_f(
                x_slice,
                d_slice,
                rows_slice,
                cols_slice,
                &mut out,
                n_total_rows,
                out_m,
            );
        } else {
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

// =============================================================================
// K-parallel blocking parameters
// =============================================================================

/// Block size for output matrix dimensions.
const THRESH1D: usize = 32;

/// Ratio of k-dimension block size to output block size.
/// k-blocks are KRATIO × THRESH1D = 512 rows.
const KRATIO: usize = 16;

/// K-parallel dense sandwich for C-order matrices.
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

/// Optimized dense sandwich for tall matrices (C-order) with negative weights.
/// Partitions rows into positive/negative, processes each with k-parallel dsyrk.
fn dense_sandwich_tall_signed_c(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    stride: usize,
    out_m: usize,
) {
    // Partition rows into positive and negative weight sets
    let mut pos_rows: Vec<i32> = Vec::new();
    let mut neg_rows: Vec<i32> = Vec::new();
    for &r in rows {
        if d[r as usize] >= 0.0 {
            pos_rows.push(r);
        } else {
            neg_rows.push(r);
        }
    }

    let out_size = out_m * out_m;

    // Process positive rows
    if !pos_rows.is_empty() {
        dense_sandwich_tall_c_inner(x, d, &pos_rows, cols, out, stride, pos_rows.len(), out_m, 1.0);
    }

    // Process negative rows (subtract)
    if !neg_rows.is_empty() {
        let mut neg_out = vec![0.0f64; out_size];
        dense_sandwich_tall_c_inner(
            x,
            d,
            &neg_rows,
            cols,
            &mut neg_out,
            stride,
            neg_rows.len(),
            out_m,
            -1.0, // use |d| but will subtract result
        );
        // Subtract negative contribution
        for i in 0..out_size {
            out[i] -= neg_out[i];
        }
    }
}

/// Inner function for tall C-order sandwich with sign parameter.
fn dense_sandwich_tall_c_inner(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    stride: usize,
    in_n: usize,
    out_m: usize,
    sign: f64, // 1.0 for positive d, -1.0 for negative d (use |d|)
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

                // Build x_sub scaled by sqrt(|d|)
                for (k_local, rkk) in (rk..rkmax).enumerate() {
                    let kk = rows[rkk] as usize;
                    let d_sqrt = if sign > 0.0 {
                        d[kk].sqrt()
                    } else {
                        (-d[kk]).sqrt()
                    };
                    for (j_local, &col_j) in cols_usize.iter().enumerate() {
                        tl.x_sub[k_local * out_m + j_local] = d_sqrt * x[kk * stride + col_j];
                    }
                }

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

/// Optimized dense sandwich for tall matrices (F-order) with negative weights.
fn dense_sandwich_tall_signed_f(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    stride: usize,
    out_m: usize,
) {
    let mut pos_rows: Vec<i32> = Vec::new();
    let mut neg_rows: Vec<i32> = Vec::new();
    for &r in rows {
        if d[r as usize] >= 0.0 {
            pos_rows.push(r);
        } else {
            neg_rows.push(r);
        }
    }

    let out_size = out_m * out_m;

    if !pos_rows.is_empty() {
        dense_sandwich_tall_f_inner(x, d, &pos_rows, cols, out, stride, pos_rows.len(), out_m, 1.0);
    }

    if !neg_rows.is_empty() {
        let mut neg_out = vec![0.0f64; out_size];
        dense_sandwich_tall_f_inner(
            x,
            d,
            &neg_rows,
            cols,
            &mut neg_out,
            stride,
            neg_rows.len(),
            out_m,
            -1.0,
        );
        for i in 0..out_size {
            out[i] -= neg_out[i];
        }
    }
}

/// Inner function for tall F-order sandwich with sign parameter.
fn dense_sandwich_tall_f_inner(
    x: &[f64],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    stride: usize,
    in_n: usize,
    out_m: usize,
    sign: f64,
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

                // F-order: X[row, col] = x[col * stride + row]
                for (k_local, rkk) in (rk..rkmax).enumerate() {
                    let kk = rows[rkk] as usize;
                    let d_sqrt = if sign > 0.0 {
                        d[kk].sqrt()
                    } else {
                        (-d[kk]).sqrt()
                    };
                    for (j_local, &col_j) in cols_usize.iter().enumerate() {
                        tl.x_sub[k_local * out_m + j_local] = d_sqrt * x[col_j * stride + kk];
                    }
                }

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

/// Applies standardization corrections to a sandwich product in-place.
///
/// For a standardized matrix `S[i,j] = mult[j] * X[i,j] + shift[j]`, the sandwich
/// product `S.T @ diag(d) @ S` can be computed as:
///
/// ```text
/// result[i,j] = base[i,j] * mult[i] * mult[j]
///             + d_mat[i] * shift[j]
///             + shift[i] * d_mat[j]
///             + shift[i] * shift[j] * sum_d
/// ```
///
/// where:
/// - `base` is `X.T @ diag(d) @ X` (the base sandwich)
/// - `d_mat` is `mult * (X.T @ d)` (transpose matvec scaled by mult)
/// - `sum_d` is the sum of weights
///
/// This function computes the full result efficiently by combining all terms
/// in a single pass, avoiding multiple intermediate array allocations.
///
/// # Arguments
///
/// * `base` - Base sandwich product X.T @ diag(d) @ X (modified in-place)
/// * `d_mat` - mult * X.T @ d (already scaled by mult)
/// * `shift` - Shift values per column
/// * `mult` - Optional multiplier values per column (None = all 1s)
/// * `sum_d` - Sum of weights (sum of d)
///
/// # Note
///
/// The `base` array is modified in-place. The result is stored in `base`.
#[pyfunction]
#[pyo3(signature = (base, d_mat, shift, mult, sum_d))]
pub fn standardized_sandwich_correction<'py>(
    _py: Python<'py>,
    mut base: numpy::PyReadwriteArray2<f64>,
    d_mat: PyReadonlyArray1<f64>,
    shift: PyReadonlyArray1<f64>,
    mult: Option<PyReadonlyArray1<f64>>,
    sum_d: f64,
) {
    let mut base_arr = base.as_array_mut();
    let d_mat_slice = d_mat.as_slice().unwrap();
    let shift_slice = shift.as_slice().unwrap();
    let n = d_mat_slice.len();

    if n == 0 {
        return;
    }

    match mult {
        Some(mult_arr) => {
            let mult_slice = mult_arr.as_slice().unwrap();
            // Full correction with mult
            // result[i,j] = base[i,j] * mult[i] * mult[j]
            //             + d_mat[i] * shift[j]
            //             + shift[i] * d_mat[j]
            //             + shift[i] * shift[j] * sum_d
            for i in 0..n {
                let mult_i = mult_slice[i];
                let shift_i = shift_slice[i];
                let d_mat_i = d_mat_slice[i];

                for j in 0..n {
                    let mult_j = mult_slice[j];
                    let shift_j = shift_slice[j];
                    let d_mat_j = d_mat_slice[j];

                    base_arr[[i, j]] = base_arr[[i, j]] * mult_i * mult_j
                        + d_mat_i * shift_j
                        + shift_i * d_mat_j
                        + shift_i * shift_j * sum_d;
                }
            }
        }
        None => {
            // No mult, just apply shift corrections
            // result[i,j] = base[i,j]
            //             + d_mat[i] * shift[j]
            //             + shift[i] * d_mat[j]
            //             + shift[i] * shift[j] * sum_d
            for i in 0..n {
                let shift_i = shift_slice[i];
                let d_mat_i = d_mat_slice[i];

                for j in 0..n {
                    let shift_j = shift_slice[j];
                    let d_mat_j = d_mat_slice[j];

                    base_arr[[i, j]] += d_mat_i * shift_j
                        + shift_i * d_mat_j
                        + shift_i * shift_j * sum_d;
                }
            }
        }
    }
}
