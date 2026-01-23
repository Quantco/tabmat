//! Sparse matrix operations for tabmat.
//!
//! This module provides high-performance implementations of sparse matrix operations
//! commonly used in generalized linear model (GLM) fitting. Operations support both
//! CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) formats.
//!
//! # Key Operations
//!
//! - [`sparse_sandwich`]: Computes `A.T @ diag(d) @ A` for CSC matrices
//! - [`csr_matvec`] / [`csr_matvec_unrestricted`]: CSR matrix-vector products
//! - [`csc_rmatvec`] / [`csc_rmatvec_unrestricted`]: CSC transpose-vector products
//! - [`csr_dense_sandwich`]: Mixed CSR-dense sandwich product
//! - [`transpose_square_dot_weights`]: Weighted column-wise squared sums
//!
//! # Sparse Format Notes
//!
//! - **CSR** (Compressed Sparse Row): Efficient for row access, used for `X @ v`
//! - **CSC** (Compressed Sparse Column): Efficient for column access, used for `X.T @ v`
//!
//! All operations support row/column subsetting through index arrays, which is
//! essential for handling regularization and feature selection in GLM fitting.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Computes the sparse sandwich product: `A.T @ diag(d) @ A`.
///
/// This is the core operation for computing the Hessian matrix with sparse features.
///
/// # Arguments
///
/// * `a_data`, `a_indices`, `a_indptr` - CSC format sparse matrix A
/// * `at_data`, `at_indices`, `at_indptr` - CSR format of A (transpose, for efficient row access)
/// * `d` - Diagonal weight vector
/// * `rows` - Row indices to include in the computation
/// * `cols` - Column indices to include in the computation
///
/// # Returns
///
/// A symmetric dense matrix of shape (len(cols), len(cols)) containing the sandwich product.
///
/// # Algorithm
///
/// Computes the lower triangle by iterating over columns of A, then scatter-multiplies
/// with rows of A.T. The result is symmetrized at the end.
#[pyfunction]
#[pyo3(signature = (a_data, a_indices, a_indptr, at_data, at_indices, at_indptr, d, rows, cols))]
pub fn sparse_sandwich<'py>(
    py: Python<'py>,
    a_data: PyReadonlyArray1<f64>,
    a_indices: PyReadonlyArray1<i32>,
    a_indptr: PyReadonlyArray1<i32>,
    at_data: PyReadonlyArray1<f64>,
    at_indices: PyReadonlyArray1<i32>,
    at_indptr: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let a_data_slice = a_data.as_slice().unwrap();
    let a_indices_slice = a_indices.as_slice().unwrap();
    let a_indptr_slice = a_indptr.as_slice().unwrap();
    let at_data_slice = at_data.as_slice().unwrap();
    let at_indices_slice = at_indices.as_slice().unwrap();
    let at_indptr_slice = at_indptr.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();

    let m = cols_slice.len();

    if m == 0 {
        let out_2d: Vec<Vec<f64>> = vec![];
        return PyArray2::from_vec2_bound(py, &out_2d).unwrap();
    }

    // Build row_included mask
    let mut row_included = vec![false; d_slice.len()];
    for &r in rows_slice {
        row_included[r as usize] = true;
    }

    // Build col_map as array (faster than HashMap)
    let max_col = *cols_slice.iter().max().unwrap_or(&0) as usize;
    let mut col_map = vec![-1i32; max_col + 1];
    for (ci, &c) in cols_slice.iter().enumerate() {
        col_map[c as usize] = ci as i32;
    }

    // Compute lower triangle in parallel - each output column j is independent
    let result: Vec<Vec<f64>> = (0..m)
        .into_par_iter()
        .map(|cj| {
            let mut row = vec![0.0f64; m];
            let j = cols_slice[cj] as usize;
            let j_start = a_indptr_slice[j] as usize;
            let j_end = a_indptr_slice[j + 1] as usize;

            for idx in j_start..j_end {
                let k = a_indices_slice[idx] as usize;
                if !row_included[k] {
                    continue;
                }

                let a_val = a_data_slice[idx] * d_slice[k];
                let k_start = at_indptr_slice[k] as usize;
                let k_end = at_indptr_slice[k + 1] as usize;

                for idx2 in k_start..k_end {
                    let i = at_indices_slice[idx2] as usize;
                    if i > j {
                        break;
                    }

                    if i <= max_col {
                        let ci = col_map[i];
                        if ci >= 0 {
                            let at_val = at_data_slice[idx2];
                            row[ci as usize] += at_val * a_val;
                        }
                    }
                }
            }
            row
        })
        .collect();

    // Build final output with symmetrization
    let mut out_2d: Vec<Vec<f64>> = vec![vec![0.0; m]; m];
    for cj in 0..m {
        for ci in 0..=cj {
            out_2d[cj][ci] = result[cj][ci];
            out_2d[ci][cj] = result[cj][ci];
        }
    }

    PyArray2::from_vec2_bound(py, &out_2d).unwrap()
}

/// Computes CSR matrix-vector product: `A @ v` (all rows and columns).
///
/// This is the unrestricted version that processes all rows of the matrix.
/// Use [`csr_matvec`] if you need to subset rows or columns.
///
/// # Arguments
///
/// * `data`, `indices`, `indptr` - CSR format sparse matrix A
/// * `v` - Input vector
/// * `nrows` - Number of rows in the matrix
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, v, nrows))]
pub fn csr_matvec_unrestricted<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i32>,
    indptr: PyReadonlyArray1<i32>,
    v: PyReadonlyArray1<f64>,
    nrows: usize,
) -> Bound<'py, PyArray1<f64>> {
    let data_slice = data.as_slice().unwrap();
    let indices_slice = indices.as_slice().unwrap();
    let indptr_slice = indptr.as_slice().unwrap();
    let v_slice = v.as_slice().unwrap();

    let mut out = vec![0.0; nrows];

    out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let start = indptr_slice[i] as usize;
        let end = indptr_slice[i + 1] as usize;
        let mut accum = 0.0;
        for idx in start..end {
            unsafe {
                let j = *indices_slice.get_unchecked(idx) as usize;
                accum += data_slice.get_unchecked(idx) * v_slice.get_unchecked(j);
            }
        }
        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// Computes CSR matrix-vector product: `A @ v` (restricted to specified rows/columns).
///
/// Uses a column inclusion mask for efficient filtering. The output contains
/// one element per row in the `rows` array.
///
/// # Arguments
///
/// * `data`, `indices`, `indptr` - CSR format sparse matrix A
/// * `v` - Input vector
/// * `rows` - Row indices to include
/// * `cols` - Column indices to include (columns not in this set are skipped)
/// * `ncols` - Total number of columns (used for mask sizing)
///
/// # Performance
///
/// - Fast path when all columns are included (skips mask entirely)
/// - Uses branchless inner loop to avoid pipeline stalls from mispredictions
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, v, rows, cols, ncols))]
pub fn csr_matvec<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i32>,
    indptr: PyReadonlyArray1<i32>,
    v: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
    ncols: usize,
) -> Bound<'py, PyArray1<f64>> {
    let data_slice = data.as_slice().unwrap();
    let indices_slice = indices.as_slice().unwrap();
    let indptr_slice = indptr.as_slice().unwrap();
    let v_slice = v.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();

    let n = rows_slice.len();
    let mut out = vec![0.0; n];

    // Fast path: if all columns are included, skip the mask entirely
    if cols_slice.len() == ncols {
        out.par_iter_mut().enumerate().for_each(|(ci, out_val)| {
            let i = rows_slice[ci] as usize;
            let start = indptr_slice[i] as usize;
            let end = indptr_slice[i + 1] as usize;
            let mut accum = 0.0;
            for idx in start..end {
                unsafe {
                    let j = *indices_slice.get_unchecked(idx) as usize;
                    accum += data_slice.get_unchecked(idx) * v_slice.get_unchecked(j);
                }
            }
            *out_val = accum;
        });
        return PyArray1::from_vec_bound(py, out);
    }

    // Build col_included mask using f64 for branchless multiplication
    let mut col_included = vec![0.0f64; ncols];
    for &col in cols_slice {
        col_included[col as usize] = 1.0;
    }

    // Branchless inner loop: multiply by mask value instead of branching
    out.par_iter_mut().enumerate().for_each(|(ci, out_val)| {
        let i = rows_slice[ci] as usize;
        let start = indptr_slice[i] as usize;
        let end = indptr_slice[i + 1] as usize;
        let mut accum = 0.0;
        for idx in start..end {
            unsafe {
                let j = *indices_slice.get_unchecked(idx) as usize;
                accum += *col_included.get_unchecked(j)
                    * data_slice.get_unchecked(idx)
                    * v_slice.get_unchecked(j);
            }
        }
        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// Computes CSC transpose-vector product: `A.T @ v` (all rows and columns).
///
/// This is the unrestricted version that processes all columns of the matrix.
/// Use [`csc_rmatvec`] if you need to subset rows or columns.
///
/// # Arguments
///
/// * `data`, `indices`, `indptr` - CSC format sparse matrix A
/// * `v` - Input vector
/// * `ncols` - Number of columns in the matrix
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, v, ncols))]
pub fn csc_rmatvec_unrestricted<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i32>,
    indptr: PyReadonlyArray1<i32>,
    v: PyReadonlyArray1<f64>,
    ncols: usize,
) -> Bound<'py, PyArray1<f64>> {
    let data_slice = data.as_slice().unwrap();
    let indices_slice = indices.as_slice().unwrap();
    let indptr_slice = indptr.as_slice().unwrap();
    let v_slice = v.as_slice().unwrap();

    let mut out = vec![0.0; ncols];

    out.par_iter_mut().enumerate().for_each(|(j, out_val)| {
        let start = indptr_slice[j] as usize;
        let end = indptr_slice[j + 1] as usize;
        let mut accum = 0.0;
        for idx in start..end {
            unsafe {
                let i = *indices_slice.get_unchecked(idx) as usize;
                accum += data_slice.get_unchecked(idx) * v_slice.get_unchecked(i);
            }
        }
        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// Computes CSC transpose-vector product: `A.T @ v` (restricted to specified rows/columns).
///
/// Uses a row inclusion mask for efficient filtering. The output contains
/// one element per column in the `cols` array.
///
/// # Arguments
///
/// * `data`, `indices`, `indptr` - CSC format sparse matrix A
/// * `v` - Input vector
/// * `rows` - Row indices to include (rows not in this set are skipped)
/// * `cols` - Column indices to include
/// * `nrows` - Total number of rows (used for mask sizing)
///
/// # Performance
///
/// - Fast path when all rows are included (skips mask entirely)
/// - Uses branchless inner loop to avoid pipeline stalls from mispredictions
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, v, rows, cols, nrows))]
pub fn csc_rmatvec<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i32>,
    indptr: PyReadonlyArray1<i32>,
    v: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
    nrows: usize,
) -> Bound<'py, PyArray1<f64>> {
    let data_slice = data.as_slice().unwrap();
    let indices_slice = indices.as_slice().unwrap();
    let indptr_slice = indptr.as_slice().unwrap();
    let v_slice = v.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();

    let m = cols_slice.len();
    let mut out = vec![0.0; m];

    // Fast path: if all rows are included, skip the mask entirely
    if rows_slice.len() == nrows {
        out.par_iter_mut().enumerate().for_each(|(cj, out_val)| {
            let j = cols_slice[cj] as usize;
            let start = indptr_slice[j] as usize;
            let end = indptr_slice[j + 1] as usize;
            let mut accum = 0.0;
            for idx in start..end {
                unsafe {
                    let i = *indices_slice.get_unchecked(idx) as usize;
                    accum += data_slice.get_unchecked(idx) * v_slice.get_unchecked(i);
                }
            }
            *out_val = accum;
        });
        return PyArray1::from_vec_bound(py, out);
    }

    // Build row_included mask using f64 for branchless multiplication
    let mut row_included = vec![0.0f64; nrows];
    for &row in rows_slice {
        row_included[row as usize] = 1.0;
    }

    // Branchless inner loop: multiply by mask value instead of branching
    out.par_iter_mut().enumerate().for_each(|(cj, out_val)| {
        let j = cols_slice[cj] as usize;
        let start = indptr_slice[j] as usize;
        let end = indptr_slice[j + 1] as usize;
        let mut accum = 0.0;
        for idx in start..end {
            unsafe {
                let i = *indices_slice.get_unchecked(idx) as usize;
                accum += *row_included.get_unchecked(i)
                    * data_slice.get_unchecked(idx)
                    * v_slice.get_unchecked(i);
            }
        }
        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// Computes weighted column-wise squared sums for sparse matrices.
///
/// For each column j, computes: `sum_i weights[i] * X[i,j]^2`
///
/// This operation is used for computing variance-like statistics in GLM fitting.
/// Unlike the dense version, this does not subtract a shift value since sparse
/// matrices typically have implicit zeros.
///
/// # Arguments
///
/// * `data`, `indices`, `indptr` - CSC format sparse matrix X
/// * `weights` - Row weights
///
/// # Returns
///
/// A 1D array of length ncols with the weighted squared sums.
#[pyfunction(name = "sparse_transpose_square_dot_weights")]
#[pyo3(signature = (data, indices, indptr, weights))]
pub fn transpose_square_dot_weights<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i32>,
    indptr: PyReadonlyArray1<i32>,
    weights: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let data_slice = data.as_slice().unwrap();
    let indices_slice = indices.as_slice().unwrap();
    let indptr_slice = indptr.as_slice().unwrap();
    let weights_slice = weights.as_slice().unwrap();

    let ncols = indptr_slice.len() - 1;
    let mut out = vec![0.0; ncols];

    out.par_iter_mut().enumerate().for_each(|(j, out_val)| {
        let start = indptr_slice[j] as usize;
        let end = indptr_slice[j + 1] as usize;
        let mut accum = 0.0;
        for idx in start..end {
            unsafe {
                let i = *indices_slice.get_unchecked(idx) as usize;
                let v = *data_slice.get_unchecked(idx);
                accum += weights_slice.get_unchecked(i) * v * v;
            }
        }
        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// Computes mixed CSR-dense sandwich product: `A.T @ diag(d) @ B`.
///
/// This operation is used when one component of a split matrix is sparse (A)
/// and another is dense (B). Common in GLM fitting with mixed feature types.
///
/// The output `out[i, j] = sum_k A[k, a_cols[i]] * d[k] * B[k, b_cols[j]]`
///
/// # Arguments
///
/// * `a_data`, `a_indices`, `a_indptr` - CSR format sparse matrix A
/// * `b` - Dense matrix B (row-major, C-contiguous)
/// * `d` - Diagonal weight vector
/// * `rows` - Row indices to include
/// * `a_cols` - Column indices for A (determines output rows)
/// * `b_cols` - Column indices for B (determines output columns)
///
/// # Returns
///
/// Dense matrix of shape (len(a_cols), len(b_cols)).
///
/// # Performance
///
/// Uses BLIS-style blocking with:
/// - Parallel k-blocks (128 rows per block) for row-wise parallelism
/// - J-blocks (128 columns) for cache locality on B
/// - Per-thread accumulation buffers to avoid synchronization
/// - SIMD vectorization (f64x4) for inner loops
#[pyfunction]
#[pyo3(signature = (a_data, a_indices, a_indptr, b, d, rows, a_cols, b_cols))]
pub fn csr_dense_sandwich<'py>(
    py: Python<'py>,
    a_data: PyReadonlyArray1<f64>,
    a_indices: PyReadonlyArray1<i32>,
    a_indptr: PyReadonlyArray1<i32>,
    b: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    a_cols: PyReadonlyArray1<i32>,
    b_cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    use wide::f64x4;

    let a_data_slice = a_data.as_slice().unwrap();
    let a_indices_slice = a_indices.as_slice().unwrap();
    let a_indptr_slice = a_indptr.as_slice().unwrap();
    let b_arr = b.as_array();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let a_cols_slice = a_cols.as_slice().unwrap();
    let b_cols_slice = b_cols.as_slice().unwrap();

    let n_a_cols = a_cols_slice.len();
    let n_b_cols = b_cols_slice.len();
    let n_rows = rows_slice.len();
    let b_ncols = b_arr.ncols();

    // Early return for empty inputs
    if n_rows == 0 || n_a_cols == 0 || n_b_cols == 0 {
        return PyArray2::zeros_bound(py, [n_a_cols, n_b_cols], false);
    }

    // Build column map: original A column index -> output row index
    let max_a_col = a_cols_slice.iter().map(|&c| c as usize).max().unwrap_or(0);
    let mut a_col_map: Vec<i32> = vec![-1; max_a_col + 1];
    for (ci, &col) in a_cols_slice.iter().enumerate() {
        unsafe { *a_col_map.get_unchecked_mut(col as usize) = ci as i32; }
    }

    // Pre-convert b_cols to usize for indexing
    let b_cols_usize: Vec<usize> = b_cols_slice.iter().map(|&c| c as usize).collect();

    // Blocking parameters (matching C++)
    const KBLOCK: usize = 128;
    const JBLOCK: usize = 128;

    let n_kblocks = (n_rows + KBLOCK - 1) / KBLOCK;
    let out_size = n_a_cols * n_b_cols;

    // Get B as contiguous slice - required for fast path
    let b_slice = b_arr.as_slice().expect("B must be contiguous");

    // Parallel fold over k-blocks with per-thread accumulation
    // Each thread gets its own out_local AND r_block buffer (no per-j-block allocation)
    let result = (0..n_kblocks)
        .into_par_iter()
        .fold(
            || {
                // Allocate both buffers once per thread, not per iteration
                (vec![0.0f64; out_size], vec![0.0f64; KBLOCK * JBLOCK])
            },
            |(mut out_local, mut r_block), kb| {
                let k_start = kb * KBLOCK;
                let k_end = (k_start + KBLOCK).min(n_rows);
                let k_size = k_end - k_start;

                // Process j-blocks for better cache locality
                for jb_start in (0..n_b_cols).step_by(JBLOCK) {
                    let jb_end = (jb_start + JBLOCK).min(n_b_cols);
                    let j_size = jb_end - jb_start;

                    // Pre-compute R[k_local, j_local] = d[row_k] * B[row_k, B_cols[j]]
                    // Reuse r_block buffer (just overwrite, no need to zero)
                    unsafe {
                        for k_local in 0..k_size {
                            let k_idx = k_start + k_local;
                            let row_k = *rows_slice.get_unchecked(k_idx) as usize;
                            let d_k = *d_slice.get_unchecked(row_k);
                            let b_row_start = row_k * b_ncols;

                            let r_row_start = k_local * j_size;
                            for j_local in 0..j_size {
                                let b_col = *b_cols_usize.get_unchecked(jb_start + j_local);
                                *r_block.get_unchecked_mut(r_row_start + j_local) =
                                    d_k * *b_slice.get_unchecked(b_row_start + b_col);
                            }
                        }
                    }

                    // Now iterate over rows in this k-block and scatter A values
                    unsafe {
                        for k_local in 0..k_size {
                            let k_idx = k_start + k_local;
                            let row_k = *rows_slice.get_unchecked(k_idx) as usize;
                            let a_start = *a_indptr_slice.get_unchecked(row_k) as usize;
                            let a_end = *a_indptr_slice.get_unchecked(row_k + 1) as usize;
                            let r_row_start = k_local * j_size;

                            // Iterate over sparse entries in A[row_k, :]
                            for a_idx in a_start..a_end {
                                let a_col = *a_indices_slice.get_unchecked(a_idx) as usize;

                                // Check if this column is in our output
                                if a_col > max_a_col {
                                    continue;
                                }
                                let ci = *a_col_map.get_unchecked(a_col);
                                if ci < 0 {
                                    continue;
                                }
                                let ci = ci as usize;

                                let a_val = *a_data_slice.get_unchecked(a_idx);
                                let out_row_start = ci * n_b_cols + jb_start;

                                // SIMD inner loop: process 4 elements at a time
                                let a_val_simd = f64x4::splat(a_val);
                                let mut j_local = 0usize;

                                // SIMD loop for 4 elements at a time
                                while j_local + 4 <= j_size {
                                    let r_idx = r_row_start + j_local;
                                    let out_idx = out_row_start + j_local;

                                    // Load r_block values
                                    let r_vals = f64x4::new([
                                        *r_block.get_unchecked(r_idx),
                                        *r_block.get_unchecked(r_idx + 1),
                                        *r_block.get_unchecked(r_idx + 2),
                                        *r_block.get_unchecked(r_idx + 3),
                                    ]);

                                    // Load current output values
                                    let out_vals = f64x4::new([
                                        *out_local.get_unchecked(out_idx),
                                        *out_local.get_unchecked(out_idx + 1),
                                        *out_local.get_unchecked(out_idx + 2),
                                        *out_local.get_unchecked(out_idx + 3),
                                    ]);

                                    // FMA: out += a_val * r
                                    let result = a_val_simd.mul_add(r_vals, out_vals);
                                    let result_arr = result.to_array();

                                    // Store back
                                    *out_local.get_unchecked_mut(out_idx) = result_arr[0];
                                    *out_local.get_unchecked_mut(out_idx + 1) = result_arr[1];
                                    *out_local.get_unchecked_mut(out_idx + 2) = result_arr[2];
                                    *out_local.get_unchecked_mut(out_idx + 3) = result_arr[3];

                                    j_local += 4;
                                }

                                // Scalar cleanup for remaining elements
                                while j_local < j_size {
                                    let r_idx = r_row_start + j_local;
                                    let out_idx = out_row_start + j_local;
                                    *out_local.get_unchecked_mut(out_idx) +=
                                        a_val * *r_block.get_unchecked(r_idx);
                                    j_local += 1;
                                }
                            }
                        }
                    }
                }

                (out_local, r_block)
            },
        )
        .map(|(out_local, _)| out_local)
        .reduce(
            || vec![0.0f64; out_size],
            |mut a, b| {
                // SIMD reduction
                let mut i = 0;
                while i + 4 <= out_size {
                    unsafe {
                        let a_vals = f64x4::new([
                            *a.get_unchecked(i),
                            *a.get_unchecked(i + 1),
                            *a.get_unchecked(i + 2),
                            *a.get_unchecked(i + 3),
                        ]);
                        let b_vals = f64x4::new([
                            *b.get_unchecked(i),
                            *b.get_unchecked(i + 1),
                            *b.get_unchecked(i + 2),
                            *b.get_unchecked(i + 3),
                        ]);
                        let sum = a_vals + b_vals;
                        let sum_arr = sum.to_array();
                        *a.get_unchecked_mut(i) = sum_arr[0];
                        *a.get_unchecked_mut(i + 1) = sum_arr[1];
                        *a.get_unchecked_mut(i + 2) = sum_arr[2];
                        *a.get_unchecked_mut(i + 3) = sum_arr[3];
                    }
                    i += 4;
                }
                // Scalar cleanup
                while i < out_size {
                    unsafe { *a.get_unchecked_mut(i) += *b.get_unchecked(i); }
                    i += 1;
                }
                a
            },
        );

    // Create output array directly without intermediate Vec<Vec>
    let out_arr = PyArray2::zeros_bound(py, [n_a_cols, n_b_cols], false);
    unsafe {
        let mut arr_view = out_arr.as_array_mut();
        let out_ptr = arr_view.as_mut_ptr();
        std::ptr::copy_nonoverlapping(result.as_ptr(), out_ptr, out_size);
    }
    out_arr
}
