//! Split matrix operations for tabmat.
//!
//! This module provides operations for "split" matrices, which are matrices
//! composed of multiple sub-matrices of potentially different types (dense,
//! sparse, categorical). Common in regression models with mixed feature types.
//!
//! # Key Operations
//!
//! - [`sandwich_cat_cat`]: Sandwich product between two categorical matrices
//! - [`sandwich_cat_dense`]: Sandwich product between categorical and dense matrices
//! - [`split_col_subsets`]: Maps global column indices to local sub-matrix indices
//! - [`is_sorted`]: Utility to check if an array is sorted
//!
//! # Performance Notes
//!
//! The sandwich operations use per-thread accumulation buffers to avoid
//! atomic operations, following the same strategy as the C++ OpenMP implementation.
//! This provides near-linear scaling with thread count.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rayon::prelude::*;

/// Checks if an array is sorted in non-decreasing order.
///
/// Used to verify that column indices are sorted, which is required for
/// efficient column subsetting operations.
///
/// # Arguments
///
/// * `a` - Array to check (converted to i64)
///
/// # Returns
///
/// `true` if the array is sorted, `false` otherwise. Empty arrays return `true`.
#[pyfunction]
#[pyo3(signature = (a,))]
pub fn is_sorted(a: &Bound<'_, PyAny>) -> PyResult<bool> {
    // Convert to i64 array - works with various integer types
    let arr: PyReadonlyArray1<i64> = a.extract()?;
    let a_slice = arr.as_slice()?;
    
    // Empty arrays are considered sorted
    if a_slice.len() <= 1 {
        return Ok(true);
    }
    
    for i in 0..(a_slice.len() - 1) {
        if a_slice[i + 1] < a_slice[i] {
            return Ok(false);
        }
    }
    
    Ok(true)
}

/// Computes sandwich product between two categorical matrices.
///
/// Computes: `Cat_i.T @ diag(d) @ Cat_j`
///
/// This is a sparse-sparse product where both matrices are categorical
/// (one-hot encoded). The result `out[i, j]` is the sum of `d[k]` for all
/// rows k where `Cat_i[k, i] = 1` and `Cat_j[k, j] = 1`.
///
/// # Arguments
///
/// * `i_indices` - Category indices for the left matrix (length = n_total_rows)
/// * `j_indices` - Category indices for the right matrix (length = n_total_rows)
/// * `i_ncol` - Number of categories in left matrix
/// * `j_ncol` - Number of categories in right matrix
/// * `d` - Diagonal weight vector
/// * `rows` - Row indices to include in the computation
/// * `dtype` - Data type (unused, for Python compatibility)
/// * `i_drop_first` - If true, exclude category 0 from left matrix
/// * `j_drop_first` - If true, exclude category 0 from right matrix
/// * `i_has_missings` - If true, left matrix may have missing values (-1 indices)
/// * `j_has_missings` - If true, right matrix may have missing values (-1 indices)
///
/// # Returns
///
/// Dense matrix of shape (i_ncol, j_ncol).
///
/// # Performance
///
/// Uses pre-allocated per-thread accumulation buffers with a final reduction,
/// avoiding atomic operations. This matches the C++ OpenMP approach and provides
/// near-linear scaling with thread count.
#[pyfunction]
#[pyo3(signature = (i_indices, j_indices, i_ncol, j_ncol, d, rows, _dtype, i_drop_first, j_drop_first, i_has_missings, j_has_missings))]
pub fn sandwich_cat_cat<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<i32>,
    j_indices: PyReadonlyArray1<i32>,
    i_ncol: usize,
    j_ncol: usize,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    _dtype: Bound<'py, PyAny>,
    i_drop_first: bool,
    j_drop_first: bool,
    i_has_missings: bool,
    j_has_missings: bool,
) -> Bound<'py, PyArray2<f64>> {
    let i_indices_slice = i_indices.as_slice().unwrap();
    let j_indices_slice = j_indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();

    let n_rows = rows_slice.len();
    let res_size = i_ncol * j_ncol;

    // Early return for empty case
    if n_rows == 0 || i_ncol == 0 || j_ncol == 0 {
        return PyArray2::zeros_bound(py, [i_ncol, j_ncol], false);
    }

    // Get number of threads for pre-allocation
    let num_threads = rayon::current_num_threads();

    // Pre-allocate all thread buffers in one contiguous allocation
    let mut all_res = vec![0.0f64; num_threads * res_size];

    // Determine if we need the complex path (with bounds checking)
    let needs_complex = i_drop_first || j_drop_first || i_has_missings || j_has_missings;

    // Parallel iteration with thread-local accumulation
    // Each thread gets a slice of all_res based on thread index
    {
        use rayon::iter::IndexedParallelIterator;

        // Process in chunks, one per thread
        let chunk_size = (n_rows + num_threads - 1) / num_threads;

        all_res
            .par_chunks_mut(res_size)
            .enumerate()
            .for_each(|(tid, res_local)| {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(n_rows);

                if needs_complex {
                    // Complex path with drop_first and/or missing value checks
                    for k_idx in start..end {
                        let k = unsafe { *rows_slice.get_unchecked(k_idx) } as usize;

                        let i_raw = unsafe { *i_indices_slice.get_unchecked(k) };
                        let j_raw = unsafe { *j_indices_slice.get_unchecked(k) };

                        let i_idx = if i_drop_first { i_raw - 1 } else { i_raw };
                        let j_idx = if j_drop_first { j_raw - 1 } else { j_raw };

                        // Skip if either index is negative (missing or dropped first category)
                        if i_idx >= 0 && j_idx >= 0 {
                            let out_idx = i_idx as usize * j_ncol + j_idx as usize;
                            unsafe {
                                *res_local.get_unchecked_mut(out_idx) +=
                                    *d_slice.get_unchecked(k);
                            }
                        }
                    }
                } else {
                    // Fast path - no drop_first or missing checks needed
                    for k_idx in start..end {
                        let k = unsafe { *rows_slice.get_unchecked(k_idx) } as usize;
                        let i = unsafe { *i_indices_slice.get_unchecked(k) } as usize;
                        let j = unsafe { *j_indices_slice.get_unchecked(k) } as usize;
                        let out_idx = i * j_ncol + j;
                        unsafe {
                            *res_local.get_unchecked_mut(out_idx) +=
                                *d_slice.get_unchecked(k);
                        }
                    }
                }
            });
    }

    // Final reduction: sum all thread results into first buffer
    // This is much faster than atomic operations for small result sizes
    let (result, rest) = all_res.split_at_mut(res_size);
    for thread_buf in rest.chunks(res_size) {
        for (r, &t) in result.iter_mut().zip(thread_buf.iter()) {
            *r += t;
        }
    }

    // Create output array directly from flat result
    let out = PyArray2::zeros_bound(py, [i_ncol, j_ncol], false);
    {
        let mut out_rw = unsafe { out.as_array_mut() };
        for i in 0..i_ncol {
            for j in 0..j_ncol {
                out_rw[[i, j]] = result[i * j_ncol + j];
            }
        }
    }
    out
}

/// Computes sandwich product between categorical and dense matrices.
///
/// Computes: `Cat.T @ diag(d) @ Dense`
///
/// This is a mixed sparse-dense product where the left matrix is categorical
/// (one-hot encoded) and the right matrix is dense.
///
/// # Arguments
///
/// * `i_indices` - Category indices for the categorical matrix
/// * `i_ncol` - Number of categories in categorical matrix
/// * `d` - Diagonal weight vector
/// * `mat_j` - Dense matrix (row-major)
/// * `rows` - Row indices to include
/// * `j_cols` - Column indices for the dense matrix
/// * `is_c_contiguous` - Whether mat_j is C-contiguous (unused, auto-detected)
/// * `has_missings` - If true, categorical matrix may have missing values (-1 indices)
/// * `drop_first` - If true, exclude category 0 from categorical matrix
///
/// # Returns
///
/// Dense matrix of shape (i_ncol, len(j_cols)).
///
/// # Performance
///
/// Uses parallel k-blocks (1024 rows per block) with per-thread accumulation.
/// Optimized for C-contiguous dense matrices.
#[pyfunction]
#[pyo3(signature = (i_indices, i_ncol, d, mat_j, rows, j_cols, _is_c_contiguous, has_missings=false, drop_first=false))]
pub fn sandwich_cat_dense<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<i32>,
    i_ncol: usize,
    d: PyReadonlyArray1<f64>,
    mat_j: PyReadonlyArray2<f64>,
    rows: PyReadonlyArray1<i32>,
    j_cols: PyReadonlyArray1<i32>,
    _is_c_contiguous: bool,
    has_missings: bool,
    drop_first: bool,
) -> Bound<'py, PyArray2<f64>> {
    let i_indices_slice = i_indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let mat_j_arr = mat_j.as_array();
    let rows_slice = rows.as_slice().unwrap();
    let j_cols_slice = j_cols.as_slice().unwrap();

    let n_rows = rows_slice.len();
    let nj_active_cols = j_cols_slice.len();
    let res_size = i_ncol * nj_active_cols;

    // Early return for empty case
    if n_rows == 0 || nj_active_cols == 0 || i_ncol == 0 {
        return PyArray2::zeros_bound(py, [i_ncol, nj_active_cols], false);
    }

    // Get mat_j dimensions and try to get contiguous slice
    let mat_j_ncols = mat_j_arr.ncols();
    let mat_j_slice_opt = mat_j_arr.as_slice();

    // Pre-convert j_cols to usize
    let j_cols_usize: Vec<usize> = j_cols_slice.iter().map(|&c| c as usize).collect();

    // Blocking for parallelism - process rows in chunks
    const KBLOCK: usize = 1024;
    let n_kblocks = (n_rows + KBLOCK - 1) / KBLOCK;

    // Determine if we need bounds checking
    let needs_bounds_check = drop_first || has_missings;

    // Parallel fold over k-blocks with per-thread accumulation
    let result = (0..n_kblocks)
        .into_par_iter()
        .fold(
            || vec![0.0f64; res_size],
            |mut res_local, kb| {
                let k_start = kb * KBLOCK;
                let k_end = (k_start + KBLOCK).min(n_rows);

                for k_idx in k_start..k_end {
                    let k = rows_slice[k_idx] as usize;

                    let i_idx = if drop_first {
                        i_indices_slice[k] - 1
                    } else {
                        i_indices_slice[k]
                    };

                    // Skip negative indices (dropped first category or missing values)
                    if !needs_bounds_check || i_idx >= 0 {
                        let i = i_idx as usize;
                        let d_k = d_slice[k];
                        let out_row_start = i * nj_active_cols;

                        if let Some(mat_j_slice) = mat_j_slice_opt {
                            // Fast path: mat_j is C-contiguous
                            let mat_j_row_start = k * mat_j_ncols;
                            for (j_local, &j) in j_cols_usize.iter().enumerate() {
                                res_local[out_row_start + j_local] +=
                                    d_k * mat_j_slice[mat_j_row_start + j];
                            }
                        } else {
                            // Slow path: use array indexing
                            for (j_local, &j) in j_cols_usize.iter().enumerate() {
                                res_local[out_row_start + j_local] +=
                                    d_k * mat_j_arr[[k, j]];
                            }
                        }
                    }
                }

                res_local
            },
        )
        .reduce(
            || vec![0.0f64; res_size],
            |mut a, b| {
                for i in 0..res_size {
                    a[i] += b[i];
                }
                a
            },
        );

    // Convert flat result to 2D array
    let out_2d: Vec<Vec<f64>> = (0..i_ncol)
        .map(|i| result[i * nj_active_cols..(i + 1) * nj_active_cols].to_vec())
        .collect();
    PyArray2::from_vec2_bound(py, &out_2d).unwrap()
}

/// Maps global column indices to local sub-matrix indices.
///
/// For a split matrix composed of multiple sub-matrices, this function takes
/// a list of global column indices and determines which sub-matrix each column
/// belongs to, along with its local index within that sub-matrix.
///
/// # Arguments
///
/// * `indices_list` - List of arrays, where `indices_list[i]` contains the global
///   column indices that belong to sub-matrix i (must be sorted)
/// * `cols` - Array of requested global column indices (must be sorted)
///
/// # Returns
///
/// Tuple of (subset_cols_indices, subset_cols, n_cols) where:
/// - `subset_cols_indices[i]` - Positions in the `cols` array for sub-matrix i
/// - `subset_cols[i]` - Corresponding local column indices within sub-matrix i
/// - `n_cols` - Total number of requested columns
///
/// # Example
///
/// If a split matrix has sub-matrices with columns [0, 1, 2] and [3, 4, 5],
/// and we request columns [1, 4], the result tells us that:
/// - Column at position 0 (global index 1) maps to sub-matrix 0, local index 1
/// - Column at position 1 (global index 4) maps to sub-matrix 1, local index 1
#[pyfunction]
#[pyo3(signature = (indices_list, cols))]
pub fn split_col_subsets<'py>(
    py: Python<'py>,
    indices_list: &Bound<'py, PyList>,
    cols: PyReadonlyArray1<i32>,
) -> PyResult<(Vec<Bound<'py, PyArray1<i32>>>, Vec<Bound<'py, PyArray1<i32>>>, usize)> {
    let cols_slice = cols.as_slice()?;
    let n_matrices = indices_list.len();
    
    // Convert indices_list to Vec of Vec<i32> for easier access
    let mut indices: Vec<Vec<i32>> = Vec::new();
    for i in 0..n_matrices {
        let item = indices_list.get_item(i)?;
        // Try different integer dtypes
        if let Ok(arr) = item.downcast::<PyArray1<i32>>() {
            let readonly = arr.readonly();
            indices.push(readonly.as_slice()?.to_vec());
        } else if let Ok(arr) = item.downcast::<PyArray1<i64>>() {
            let readonly = arr.readonly();
            let vec_i64 = readonly.as_slice()?.to_vec();
            indices.push(vec_i64.iter().map(|&x| x as i32).collect());
        } else if let Ok(arr) = item.downcast::<PyArray1<isize>>() {
            let readonly = arr.readonly();
            let vec_isize = readonly.as_slice()?.to_vec();
            indices.push(vec_isize.iter().map(|&x| x as i32).collect());
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Expected PyArray1 with integer dtype, got {:?}", item.get_type())
            ));
        }
    }
    
    // Initialize tracking variables
    let mut next_subset_idx: Vec<usize> = vec![0; n_matrices];
    let mut subset_cols_indices_vecs: Vec<Vec<i32>> = vec![Vec::new(); n_matrices];
    let mut subset_cols_vecs: Vec<Vec<i32>> = vec![Vec::new(); n_matrices];
    
    // For each requested column
    for (i, &col) in cols_slice.iter().enumerate() {
        // Check each sub-matrix
        for j in 0..n_matrices {
            // Advance next_subset_idx[j] until we find a column >= col
            while next_subset_idx[j] < indices[j].len() 
                && indices[j][next_subset_idx[j]] < col {
                next_subset_idx[j] += 1;
            }
            
            // If we found an exact match
            if next_subset_idx[j] < indices[j].len() 
                && indices[j][next_subset_idx[j]] == col {
                // Record the global index (position in cols array)
                subset_cols_indices_vecs[j].push(i as i32);
                // Record the local index (position in this sub-matrix)
                subset_cols_vecs[j].push(next_subset_idx[j] as i32);
                next_subset_idx[j] += 1;
                break;  // Column found, move to next requested column
            }
        }
    }
    
    // Convert Vec<Vec<i32>> to Vec<PyArray1<i32>>
    let mut subset_cols_indices: Vec<Bound<'py, PyArray1<i32>>> = Vec::new();
    let mut subset_cols: Vec<Bound<'py, PyArray1<i32>>> = Vec::new();
    for j in 0..n_matrices {
        subset_cols_indices.push(PyArray1::from_vec_bound(py, subset_cols_indices_vecs[j].clone()));
        subset_cols.push(PyArray1::from_vec_bound(py, subset_cols_vecs[j].clone()));
    }

    Ok((subset_cols_indices, subset_cols, cols_slice.len()))
}

