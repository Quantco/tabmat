//! Categorical matrix operations for tabmat.
//!
//! This module provides operations on categorical matrices, which represent
//! one-hot encoded categorical variables. Instead of storing the full one-hot
//! matrix (mostly zeros), we store only the column index for each row.
//!
//! # Representation
//!
//! A categorical matrix with n rows and k categories is stored as:
//! - `indices`: array of length n where `indices[i]` is the category for row i
//!
//! This is equivalent to a sparse matrix where row i has a single 1 in column `indices[i]`.
//!
//! # drop_first Option
//!
//! When `drop_first=true`, the first category (index 0) is dropped to avoid
//! multicollinearity in regression models. Rows with category 0 become all-zeros.
//!
//! # Key Operations
//!
//! - [`categorical_sandwich`] / [`sandwich_categorical_fast`]: Diagonal sum per category
//! - [`matvec_fast`] / [`matvec_complex`]: Forward matrix-vector products
//! - [`transpose_matvec_fast`] / [`transpose_matvec_complex`]: Transpose matrix-vector products
//! - [`multiply_complex`]: Element-wise multiplication returning CSR format
//! - [`subset_categorical_complex`]: Subset to CSR format

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Computes the categorical sandwich product (diagonal).
///
/// For a one-hot categorical matrix X, the sandwich `X.T @ diag(d) @ X` is diagonal.
/// This function returns just the diagonal: `out[j] = sum_{i where indices[i]==j} d[i]`
///
/// This is equivalent to [`sandwich_categorical_fast`].
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `d` - Diagonal weight vector
/// * `rows` - Row indices to include in the computation
/// * `n_cols` - Number of categories (output length)
#[pyfunction]
#[pyo3(signature = (indices, d, rows, n_cols))]
pub fn categorical_sandwich<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    n_cols: usize,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();

    let mut res = vec![0.0; n_cols];

    for &k in rows_slice {
        let col_idx = indices_slice[k as usize] as usize;
        res[col_idx] += d_slice[k as usize];
    }

    PyArray1::from_vec_bound(py, res)
}

/// Computes categorical matrix-vector product: `X @ v` (simple case).
///
/// For row i, returns `v[indices[i]]` - just a lookup since each row has a single 1.
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `other` - Input vector v (length = number of categories)
/// * `nrows` - Number of rows in the output
#[pyfunction]
#[pyo3(signature = (indices, other, nrows))]
pub fn matvec_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    nrows: usize,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let other_slice = other.as_slice().unwrap();

    let mut out = vec![0.0; nrows];

    out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let col_idx = indices_slice[i] as usize;
        *out_val = other_slice[col_idx];
    });

    PyArray1::from_vec_bound(py, out)
}

/// Computes categorical matrix-vector product: `X @ v` (with drop_first option).
///
/// When `drop_first=true`, rows with category 0 produce output 0 (dropped category).
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `other` - Input vector v
/// * `nrows` - Number of rows in the output
/// * `drop_first` - If true, subtract 1 from indices and skip negative results
#[pyfunction]
#[pyo3(signature = (indices, other, nrows, drop_first))]
pub fn matvec_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    nrows: usize,
    drop_first: bool,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let other_slice = other.as_slice().unwrap();

    let mut out = vec![0.0; nrows];

    out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let col_idx = if drop_first {
            indices_slice[i] - 1
        } else {
            indices_slice[i]
        };
        
        if col_idx >= 0 {
            *out_val = other_slice[col_idx as usize];
        }
    });

    PyArray1::from_vec_bound(py, out)
}

/// Computes categorical transpose-vector product: `X.T @ v` (simple case).
///
/// Sums elements of v by category: `out[j] = sum_{i where indices[i]==j} v[i]`
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `other` - Input vector v
/// * `n_cols` - Number of categories (output length)
#[pyfunction]
#[pyo3(signature = (indices, other, n_cols))]
pub fn transpose_matvec_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    n_cols: usize,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let other_slice = other.as_slice().unwrap();

    let mut out = vec![0.0; n_cols];

    for (i, &other_val) in other_slice.iter().enumerate() {
        let col_idx = indices_slice[i] as usize;
        out[col_idx] += other_val;
    }

    PyArray1::from_vec_bound(py, out)
}

/// Computes categorical transpose-vector product: `X.T @ v` (with drop_first option).
///
/// When `drop_first=true`, rows with category 0 are skipped in the accumulation.
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `other` - Input vector v
/// * `n_cols` - Number of categories (output length)
/// * `drop_first` - If true, subtract 1 from indices and skip negative results
#[pyfunction]
#[pyo3(signature = (indices, other, n_cols, drop_first))]
pub fn transpose_matvec_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    n_cols: usize,
    drop_first: bool,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let other_slice = other.as_slice().unwrap();

    let mut out = vec![0.0; n_cols];

    for (i, &other_val) in other_slice.iter().enumerate() {
        let col_idx = if drop_first {
            indices_slice[i] - 1
        } else {
            indices_slice[i]
        };
        
        if col_idx >= 0 {
            out[col_idx as usize] += other_val;
        }
    }

    PyArray1::from_vec_bound(py, out)
}

/// Computes categorical sandwich product (simple case).
///
/// Returns the diagonal of `X.T @ diag(d) @ X`.
/// Equivalent to [`categorical_sandwich`].
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `d` - Diagonal weight vector
/// * `rows` - Row indices to include
/// * `n_cols` - Number of categories (output length)
#[pyfunction]
#[pyo3(signature = (indices, d, rows, n_cols))]
pub fn sandwich_categorical_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    n_cols: usize,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();

    let mut res = vec![0.0; n_cols];

    for &k in rows_slice {
        let col_idx = indices_slice[k as usize] as usize;
        res[col_idx] += d_slice[k as usize];
    }

    PyArray1::from_vec_bound(py, res)
}

/// Computes categorical sandwich product (with drop_first option).
///
/// Returns the diagonal of `X.T @ diag(d) @ X` where rows with category 0
/// are excluded when `drop_first=true`.
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `d` - Diagonal weight vector
/// * `rows` - Row indices to include
/// * `n_cols` - Number of categories (output length)
/// * `drop_first` - If true, exclude rows with category 0
#[pyfunction]
#[pyo3(signature = (indices, d, rows, n_cols, drop_first))]
pub fn sandwich_categorical_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    n_cols: usize,
    drop_first: bool,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();

    let mut res = vec![0.0; n_cols];

    for &k in rows_slice {
        let col_idx = if drop_first {
            indices_slice[k as usize] - 1
        } else {
            indices_slice[k as usize]
        };
        
        if col_idx >= 0 {
            res[col_idx as usize] += d_slice[k as usize];
        }
    }

    PyArray1::from_vec_bound(py, res)
}

/// Element-wise multiplication of categorical matrix by diagonal vector.
///
/// Computes `diag(d) @ X` and returns the result in CSR format.
/// Since each row of X has at most one non-zero, the result is sparse.
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `d` - Diagonal values to multiply by
/// * `drop_first` - If true, exclude rows with category 0
///
/// # Returns
///
/// Tuple of (data, indices, indptr) arrays representing the CSR result.
#[pyfunction]
#[pyo3(signature = (indices, d, drop_first))]
pub fn multiply_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    drop_first: bool,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>) {
    let indices_slice = indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let nrows = indices_slice.len();

    let mut data = Vec::new();
    let mut new_indices = Vec::new();
    let mut indptr = vec![0];

    for i in 0..nrows {
        let col_idx = if drop_first {
            indices_slice[i] - 1
        } else {
            indices_slice[i]
        };
        
        if col_idx >= 0 {
            data.push(d_slice[i]);
            new_indices.push(col_idx);
        }
        
        indptr.push(data.len() as i32);
    }

    (
        PyArray1::from_vec_bound(py, data),
        PyArray1::from_vec_bound(py, new_indices),
        PyArray1::from_vec_bound(py, indptr),
    )
}

/// Converts a categorical matrix to CSR format (for subsetting).
///
/// Creates an all-ones CSR matrix with the same sparsity pattern as the
/// categorical matrix. Used when subsetting operations need CSR format.
///
/// # Arguments
///
/// * `indices` - Category index for each row
/// * `drop_first` - If true, exclude rows with category 0
///
/// # Returns
///
/// Tuple of (nnz, indices, indptr) where nnz is the number of non-zeros.
/// The data array (all ones) is not returned since it can be reconstructed.
#[pyfunction]
#[pyo3(signature = (indices, drop_first))]
pub fn subset_categorical_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    drop_first: bool,
) -> (usize, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>) {
    let indices_slice = indices.as_slice().unwrap();
    let _nrows = indices_slice.len();

    let mut new_indices = Vec::new();
    let mut indptr = vec![0];

    for &idx in indices_slice {
        let col_idx = if drop_first { idx - 1 } else { idx };
        
        if col_idx >= 0 {
            new_indices.push(col_idx);
        }
        
        indptr.push(new_indices.len() as i32);
    }

    let nonzero_cnt = new_indices.len();

    (
        nonzero_cnt,
        PyArray1::from_vec_bound(py, new_indices),
        PyArray1::from_vec_bound(py, indptr),
    )
}

/// Creates a column inclusion mask from column indices.
///
/// Returns a binary mask where `mask[i] = 1` if column i is in the `cols` array.
///
/// # Arguments
///
/// * `cols` - Column indices that should be included
/// * `n_cols` - Total number of columns (mask length)
///
/// # Returns
///
/// A binary mask array of length n_cols.
#[pyfunction]
#[pyo3(signature = (cols, n_cols))]
pub fn get_col_included<'py>(
    py: Python<'py>,
    cols: PyReadonlyArray1<i32>,
    n_cols: usize,
) -> Bound<'py, PyArray1<i32>> {
    let cols_slice = cols.as_slice().unwrap();
    let mut col_included = vec![0; n_cols];

    for &col in cols_slice {
        col_included[col as usize] = 1;
    }

    PyArray1::from_vec_bound(py, col_included)
}
