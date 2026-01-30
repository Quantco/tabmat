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

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Chunk size for parallel reduction operations.
const CHUNK_SIZE: usize = 4096;

// =============================================================================
// Sandwich operations
// =============================================================================

/// Computes categorical sandwich product: diagonal of `X.T @ diag(d) @ X`.
///
/// For a one-hot categorical matrix X, the sandwich is diagonal.
/// Returns: `out[j] = sum_{i where indices[i]==j} d[i]`
#[pyfunction]
#[pyo3(signature = (indices, d, rows, n_cols))]
pub fn categorical_sandwich<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    n_cols: usize,
) -> Bound<'py, PyArray1<f64>> {
    sandwich_categorical_complex(py, indices, d, rows, n_cols, false)
}

/// Alias for categorical_sandwich (for API compatibility).
#[pyfunction]
#[pyo3(signature = (indices, d, rows, n_cols))]
pub fn sandwich_categorical_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    n_cols: usize,
) -> Bound<'py, PyArray1<f64>> {
    sandwich_categorical_complex(py, indices, d, rows, n_cols, false)
}

/// Computes categorical sandwich product with drop_first option.
///
/// When `drop_first=true`, rows with category 0 are excluded.
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
    let n_keep_rows = rows_slice.len();

    let n_chunks = (n_keep_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;

    let res = (0..n_chunks)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n_cols],
            |mut acc, chunk_idx| {
                let start = chunk_idx * CHUNK_SIZE;
                let end = (start + CHUNK_SIZE).min(n_keep_rows);
                for row_idx_idx in start..end {
                    let k = rows_slice[row_idx_idx] as usize;
                    let col_idx = if drop_first {
                        indices_slice[k] - 1
                    } else {
                        indices_slice[k]
                    };
                    if col_idx >= 0 {
                        acc[col_idx as usize] += d_slice[k];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n_cols],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += bi;
                }
                a
            },
        );

    PyArray1::from_vec_bound(py, res)
}

// =============================================================================
// Matrix-vector products (X @ v)
// =============================================================================

/// Computes categorical matrix-vector product: `X @ v` (simple case).
///
/// For row i, returns `v[indices[i]]`.
#[pyfunction]
#[pyo3(signature = (indices, other, nrows))]
pub fn matvec_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    nrows: usize,
) -> Bound<'py, PyArray1<f64>> {
    matvec_complex(py, indices, other, nrows, false)
}

/// Computes categorical matrix-vector product with drop_first option.
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

/// Computes categorical matrix-vector product with column restriction.
#[pyfunction]
#[pyo3(signature = (indices, other, col_included, nrows, drop_first))]
pub fn matvec_restricted<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    col_included: PyReadonlyArray1<i32>,
    nrows: usize,
    drop_first: bool,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let other_slice = other.as_slice().unwrap();
    let col_included_slice = col_included.as_slice().unwrap();

    let mut out = vec![0.0; nrows];

    out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let col_idx = if drop_first {
            indices_slice[i] - 1
        } else {
            indices_slice[i]
        };
        if col_idx >= 0 {
            let col = col_idx as usize;
            if col < col_included_slice.len() && col_included_slice[col] != 0 {
                *out_val = other_slice[col];
            }
        }
    });

    PyArray1::from_vec_bound(py, out)
}

// =============================================================================
// Transpose matrix-vector products (X.T @ v)
// =============================================================================

/// Computes categorical transpose-vector product: `X.T @ v`.
///
/// Sums elements of v by category: `out[j] = sum_{i where indices[i]==j} v[i]`
#[pyfunction]
#[pyo3(signature = (indices, other, n_cols))]
pub fn transpose_matvec_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    n_cols: usize,
) -> Bound<'py, PyArray1<f64>> {
    transpose_matvec_complex(py, indices, other, n_cols, false)
}

/// Computes categorical transpose-vector product with row restriction.
#[pyfunction]
#[pyo3(signature = (indices, other, rows, n_cols))]
pub fn transpose_matvec_fast_rows<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    n_cols: usize,
) -> Bound<'py, PyArray1<f64>> {
    transpose_matvec_complex_rows(py, indices, other, rows, n_cols, false)
}

/// Computes categorical transpose-vector product with drop_first option.
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
    let n_rows = other_slice.len();

    let n_chunks = (n_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;

    let out = (0..n_chunks)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n_cols],
            |mut acc, chunk_idx| {
                let start = chunk_idx * CHUNK_SIZE;
                let end = (start + CHUNK_SIZE).min(n_rows);
                for i in start..end {
                    let col_idx = if drop_first {
                        indices_slice[i] - 1
                    } else {
                        indices_slice[i]
                    };
                    if col_idx >= 0 {
                        acc[col_idx as usize] += other_slice[i];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n_cols],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += bi;
                }
                a
            },
        );

    PyArray1::from_vec_bound(py, out)
}

/// Computes categorical transpose-vector product with row restriction and drop_first.
#[pyfunction]
#[pyo3(signature = (indices, other, rows, n_cols, drop_first))]
pub fn transpose_matvec_complex_rows<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    n_cols: usize,
    drop_first: bool,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let other_slice = other.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let n_keep_rows = rows_slice.len();

    let n_chunks = (n_keep_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;

    let out = (0..n_chunks)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n_cols],
            |mut acc, chunk_idx| {
                let start = chunk_idx * CHUNK_SIZE;
                let end = (start + CHUNK_SIZE).min(n_keep_rows);
                for row_idx_idx in start..end {
                    let i = rows_slice[row_idx_idx] as usize;
                    let col_idx = if drop_first {
                        indices_slice[i] - 1
                    } else {
                        indices_slice[i]
                    };
                    if col_idx >= 0 {
                        acc[col_idx as usize] += other_slice[i];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n_cols],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += bi;
                }
                a
            },
        );

    PyArray1::from_vec_bound(py, out)
}

/// Computes categorical transpose-vector product with row and column restrictions.
#[pyfunction]
#[pyo3(signature = (indices, other, rows, col_included, n_cols, drop_first))]
pub fn transpose_matvec_restricted<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    other: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    col_included: PyReadonlyArray1<i32>,
    n_cols: usize,
    drop_first: bool,
) -> Bound<'py, PyArray1<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let other_slice = other.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let col_included_slice = col_included.as_slice().unwrap();
    let n_keep_rows = rows_slice.len();
    let col_included_len = col_included_slice.len();

    let n_chunks = (n_keep_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;

    let out = (0..n_chunks)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n_cols],
            |mut acc, chunk_idx| {
                let start = chunk_idx * CHUNK_SIZE;
                let end = (start + CHUNK_SIZE).min(n_keep_rows);
                for row_idx_idx in start..end {
                    let i = rows_slice[row_idx_idx] as usize;
                    let col_idx = if drop_first {
                        indices_slice[i] - 1
                    } else {
                        indices_slice[i]
                    };
                    if col_idx >= 0 {
                        let col = col_idx as usize;
                        if col < col_included_len && col_included_slice[col] != 0 {
                            acc[col] += other_slice[i];
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n_cols],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += bi;
                }
                a
            },
        );

    PyArray1::from_vec_bound(py, out)
}

// =============================================================================
// Utility functions
// =============================================================================

/// Element-wise multiplication returning CSR format: `diag(d) @ X`.
#[pyfunction]
#[pyo3(signature = (indices, d, drop_first))]
pub fn multiply_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    drop_first: bool,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
) {
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

/// Converts categorical matrix to CSR format for subsetting.
#[pyfunction]
#[pyo3(signature = (indices, drop_first))]
pub fn subset_categorical_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    drop_first: bool,
) -> (usize, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>) {
    let indices_slice = indices.as_slice().unwrap();

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
