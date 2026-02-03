//! Rust backend for tabmat categorical matrix operations.
//!
//! This module provides idiomatic Rust implementations of the categorical matrix
//! operations that are also available in the C++ backend. The implementations
//! use simple sequential iteration for clarity and correctness.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

mod categorical;

/// Rust backend module for tabmat categorical operations.
#[pymodule]
fn tabmat_rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transpose_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(matvec, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_categorical, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_cat_cat, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_cat_dense, m)?)?;
    Ok(())
}

// =============================================================================
// transpose_matvec
// =============================================================================

/// Transpose matrix-vector multiplication: out += X.T @ other
///
/// For categorical matrices, X[i, indices[i]] = 1 and all other elements are 0.
/// So (X.T @ other)[j] = sum over all rows i where indices[i] == j of other[i].
#[pyfunction]
fn transpose_matvec<'py>(
    _py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    other: PyReadonlyArray1<'py, f64>,
    out: &Bound<'py, PyArray1<f64>>,
    drop_first: bool,
) -> PyResult<()> {
    let indices = indices.as_slice()?;
    let other = other.as_slice()?;
    let out_len = out.len();

    let result = categorical::transpose_matvec(indices, other, out_len, drop_first);

    // Add results to output array
    {
        let out_slice = unsafe { out.as_slice_mut()? };
        for (o, r) in out_slice.iter_mut().zip(result.iter()) {
            *o += *r;
        }
    }

    Ok(())
}

// =============================================================================
// matvec
// =============================================================================

/// Matrix-vector multiplication: out[i] += other[indices[i]]
///
/// For categorical matrices with one-hot encoding, this is just a lookup.
#[pyfunction]
fn matvec<'py>(
    _py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    other: PyReadonlyArray1<'py, f64>,
    out: &Bound<'py, PyArray1<f64>>,
    drop_first: bool,
) -> PyResult<()> {
    let indices = indices.as_slice()?;
    let other = other.as_slice()?;

    {
        let out_slice = unsafe { out.as_slice_mut()? };
        categorical::matvec(indices, other, out_slice, drop_first);
    }

    Ok(())
}

// =============================================================================
// sandwich_categorical
// =============================================================================

/// Sandwich product for categorical matrix: X.T @ diag(d) @ X
///
/// For categorical matrices, the result is always diagonal.
/// Result[j] = sum of d[i] for all rows i where indices[i] == j.
#[pyfunction]
fn sandwich_categorical<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    d: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    n_cols: usize,
    drop_first: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let indices = indices.as_slice()?;
    let d = d.as_slice()?;
    let rows = rows.as_slice()?;

    let result = categorical::sandwich_diagonal(indices, d, rows, n_cols, drop_first);

    Ok(PyArray1::from_vec(py, result).into())
}

// =============================================================================
// sandwich_cat_cat
// =============================================================================

/// Cat-cat sandwich: X1.T @ diag(d) @ X2
///
/// Result[i, j] = sum of d[k] for all rows k where i_indices[k] == i and j_indices[k] == j.
#[pyfunction]
fn sandwich_cat_cat<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<'py, i32>,
    j_indices: PyReadonlyArray1<'py, i32>,
    d: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    i_ncol: usize,
    j_ncol: usize,
    i_drop_first: bool,
    j_drop_first: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let i_indices = i_indices.as_slice()?;
    let j_indices = j_indices.as_slice()?;
    let d = d.as_slice()?;
    let rows = rows.as_slice()?;

    let result = categorical::sandwich_cat_cat(
        i_indices,
        j_indices,
        d,
        rows,
        i_ncol,
        j_ncol,
        i_drop_first,
        j_drop_first,
    );

    // Convert flat Vec to 2D array
    let array = PyArray2::from_vec2(
        py,
        &result
            .chunks(j_ncol)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok(array.into())
}

// =============================================================================
// sandwich_cat_dense
// =============================================================================

/// Cat-dense sandwich: X_cat.T @ diag(d) @ X_dense
///
/// Result[i, j] = sum over k of (d[k] * X_dense[k, j]) where i_indices[k] == i.
#[pyfunction]
fn sandwich_cat_dense<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<'py, i32>,
    d: PyReadonlyArray1<'py, f64>,
    mat_j: PyReadonlyArray2<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    j_cols: PyReadonlyArray1<'py, i32>,
    i_ncol: usize,
    is_c_contiguous: bool,
    drop_first: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let i_indices = i_indices.as_slice()?;
    let d = d.as_slice()?;
    let rows = rows.as_slice()?;
    let j_cols_slice = j_cols.as_slice()?;
    let shape = mat_j.shape();
    let mat_j_shape = (shape[0], shape[1]);

    let mat_j_slice = mat_j.as_slice()?;

    let result = categorical::sandwich_cat_dense(
        i_indices,
        d,
        mat_j_slice,
        mat_j_shape,
        rows,
        j_cols_slice,
        i_ncol,
        is_c_contiguous,
        drop_first,
    );

    let n_j_cols = j_cols_slice.len();
    let array = PyArray2::from_vec2(
        py,
        &result
            .chunks(n_j_cols)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok(array.into())
}
