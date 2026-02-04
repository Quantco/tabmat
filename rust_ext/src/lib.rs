//! Rust backend for tabmat matrix operations.
//!
//! This module provides idiomatic Rust implementations of categorical and sparse
//! matrix operations that are also available in the C++ backend. The implementations
//! use simple sequential iteration for clarity and correctness.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

mod categorical;
mod dense;
mod sparse;

/// Rust backend module for tabmat operations.
#[pymodule]
fn tabmat_rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Categorical operations
    m.add_function(wrap_pyfunction!(transpose_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(matvec, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_categorical, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_cat_cat, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_cat_dense, m)?)?;

    // Sparse operations
    m.add_function(wrap_pyfunction!(csr_matvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(csr_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(csc_rmatvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(csc_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(csr_dense_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_square_dot_weights, m)?)?;

    // Dense operations
    m.add_function(wrap_pyfunction!(dense_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(dense_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense_transpose_square_dot_weights, m)?)?;
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

    let n_j_cols = j_cols_slice.len();

    // Handle edge case: empty j_cols produces empty result with correct shape
    if n_j_cols == 0 || i_ncol == 0 {
        let array = PyArray2::zeros(py, [i_ncol, n_j_cols], false);
        return Ok(array.unbind());
    }

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

    let array = PyArray2::from_vec2(
        py,
        &result
            .chunks(n_j_cols)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok(array.into())
}

// =============================================================================
// Sparse operations
// =============================================================================

/// CSR matrix-vector multiplication: out += X @ v (unrestricted)
#[pyfunction]
fn csr_matvec_unrestricted<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    v: PyReadonlyArray1<'py, f64>,
    out: &Bound<'py, PyArray1<f64>>,
) -> PyResult<()> {
    let data = data.as_slice()?;
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let v = v.as_slice()?;

    {
        let out_slice = unsafe { out.as_slice_mut()? };
        sparse::csr_matvec_unrestricted(data, indices, indptr, v, out_slice);
    }

    Ok(())
}

/// CSR matrix-vector multiplication with row/column restrictions
#[pyfunction]
fn csr_matvec<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    v: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
    n_cols_total: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let v = v.as_slice()?;
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;

    let result = sparse::csr_matvec(data, indices, indptr, v, rows, cols, n_cols_total);

    Ok(PyArray1::from_vec(py, result).into())
}

/// CSC transpose matrix-vector multiplication: out += XT.T @ v (unrestricted)
#[pyfunction]
fn csc_rmatvec_unrestricted<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    v: PyReadonlyArray1<'py, f64>,
    out: &Bound<'py, PyArray1<f64>>,
) -> PyResult<()> {
    let data = data.as_slice()?;
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let v = v.as_slice()?;

    {
        let out_slice = unsafe { out.as_slice_mut()? };
        sparse::csc_rmatvec_unrestricted(data, indices, indptr, v, out_slice);
    }

    Ok(())
}

/// CSC transpose matrix-vector multiplication with restrictions
#[pyfunction]
fn csc_rmatvec<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    v: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
    n_rows_total: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let v = v.as_slice()?;
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;

    let result = sparse::csc_rmatvec(data, indices, indptr, v, rows, cols, n_rows_total);

    Ok(PyArray1::from_vec(py, result).into())
}

/// Sparse sandwich product: AT @ diag(d) @ A
#[pyfunction]
fn sparse_sandwich<'py>(
    py: Python<'py>,
    a_data: PyReadonlyArray1<'py, f64>,
    a_indices: PyReadonlyArray1<'py, i32>,
    a_indptr: PyReadonlyArray1<'py, i32>,
    at_data: PyReadonlyArray1<'py, f64>,
    at_indices: PyReadonlyArray1<'py, i32>,
    at_indptr: PyReadonlyArray1<'py, i32>,
    d: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
    n_rows_total: usize,
    n_cols_total: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let a_data = a_data.as_slice()?;
    let a_indices = a_indices.as_slice()?;
    let a_indptr = a_indptr.as_slice()?;
    let at_data = at_data.as_slice()?;
    let at_indices = at_indices.as_slice()?;
    let at_indptr = at_indptr.as_slice()?;
    let d = d.as_slice()?;
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;

    let m = cols.len();

    // Handle edge case: empty cols produces empty result with correct shape
    if m == 0 {
        let array = PyArray2::zeros(py, [0, 0], false);
        return Ok(array.unbind());
    }

    let result = sparse::sparse_sandwich(
        a_data,
        a_indices,
        a_indptr,
        at_data,
        at_indices,
        at_indptr,
        d,
        rows,
        cols,
        n_rows_total,
        n_cols_total,
    );

    let array = PyArray2::from_vec2(
        py,
        &result
            .chunks(m)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok(array.into())
}

/// CSR-dense sandwich: A.T @ diag(d) @ B
#[pyfunction]
fn csr_dense_sandwich<'py>(
    py: Python<'py>,
    a_data: PyReadonlyArray1<'py, f64>,
    a_indices: PyReadonlyArray1<'py, i32>,
    a_indptr: PyReadonlyArray1<'py, i32>,
    b: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    a_cols: PyReadonlyArray1<'py, i32>,
    b_cols: PyReadonlyArray1<'py, i32>,
    a_ncol: usize,
    is_c_contiguous: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let a_data = a_data.as_slice()?;
    let a_indices = a_indices.as_slice()?;
    let a_indptr = a_indptr.as_slice()?;
    let d = d.as_slice()?;
    let rows = rows.as_slice()?;
    let a_cols = a_cols.as_slice()?;
    let b_cols = b_cols.as_slice()?;
    let b_shape = (b.shape()[0], b.shape()[1]);
    let b_slice = b.as_slice()?;

    let n_a_cols = a_cols.len();
    let n_b_cols = b_cols.len();

    // Handle edge cases: empty dimensions produce empty result with correct shape
    if n_a_cols == 0 || n_b_cols == 0 {
        // Use zeros to create array with correct shape
        let array = PyArray2::zeros(py, [n_a_cols, n_b_cols], false);
        return Ok(array.unbind());
    }

    let result = sparse::csr_dense_sandwich(
        a_data,
        a_indices,
        a_indptr,
        b_slice,
        b_shape,
        d,
        rows,
        a_cols,
        b_cols,
        a_ncol,
        is_c_contiguous,
    );

    let array = PyArray2::from_vec2(
        py,
        &result
            .chunks(n_b_cols)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok(array.into())
}

/// Compute weighted squared column norms for CSC matrix
#[pyfunction]
fn transpose_square_dot_weights<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let weights = weights.as_slice()?;

    let result = sparse::transpose_square_dot_weights(data, indices, indptr, weights);

    Ok(PyArray1::from_vec(py, result).into())
}

// =============================================================================
// Dense operations
// =============================================================================

/// Dense sandwich product: X.T @ diag(d) @ X
#[pyfunction]
fn dense_sandwich<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_slice = x.as_slice()?;
    let x_shape = (x.shape()[0], x.shape()[1]);
    let d = d.as_slice()?;
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;
    let is_c_contiguous = x.is_c_contiguous();

    let out_m = cols.len();

    if out_m == 0 {
        let array = PyArray2::zeros(py, [0, 0], false);
        return Ok(array.unbind());
    }

    let result = dense::dense_sandwich(x_slice, x_shape, d, rows, cols, is_c_contiguous);

    let array = PyArray2::from_vec2(
        py,
        &result
            .chunks(out_m)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok(array.into())
}

/// Dense transpose matrix-vector multiplication: X.T @ v
#[pyfunction]
fn dense_rmatvec<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x.as_slice()?;
    let x_shape = (x.shape()[0], x.shape()[1]);
    let v = v.as_slice()?;
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;
    let is_c_contiguous = x.is_c_contiguous();

    let result = dense::dense_rmatvec(x_slice, x_shape, v, rows, cols, is_c_contiguous);

    Ok(PyArray1::from_vec(py, result).into())
}

/// Dense matrix-vector multiplication: X @ v
#[pyfunction]
fn dense_matvec<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x.as_slice()?;
    let x_shape = (x.shape()[0], x.shape()[1]);
    let v = v.as_slice()?;
    let rows = rows.as_slice()?;
    let cols = cols.as_slice()?;
    let is_c_contiguous = x.is_c_contiguous();

    let result = dense::dense_matvec(x_slice, x_shape, v, rows, cols, is_c_contiguous);

    Ok(PyArray1::from_vec(py, result).into())
}

/// Compute weighted squared column norms with shift for dense matrix
#[pyfunction]
fn dense_transpose_square_dot_weights<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    shift: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x.as_slice()?;
    let x_shape = (x.shape()[0], x.shape()[1]);
    let weights = weights.as_slice()?;
    let shift = shift.as_slice()?;
    let is_c_contiguous = x.is_c_contiguous();

    let result = dense::dense_transpose_square_dot_weights(x_slice, x_shape, weights, shift, is_c_contiguous);

    Ok(PyArray1::from_vec(py, result).into())
}
