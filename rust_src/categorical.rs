// Complete categorical matrix operations for tabmat

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Categorical sandwich (already implemented)
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

/// Matrix-vector multiplication for categorical matrices (simple case)
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

/// Matrix-vector multiplication for categorical matrices (drop_first case)
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

/// Transpose matrix-vector multiplication (simple case)
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

/// Transpose matrix-vector multiplication (drop_first case)
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

/// Sandwich for categorical (simple case)
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

/// Sandwich for categorical (drop_first case)
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

/// Multiply categorical matrix by vector d
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

/// Subset categorical matrix into CSR format
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

/// Get column inclusion mask
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
