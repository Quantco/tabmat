// Complete sparse matrix operations for tabmat

use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;
use std::collections::HashSet;

/// Sparse sandwich product: AT @ diag(d) @ A
/// A is in CSC format, AT is the transpose (passed as CSR which is CSC transposed)
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
    let mut out = vec![vec![0.0; m]; m];

    let row_set: HashSet<i32> = rows_slice.iter().copied().collect();
    let col_map: std::collections::HashMap<i32, usize> = cols_slice
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    for (cj, &j) in cols_slice.iter().enumerate() {
        let j_start = a_indptr_slice[j as usize] as usize;
        let j_end = a_indptr_slice[(j + 1) as usize] as usize;

        for idx in j_start..j_end {
            let k = a_indices_slice[idx];
            if !row_set.contains(&k) {
                continue;
            }

            let a_val = a_data_slice[idx] * d_slice[k as usize];

            let k_start = at_indptr_slice[k as usize] as usize;
            let k_end = at_indptr_slice[(k + 1) as usize] as usize;

            for idx2 in k_start..k_end {
                let i = at_indices_slice[idx2];
                if i > j {
                    break;
                }

                if let Some(&ci) = col_map.get(&i) {
                    let at_val = at_data_slice[idx2];
                    out[cj][ci] += at_val * a_val;
                }
            }
        }
    }

    // Symmetrize
    for i in 0..m {
        for j in (i + 1)..m {
            out[i][j] = out[j][i];
        }
    }

    PyArray2::from_vec2_bound(py, &out).unwrap()
}

/// CSR matrix-vector product (unrestricted)
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
            let j = indices_slice[idx] as usize;
            accum += data_slice[idx] * v_slice[j];
        }

        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// CSR matrix-vector product (restricted to rows/cols)
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

    // Build col_included mask
    let mut col_included = vec![false; ncols];
    for &col in cols_slice {
        col_included[col as usize] = true;
    }

    out.par_iter_mut().enumerate().for_each(|(ci, out_val)| {
        let i = rows_slice[ci] as usize;
        let start = indptr_slice[i] as usize;
        let end = indptr_slice[i + 1] as usize;
        let mut accum = 0.0;

        for idx in start..end {
            let j = indices_slice[idx] as usize;
            if col_included[j] {
                accum += data_slice[idx] * v_slice[j];
            }
        }

        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// CSC transpose matrix-vector product (unrestricted)
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
            let i = indices_slice[idx] as usize;
            accum += data_slice[idx] * v_slice[i];
        }

        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// CSC transpose matrix-vector product (restricted to rows/cols)
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

    // Build row_included mask
    let mut row_included = vec![false; nrows];
    for &row in rows_slice {
        row_included[row as usize] = true;
    }

    out.par_iter_mut().enumerate().for_each(|(cj, out_val)| {
        let j = cols_slice[cj] as usize;
        let start = indptr_slice[j] as usize;
        let end = indptr_slice[j + 1] as usize;
        let mut accum = 0.0;

        for idx in start..end {
            let i = indices_slice[idx] as usize;
            if row_included[i] {
                accum += data_slice[idx] * v_slice[i];
            }
        }

        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// Transpose square dot weights for sparse matrices
/// Computes sum of weights[i] * X[i,j]^2 for each column j
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

    // Parallel iteration over columns
    out.par_iter_mut().enumerate().for_each(|(j, out_val)| {
        let start = indptr_slice[j] as usize;
        let end = indptr_slice[j + 1] as usize;
        let mut accum = 0.0;

        for idx in start..end {
            let i = indices_slice[idx] as usize;
            let v = data_slice[idx];
            accum += weights_slice[i] * v * v;
        }

        *out_val = accum;
    });

    PyArray1::from_vec_bound(py, out)
}

/// CSR-dense sandwich: (A.T @ diag(d)) @ B where A is CSR
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
    let a_data_slice = a_data.as_slice().unwrap();
    let a_indices_slice = a_indices.as_slice().unwrap();
    let a_indptr_slice = a_indptr.as_slice().unwrap();
    let b_arr = b.as_array();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let a_cols_slice = a_cols.as_slice().unwrap();
    let b_cols_slice = b_cols.as_slice().unwrap();

    let m = a_cols_slice.len();
    let n = b_cols_slice.len();
    let mut out = vec![vec![0.0; n]; m];

    // Build mask for which rows to include
    let row_set: HashSet<i32> = rows_slice.iter().copied().collect();

    for (ca, &a_col) in a_cols_slice.iter().enumerate() {
        for (cb, &b_col) in b_cols_slice.iter().enumerate() {
            let mut accum = 0.0;

            // Iterate over rows
            for &row in rows_slice {
                if !row_set.contains(&row) {
                    continue;
                }

                let i = row as usize;
                let start = a_indptr_slice[i] as usize;
                let end = a_indptr_slice[i + 1] as usize;

                // Find value at A[i, a_col]
                let mut a_val = 0.0;
                for idx in start..end {
                    if a_indices_slice[idx] == a_col {
                        a_val = a_data_slice[idx];
                        break;
                    }
                }

                let b_val = b_arr[[i, b_col as usize]];
                accum += a_val * d_slice[i] * b_val;
            }

            out[ca][cb] = accum;
        }
    }

    PyArray2::from_vec2_bound(py, &out).unwrap()
}
