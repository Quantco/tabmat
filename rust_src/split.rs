// Split matrix operations for tabmat

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

/// Check if an array is sorted
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

/// Sandwich product between two categorical matrices
#[pyfunction]
#[pyo3(signature = (i_indices, j_indices, d, rows, i_ncol, j_ncol, i_drop_first, j_drop_first))]
pub fn sandwich_cat_cat<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<i32>,
    j_indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    i_ncol: usize,
    j_ncol: usize,
    i_drop_first: bool,
    j_drop_first: bool,
) -> Bound<'py, PyArray2<f64>> {
    let i_indices_slice = i_indices.as_slice().unwrap();
    let j_indices_slice = j_indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();

    let mut res = vec![vec![0.0; j_ncol]; i_ncol];

    for &k in rows_slice {
        let i_idx = if i_drop_first {
            i_indices_slice[k as usize] - 1
        } else {
            i_indices_slice[k as usize]
        };

        let j_idx = if j_drop_first {
            j_indices_slice[k as usize] - 1
        } else {
            j_indices_slice[k as usize]
        };

        if i_idx >= 0 && j_idx >= 0 {
            res[i_idx as usize][j_idx as usize] += d_slice[k as usize];
        }
    }

    PyArray2::from_vec2_bound(py, &res).unwrap()
}

/// Sandwich product between categorical and dense matrices
#[pyfunction]
#[pyo3(signature = (i_indices, d, mat_j, rows, j_cols, i_ncol, drop_first))]
pub fn sandwich_cat_dense<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    mat_j: PyReadonlyArray2<f64>,
    rows: PyReadonlyArray1<i32>,
    j_cols: PyReadonlyArray1<i32>,
    i_ncol: usize,
    drop_first: bool,
) -> Bound<'py, PyArray2<f64>> {
    let i_indices_slice = i_indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let mat_j_arr = mat_j.as_array();
    let rows_slice = rows.as_slice().unwrap();
    let j_cols_slice = j_cols.as_slice().unwrap();

    let nj_active_cols = j_cols_slice.len();
    let mut res = vec![vec![0.0; nj_active_cols]; i_ncol];

    for &k in rows_slice {
        let i_idx = if drop_first {
            i_indices_slice[k as usize] - 1
        } else {
            i_indices_slice[k as usize]
        };

        if i_idx >= 0 {
            let i_idx_usize = i_idx as usize;
            let k_usize = k as usize;
            
            for (cj, &j) in j_cols_slice.iter().enumerate() {
                let val = mat_j_arr[[k_usize, j as usize]];
                res[i_idx_usize][cj] += d_slice[k_usize] * val;
            }
        }
    }

    PyArray2::from_vec2_bound(py, &res).unwrap()
}

/// Split column subsets (simplified version)
/// This is a placeholder - the full implementation requires complex logic
#[pyfunction]
#[pyo3(signature = (indices_list, cols))]
pub fn split_col_subsets<'py>(
    py: Python<'py>,
    indices_list: &Bound<'py, PyList>,
    cols: PyReadonlyArray1<i32>,
) -> (Vec<Bound<'py, PyArray1<i32>>>, Vec<Bound<'py, PyArray1<i32>>>, usize) {
    let cols_slice = cols.as_slice().unwrap();
    let n_matrices = indices_list.len();
    
    let mut subset_cols_indices: Vec<Bound<'py, PyArray1<i32>>> = Vec::new();
    let mut subset_cols: Vec<Bound<'py, PyArray1<i32>>> = Vec::new();

    // Simplified implementation - just return empty arrays for now
    for _ in 0..n_matrices {
        subset_cols_indices.push(PyArray1::from_vec_bound(py, vec![]));
        subset_cols.push(PyArray1::from_vec_bound(py, vec![]));
    }

    (subset_cols_indices, subset_cols, cols_slice.len())
}
