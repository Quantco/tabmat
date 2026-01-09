use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Dense matrix sandwich product: X.T @ diag(d) @ X
#[pyfunction]
#[pyo3(signature = (x, d, rows, cols))]
pub fn dense_sandwich<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x = x.as_array();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();
    
    let out_m = cols_slice.len();
    
    if rows_slice.is_empty() || out_m == 0 {
        let empty_result: Vec<Vec<f64>> = vec![vec![0.0; out_m]; out_m];
        return PyArray2::from_vec2_bound(py, &empty_result).unwrap();
    }
    
    // Compute X.T @ diag(d) @ X using parallel outer product accumulation
    let mut out_2d = vec![vec![0.0; out_m]; out_m];
    
    // Parallel over output rows
    out_2d.par_iter_mut().enumerate().for_each(|(i, out_row)| {
        let col_i = cols_slice[i] as usize;
        
        for j in 0..out_m {
            let col_j = cols_slice[j] as usize;
            let mut accum = 0.0;
            
            for &row_idx in rows_slice.iter() {
                let row = row_idx as usize;
                accum += x[[row, col_i]] * d_slice[row] * x[[row, col_j]];
            }
            
            out_row[j] = accum;
        }
    });
    
    PyArray2::from_vec2_bound(py, &out_2d).unwrap()
}

/// Dense reverse matrix-vector product: X.T @ v
#[pyfunction]
#[pyo3(signature = (x, v, rows, cols))]
pub fn dense_rmatvec<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    v: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x = x.as_array();
    let v_slice = v.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();
    
    let n_cols = cols_slice.len();
    let mut out = vec![0.0; n_cols];
    
    if rows_slice.is_empty() || n_cols == 0 {
        let result_2d: Vec<Vec<f64>> = vec![out];
        return PyArray2::from_vec2_bound(py, &result_2d).unwrap();
    }
    
    // Compute X.T @ v for selected rows and columns
    out.par_iter_mut()
        .enumerate()
        .for_each(|(j, out_val)| {
            let col_idx = cols_slice[j] as usize;
            let mut accum = 0.0;
            
            for (i, &row_idx) in rows_slice.iter().enumerate() {
                let row = row_idx as usize;
                accum += x[[row, col_idx]] * v_slice[i];
            }
            
            *out_val = accum;
        });
    
    let result_2d: Vec<Vec<f64>> = vec![out];
    PyArray2::from_vec2_bound(py, &result_2d).unwrap()
}

/// Dense matrix-vector product: X @ v
#[pyfunction]
#[pyo3(signature = (x, v, rows, cols))]
pub fn dense_matvec<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    v: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x = x.as_array();
    let v_slice = v.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();
    
    let n_rows = rows_slice.len();
    let mut out = vec![0.0; n_rows];
    
    if n_rows == 0 || cols_slice.is_empty() {
        let result_2d: Vec<Vec<f64>> = vec![out];
        return PyArray2::from_vec2_bound(py, &result_2d).unwrap();
    }
    
    // Compute X @ v for selected rows and columns
    out.par_iter_mut()
        .enumerate()
        .for_each(|(i, out_val)| {
            let row_idx = rows_slice[i] as usize;
            let mut accum = 0.0;
            
            for (j, &col_idx) in cols_slice.iter().enumerate() {
                let col = col_idx as usize;
                accum += x[[row_idx, col]] * v_slice[j];
            }
            
            *out_val = accum;
        });
    
    let result_2d: Vec<Vec<f64>> = vec![out];
    PyArray2::from_vec2_bound(py, &result_2d).unwrap()
}

/// Transpose square dot weights: sum_i weights[i] * (X[i,j] - shift[j])^2
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
