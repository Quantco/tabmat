use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Dense matrix sandwich product: X.T @ diag(d) @ X
/// Optimized with cache blocking, avoiding ndarray indexing overhead
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
    let n_rows = rows_slice.len();
    
    if n_rows == 0 || out_m == 0 {
        let empty_result: Vec<Vec<f64>> = vec![vec![0.0; out_m]; out_m];
        return PyArray2::from_vec2_bound(py, &empty_result).unwrap();
    }
    
    // Check matrix layout
    let strides = x.strides();
    let is_fortran = strides[0] == 1;
    
    //Cache blocking tuned for L2 cache
    const BLOCK_K: usize = 1024;
    
    // Pre-compute weighted columns: Xw[j][k] = sqrt(d[rows[k]]) * X[rows[k], cols[j]]
    let xw: Vec<Vec<f64>> = cols_slice
        .par_iter()
        .map(|&col_j| {
            let col_j = col_j as usize;
            let mut xw_col = Vec::with_capacity(n_rows);
            
            for &row_k in rows_slice.iter() {
                let row_k = row_k as usize;
                let val = d_slice[row_k].sqrt() * x[[row_k, col_j]];
                xw_col.push(val);
            }
            xw_col
        })
        .collect();
    
    // Compute Xw.T @ Xw in parallel over rows, only upper triangle
    let out_flat: Vec<f64> = (0..out_m)
        .into_par_iter()
        .flat_map(|i| {
            let mut row = vec![0.0; out_m];
            
            // Only compute from i onwards (upper triangle)
            for j in i..out_m {
                let mut sum = 0.0;
                
                // Process in blocks for better cache reuse
                for k_start in (0..n_rows).step_by(BLOCK_K) {
                    let k_end = (k_start + BLOCK_K).min(n_rows);
                    
                    let xw_i = &xw[i][k_start..k_end];
                    let xw_j = &xw[j][k_start..k_end];
                    
                    // Unroll by 8 for ILP and auto-vectorization
                    let len = xw_i.len();
                    let len_unroll = (len / 8) * 8;
                    
                    let mut k = 0;
                    while k < len_unroll {
                        sum += xw_i[k] * xw_j[k];
                        sum += xw_i[k + 1] * xw_j[k + 1];
                        sum += xw_i[k + 2] * xw_j[k + 2];
                        sum += xw_i[k + 3] * xw_j[k + 3];
                        sum += xw_i[k + 4] * xw_j[k + 4];
                        sum += xw_i[k + 5] * xw_j[k + 5];
                        sum += xw_i[k + 6] * xw_j[k + 6];
                        sum += xw_i[k + 7] * xw_j[k + 7];
                        k += 8;
                    }
                    
                    while k < len {
                        sum += xw_i[k] * xw_j[k];
                        k += 1;
                    }
                }
                
                row[j] = sum;
            }
            row
        })
        .collect();
    
    // Fill symmetric lower triangle
    let mut out_2d = vec![vec![0.0; out_m]; out_m];
    for i in 0..out_m {
        for j in i..out_m {
            let val = out_flat[i * out_m + j];
            out_2d[i][j] = val;
            out_2d[j][i] = val;
        }
    }
    
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
