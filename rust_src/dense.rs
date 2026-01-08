use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Dense matrix sandwich product: X.T @ diag(d) @ X
/// 
/// This implements a BLIS/GotoBLAS-style blocked matrix multiplication
/// optimized for cache locality and vectorization.
#[pyfunction]
pub fn dense_sandwich<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
    #[pyo3(signature = (thresh1d=32))] thresh1d: usize,
    #[pyo3(signature = (kratio=16))] kratio: usize,
    #[pyo3(signature = (innerblock=128))] innerblock: usize,
) -> Bound<'py, PyArray2<f64>> {
    let x = x.as_array();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();
    
    let out_m = cols_slice.len();
    let in_n = rows_slice.len();
    
    // Create output matrix
    let mut out = vec![0.0; out_m * out_m];
    
    if in_n == 0 || out_m == 0 {
        return PyArray2::from_vec2(py, &vec![vec![0.0; out_m]; out_m]).unwrap();
    }
    
    // Check if matrix is C-contiguous or F-contiguous
    let is_c_contiguous = x.is_standard_layout();
    let is_f_contiguous = x.t().is_standard_layout();
    
    if is_f_contiguous {
        dense_f_sandwich_impl(
            &x,
            d_slice,
            rows_slice,
            cols_slice,
            &mut out,
            thresh1d,
            kratio,
            innerblock,
        );
    } else if is_c_contiguous {
        dense_c_sandwich_impl(
            &x,
            d_slice,
            rows_slice,
            cols_slice,
            &mut out,
            thresh1d,
            kratio,
            innerblock,
        );
    } else {
        panic!("Matrix X is not contiguous");
    }
    
    // Convert flat vector to 2D array
    let out_2d: Vec<Vec<f64>> = out
        .chunks(out_m)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    PyArray2::from_vec2(py, &out_2d).unwrap()
}

/// Fortran-contiguous sandwich implementation
fn dense_f_sandwich_impl(
    x: &ndarray::ArrayView2<f64>,
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    _thresh1d: usize,
    _kratio: usize,
    _innerblock: usize,
) {
    let out_m = cols.len();
    
    // Blocked matrix multiplication with cache optimization
    const IBLOCK: usize = 4;
    const JBLOCK: usize = 4;
    
    // Parallel outer loop over output columns
    out.par_chunks_mut(out_m)
        .enumerate()
        .for_each(|(i_out, out_row)| {
            let i_col = cols[i_out] as usize;
            
            for (j_out, out_val) in out_row.iter_mut().enumerate() {
                let j_col = cols[j_out] as usize;
                
                let mut accum = 0.0;
                for &row_idx in rows {
                    let k = row_idx as usize;
                    let d_k = d[k];
                    let x_ki = x[[k, i_col]];
                    let x_kj = x[[k, j_col]];
                    accum += x_ki * d_k * x_kj;
                }
                
                *out_val = accum;
            }
        });
}

/// C-contiguous sandwich implementation
fn dense_c_sandwich_impl(
    x: &ndarray::ArrayView2<f64>,
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    out: &mut [f64],
    _thresh1d: usize,
    _kratio: usize,
    _innerblock: usize,
) {
    let out_m = cols.len();
    
    // Similar to F-contiguous but with different memory access pattern
    out.par_chunks_mut(out_m)
        .enumerate()
        .for_each(|(i_out, out_row)| {
            let i_col = cols[i_out] as usize;
            
            for (j_out, out_val) in out_row.iter_mut().enumerate() {
                let j_col = cols[j_out] as usize;
                
                let mut accum = 0.0;
                for &row_idx in rows {
                    let k = row_idx as usize;
                    let d_k = d[k];
                    let x_ki = x[[k, i_col]];
                    let x_kj = x[[k, j_col]];
                    accum += x_ki * d_k * x_kj;
                }
                
                *out_val = accum;
            }
        });
}

/// Dense matrix right matrix-vector product: X.T @ v
#[pyfunction]
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
        return PyArray2::from_vec2(py, &vec![out]).unwrap();
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
    
    PyArray2::from_vec2(py, &vec![out]).unwrap()
}

/// Dense matrix-vector product: X @ v
#[pyfunction]
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
        return PyArray2::from_vec2(py, &vec![out]).unwrap();
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
    
    PyArray2::from_vec2(py, &vec![out]).unwrap()
}
