use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Sparse matrix sandwich product using CSC format
#[pyfunction]
pub fn sparse_sandwich<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i32>,
    indptr: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let data_slice = data.as_slice().unwrap();
    let indices_slice = indices.as_slice().unwrap();
    let indptr_slice = indptr.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();
    
    let out_m = cols_slice.len();
    let mut out = vec![0.0; out_m * out_m];
    
    if rows_slice.is_empty() || out_m == 0 {
        return PyArray2::from_vec2(py, &vec![vec![0.0; out_m]; out_m]).unwrap();
    }
    
    // Build row set for efficient lookup
    let row_set: std::collections::HashSet<i32> = rows_slice.iter().copied().collect();
    
    // Compute sandwich product: X.T @ diag(d) @ X
    // For sparse CSC format, iterate over columns
    for (i_out, &i_col) in cols_slice.iter().enumerate() {
        let i_col = i_col as usize;
        
        for (j_out, &j_col) in cols_slice.iter().enumerate() {
            let j_col = j_col as usize;
            
            let mut accum = 0.0;
            
            // Get column i data
            let i_start = indptr_slice[i_col] as usize;
            let i_end = indptr_slice[i_col + 1] as usize;
            
            // Get column j data
            let j_start = indptr_slice[j_col] as usize;
            let j_end = indptr_slice[j_col + 1] as usize;
            
            // Find common rows (intersection)
            let mut i_ptr = i_start;
            let mut j_ptr = j_start;
            
            while i_ptr < i_end && j_ptr < j_end {
                let i_row = indices_slice[i_ptr];
                let j_row = indices_slice[j_ptr];
                
                if i_row == j_row && row_set.contains(&i_row) {
                    let row_idx = i_row as usize;
                    accum += data_slice[i_ptr] * d_slice[row_idx] * data_slice[j_ptr];
                    i_ptr += 1;
                    j_ptr += 1;
                } else if i_row < j_row {
                    i_ptr += 1;
                } else {
                    j_ptr += 1;
                }
            }
            
            out[i_out * out_m + j_out] = accum;
        }
    }
    
    // Convert to 2D array
    let out_2d: Vec<Vec<f64>> = out
        .chunks(out_m)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    PyArray2::from_vec2(py, &out_2d).unwrap()
}
