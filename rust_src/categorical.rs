use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Categorical matrix sandwich product
/// 
/// Exploits the structure of categorical matrices where each row has exactly
/// one non-zero entry (which is 1), making this O(n_categories²) instead of
/// O(n_samples * n_categories).
#[pyfunction]
pub fn categorical_sandwich<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
    n_categories: usize,
) -> Bound<'py, PyArray2<f64>> {
    let indices_slice = indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();
    
    let out_m = cols_slice.len();
    let mut out = vec![0.0; out_m * out_m];
    
    if rows_slice.is_empty() || out_m == 0 {
        return PyArray2::from_vec2(py, &vec![vec![0.0; out_m]; out_m]).unwrap();
    }
    
    // Build weighted sums for each category
    // weighted_sums[cat] = sum of d[i] for all rows i where indices[i] == cat
    let mut weighted_sums = vec![0.0; n_categories];
    
    for &row_idx in rows_slice {
        let row = row_idx as usize;
        let cat = indices_slice[row] as usize;
        weighted_sums[cat] += d_slice[row];
    }
    
    // Compute sandwich: for categories i and j, result[i,j] = weighted_sums[i] * (i == j)
    // Since each row has exactly one 1, X.T @ diag(d) @ X is diagonal with weighted_sums
    for (i_out, &i_col) in cols_slice.iter().enumerate() {
        let i_cat = i_col as usize;
        
        for (j_out, &j_col) in cols_slice.iter().enumerate() {
            let j_cat = j_col as usize;
            
            if i_cat == j_cat {
                out[i_out * out_m + j_out] = weighted_sums[i_cat];
            }
        }
    }
    
    // Convert to 2D array
    let out_2d: Vec<Vec<f64>> = out
        .chunks(out_m)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    PyArray2::from_vec2(py, &out_2d).unwrap()
}
