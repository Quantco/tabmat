use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use wide::f64x4;

/// Dense matrix sandwich product: X.T @ diag(d) @ X
/// Optimized with 3D cache blocking (i, j, k dimensions)
#[pyfunction]
#[pyo3(signature = (x, d, rows, cols))]
pub fn dense_sandwich<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x_arr = x.as_array();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();
    let cols_slice = cols.as_slice().unwrap();
    
    let out_m = cols_slice.len();
    let n_rows = rows_slice.len();
    
    if n_rows == 0 || out_m == 0 {
        let empty_result: Vec<Vec<f64>> = vec![vec![0.0; out_m]; out_m];
        return PyArray2::from_vec2_bound(py, &empty_result).unwrap();
    }
    
    // Get raw slice and determine layout
    let x_slice = x_arr.as_slice();
    let (n_total_rows, n_total_cols) = (x_arr.nrows(), x_arr.ncols());
    let strides = x_arr.strides();
    let is_c_contiguous = strides[1] == 1;
    
    // Allocate output matrix
    let mut out_flat = vec![0.0; out_m * out_m];
    
    // Cache blocking parameters tuned for cache sizes
    const K_BLOCK: usize = 512;  // Block size for k dimension
    
    if let Some(x_slice) = x_slice {
        // Precompute sqrt(d) for all rows once
        let d_sqrt: Vec<f64> = rows_slice.iter()
            .map(|&r| d_slice[r as usize].sqrt())
            .collect();
        
        // Process in blocks of k dimension
        for k_block_start in (0..n_rows).step_by(K_BLOCK) {
            let k_block_end = (k_block_start + K_BLOCK).min(n_rows);
            let k_block_size = k_block_end - k_block_start;
            
            // Allocate flat buffer for weighted columns (column-major layout)
            // This matches C++'s L and R arrays
            let mut xw_flat = vec![0.0; out_m * k_block_size];
            
            // Precompute all weighted columns for this k-block in parallel
            // Store in column-major order: xw_flat[col * k_block_size + k_offset]
            xw_flat.par_chunks_mut(k_block_size)
                .enumerate()
                .for_each(|(col_idx, col_chunk)| {
                    let col_j = cols_slice[col_idx] as usize;
                    
                    for (k_offset, k_idx) in (k_block_start..k_block_end).enumerate() {
                        let r = rows_slice[k_idx] as usize;
                        let d_sqrt_val = d_sqrt[k_idx];
                        
                        col_chunk[k_offset] = if is_c_contiguous {
                            x_slice[r * n_total_cols + col_j] * d_sqrt_val
                        } else {
                            x_slice[col_j * n_total_rows + r] * d_sqrt_val
                        };
                    }
                });
            
            // Compute contributions using this k-block
            // Process upper triangle only
            let block_results: Vec<(usize, usize, f64)> = (0..out_m).into_par_iter()
                .flat_map(|i| {
                    let mut local_results = Vec::with_capacity(out_m - i);
                    let xi_base = i * k_block_size;
                    
                    for j in i..out_m {
                        let xj_base = j * k_block_size;
                        let mut sum = 0.0;
                        
                        // SIMD dot product with 4-way unrolling
                        let mut k = 0;
                        let k_simd_end = (k_block_size / 4) * 4;
                        
                        while k < k_simd_end {
                            let xi = f64x4::new([
                                xw_flat[xi_base + k],
                                xw_flat[xi_base + k + 1],
                                xw_flat[xi_base + k + 2],
                                xw_flat[xi_base + k + 3],
                            ]);
                            let xj = f64x4::new([
                                xw_flat[xj_base + k],
                                xw_flat[xj_base + k + 1],
                                xw_flat[xj_base + k + 2],
                                xw_flat[xj_base + k + 3],
                            ]);
                            let prod = xi * xj;
                            let arr = prod.to_array();
                            sum += arr[0] + arr[1] + arr[2] + arr[3];
                            k += 4;
                        }
                        
                        // Handle remainder
                        while k < k_block_size {
                            sum += xw_flat[xi_base + k] * xw_flat[xj_base + k];
                            k += 1;
                        }
                        
                        local_results.push((i, j, sum));
                    }
                    local_results
                })
                .collect();
            
            // Accumulate results into output matrix
            for (i, j, val) in block_results {
                out_flat[i * out_m + j] += val;
            }
        }
    } else {
        // Fallback for non-contiguous arrays
        let d_sqrt: Vec<f64> = rows_slice.iter()
            .map(|&r| d_slice[r as usize].sqrt())
            .collect();
        
        let results: Vec<(usize, usize, f64)> = (0..out_m).into_par_iter()
            .flat_map(|i| {
                let col_i = cols_slice[i] as usize;
                let mut local_results = Vec::new();
                
                for j in i..out_m {
                    let col_j = cols_slice[j] as usize;
                    let mut sum = 0.0;
                    
                    for k in 0..n_rows {
                        let r = rows_slice[k] as usize;
                        let d_sqrt_k = d_sqrt[k];
                        let xi_val = x_arr[[r, col_i]] * d_sqrt_k;
                        let xj_val = x_arr[[r, col_j]] * d_sqrt_k;
                        sum += xi_val * xj_val;
                    }
                    
                    local_results.push((i, j, sum));
                }
                local_results
            })
            .collect();
        
        for (i, j, val) in results {
            out_flat[i * out_m + j] = val;
        }
    }
    
    // Fill lower triangle by symmetry
    for i in 0..out_m {
        for j in (i + 1)..out_m {
            out_flat[j * out_m + i] = out_flat[i * out_m + j];
        }
    }
    
    // Convert flat vector to 2D for return
    let out_2d: Vec<Vec<f64>> = (0..out_m)
        .map(|i| {
            out_flat[i * out_m..(i + 1) * out_m].to_vec()
        })
        .collect();
    
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
