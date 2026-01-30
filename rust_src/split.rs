//! Split matrix operations for tabmat.
//!
//! Operations for matrices composed of multiple sub-matrices (dense, sparse, categorical).
//! Common in regression models with mixed feature types.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rayon::prelude::*;

/// Block size for k-dimension parallelism.
const KBLOCK: usize = 1024;

// =============================================================================
// Utility functions
// =============================================================================

/// Checks if an array is sorted in non-decreasing order.
#[pyfunction]
#[pyo3(signature = (a,))]
pub fn is_sorted(a: &Bound<'_, PyAny>) -> PyResult<bool> {
    let arr: PyReadonlyArray1<i64> = a.extract()?;
    let a_slice = arr.as_slice()?;

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

/// Maps global column indices to local sub-matrix indices.
///
/// Returns (subset_cols_indices, subset_cols, n_cols) where:
/// - subset_cols_indices[i]: positions in cols array for sub-matrix i
/// - subset_cols[i]: local column indices within sub-matrix i
#[pyfunction]
#[pyo3(signature = (indices_list, cols))]
pub fn split_col_subsets<'py>(
    py: Python<'py>,
    indices_list: &Bound<'py, PyList>,
    cols: PyReadonlyArray1<i32>,
) -> PyResult<(
    Vec<Bound<'py, PyArray1<i32>>>,
    Vec<Bound<'py, PyArray1<i32>>>,
    usize,
)> {
    let cols_slice = cols.as_slice()?;
    let n_matrices = indices_list.len();

    // Convert indices_list to Vec of Vec<i32>
    let mut indices: Vec<Vec<i32>> = Vec::new();
    for i in 0..n_matrices {
        let item = indices_list.get_item(i)?;
        if let Ok(arr) = item.downcast::<PyArray1<i32>>() {
            indices.push(arr.readonly().as_slice()?.to_vec());
        } else if let Ok(arr) = item.downcast::<PyArray1<i64>>() {
            indices.push(arr.readonly().as_slice()?.iter().map(|&x| x as i32).collect());
        } else if let Ok(arr) = item.downcast::<PyArray1<isize>>() {
            indices.push(arr.readonly().as_slice()?.iter().map(|&x| x as i32).collect());
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Expected PyArray1 with integer dtype, got {:?}",
                item.get_type()
            )));
        }
    }

    let mut next_subset_idx: Vec<usize> = vec![0; n_matrices];
    let mut subset_cols_indices_vecs: Vec<Vec<i32>> = vec![Vec::new(); n_matrices];
    let mut subset_cols_vecs: Vec<Vec<i32>> = vec![Vec::new(); n_matrices];

    for (i, &col) in cols_slice.iter().enumerate() {
        for j in 0..n_matrices {
            while next_subset_idx[j] < indices[j].len() && indices[j][next_subset_idx[j]] < col {
                next_subset_idx[j] += 1;
            }

            if next_subset_idx[j] < indices[j].len() && indices[j][next_subset_idx[j]] == col {
                subset_cols_indices_vecs[j].push(i as i32);
                subset_cols_vecs[j].push(next_subset_idx[j] as i32);
                next_subset_idx[j] += 1;
                break;
            }
        }
    }

    let subset_cols_indices: Vec<_> = subset_cols_indices_vecs
        .into_iter()
        .map(|v| PyArray1::from_vec_bound(py, v))
        .collect();
    let subset_cols: Vec<_> = subset_cols_vecs
        .into_iter()
        .map(|v| PyArray1::from_vec_bound(py, v))
        .collect();

    Ok((subset_cols_indices, subset_cols, cols_slice.len()))
}

// =============================================================================
// Sandwich products
// =============================================================================

/// Computes sandwich product: `Cat_i.T @ diag(d) @ Cat_j`.
#[pyfunction]
#[pyo3(signature = (i_indices, j_indices, i_ncol, j_ncol, d, rows, _dtype, i_drop_first, j_drop_first, i_has_missings, j_has_missings))]
pub fn sandwich_cat_cat<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<i32>,
    j_indices: PyReadonlyArray1<i32>,
    i_ncol: usize,
    j_ncol: usize,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    _dtype: Bound<'py, PyAny>,
    i_drop_first: bool,
    j_drop_first: bool,
    i_has_missings: bool,
    j_has_missings: bool,
) -> Bound<'py, PyArray2<f64>> {
    let i_indices_slice = i_indices.as_slice().unwrap();
    let j_indices_slice = j_indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let rows_slice = rows.as_slice().unwrap();

    let n_rows = rows_slice.len();
    let res_size = i_ncol * j_ncol;

    if n_rows == 0 || i_ncol == 0 || j_ncol == 0 {
        return PyArray2::zeros_bound(py, [i_ncol, j_ncol], false);
    }

    let num_threads = rayon::current_num_threads();
    let mut all_res = vec![0.0f64; num_threads * res_size];
    let needs_complex = i_drop_first || j_drop_first || i_has_missings || j_has_missings;

    {
        use rayon::iter::IndexedParallelIterator;
        let chunk_size = (n_rows + num_threads - 1) / num_threads;

        all_res
            .par_chunks_mut(res_size)
            .enumerate()
            .for_each(|(tid, res_local)| {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(n_rows);

                if needs_complex {
                    for k_idx in start..end {
                        let k = unsafe { *rows_slice.get_unchecked(k_idx) } as usize;
                        let i_raw = unsafe { *i_indices_slice.get_unchecked(k) };
                        let j_raw = unsafe { *j_indices_slice.get_unchecked(k) };
                        let i_idx = if i_drop_first { i_raw - 1 } else { i_raw };
                        let j_idx = if j_drop_first { j_raw - 1 } else { j_raw };

                        if i_idx >= 0 && j_idx >= 0 {
                            let out_idx = i_idx as usize * j_ncol + j_idx as usize;
                            unsafe {
                                *res_local.get_unchecked_mut(out_idx) += *d_slice.get_unchecked(k);
                            }
                        }
                    }
                } else {
                    for k_idx in start..end {
                        let k = unsafe { *rows_slice.get_unchecked(k_idx) } as usize;
                        let i = unsafe { *i_indices_slice.get_unchecked(k) } as usize;
                        let j = unsafe { *j_indices_slice.get_unchecked(k) } as usize;
                        unsafe {
                            *res_local.get_unchecked_mut(i * j_ncol + j) += *d_slice.get_unchecked(k);
                        }
                    }
                }
            });
    }

    // Reduce thread results
    let (result, rest) = all_res.split_at_mut(res_size);
    for thread_buf in rest.chunks(res_size) {
        for (r, &t) in result.iter_mut().zip(thread_buf.iter()) {
            *r += t;
        }
    }

    let out = PyArray2::zeros_bound(py, [i_ncol, j_ncol], false);
    {
        let mut out_rw = unsafe { out.as_array_mut() };
        for i in 0..i_ncol {
            for j in 0..j_ncol {
                out_rw[[i, j]] = result[i * j_ncol + j];
            }
        }
    }
    out
}

/// Computes sandwich product: `Cat.T @ diag(d) @ Dense`.
#[pyfunction]
#[pyo3(signature = (i_indices, i_ncol, d, mat_j, rows, j_cols, _is_c_contiguous, has_missings=false, drop_first=false))]
pub fn sandwich_cat_dense<'py>(
    py: Python<'py>,
    i_indices: PyReadonlyArray1<i32>,
    i_ncol: usize,
    d: PyReadonlyArray1<f64>,
    mat_j: PyReadonlyArray2<f64>,
    rows: PyReadonlyArray1<i32>,
    j_cols: PyReadonlyArray1<i32>,
    _is_c_contiguous: bool,
    has_missings: bool,
    drop_first: bool,
) -> Bound<'py, PyArray2<f64>> {
    let i_indices_slice = i_indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let mat_j_arr = mat_j.as_array();
    let rows_slice = rows.as_slice().unwrap();
    let j_cols_slice = j_cols.as_slice().unwrap();

    let n_rows = rows_slice.len();
    let nj_active_cols = j_cols_slice.len();
    let res_size = i_ncol * nj_active_cols;

    if n_rows == 0 || nj_active_cols == 0 || i_ncol == 0 {
        return PyArray2::zeros_bound(py, [i_ncol, nj_active_cols], false);
    }

    let mat_j_ncols = mat_j_arr.ncols();
    let mat_j_slice_opt = mat_j_arr.as_slice();
    let j_cols_usize: Vec<usize> = j_cols_slice.iter().map(|&c| c as usize).collect();
    let n_kblocks = (n_rows + KBLOCK - 1) / KBLOCK;
    let needs_bounds_check = drop_first || has_missings;

    let result = (0..n_kblocks)
        .into_par_iter()
        .fold(
            || vec![0.0f64; res_size],
            |mut res_local, kb| {
                let k_start = kb * KBLOCK;
                let k_end = (k_start + KBLOCK).min(n_rows);

                for k_idx in k_start..k_end {
                    let k = rows_slice[k_idx] as usize;
                    let i_idx = if drop_first {
                        i_indices_slice[k] - 1
                    } else {
                        i_indices_slice[k]
                    };

                    if !needs_bounds_check || i_idx >= 0 {
                        let i = i_idx as usize;
                        let d_k = d_slice[k];
                        let out_row_start = i * nj_active_cols;

                        if let Some(mat_j_slice) = mat_j_slice_opt {
                            let mat_j_row_start = k * mat_j_ncols;
                            for (j_local, &j) in j_cols_usize.iter().enumerate() {
                                res_local[out_row_start + j_local] +=
                                    d_k * mat_j_slice[mat_j_row_start + j];
                            }
                        } else {
                            for (j_local, &j) in j_cols_usize.iter().enumerate() {
                                res_local[out_row_start + j_local] += d_k * mat_j_arr[[k, j]];
                            }
                        }
                    }
                }
                res_local
            },
        )
        .reduce(
            || vec![0.0f64; res_size],
            |mut a, b| {
                for i in 0..res_size {
                    a[i] += b[i];
                }
                a
            },
        );

    let out_2d: Vec<Vec<f64>> = (0..i_ncol)
        .map(|i| result[i * nj_active_cols..(i + 1) * nj_active_cols].to_vec())
        .collect();
    PyArray2::from_vec2_bound(py, &out_2d).unwrap()
}

/// Computes sandwich product: `Cat.T @ diag(d) @ Sparse` (CSC format).
#[pyfunction]
#[pyo3(signature = (cat_indices, cat_ncol, d, sparse_data, sparse_indices, sparse_indptr, rows, l_cols, r_cols, has_missings=false, drop_first=false))]
pub fn sandwich_cat_sparse<'py>(
    py: Python<'py>,
    cat_indices: PyReadonlyArray1<i32>,
    cat_ncol: usize,
    d: PyReadonlyArray1<f64>,
    sparse_data: PyReadonlyArray1<f64>,
    sparse_indices: PyReadonlyArray1<i32>,
    sparse_indptr: PyReadonlyArray1<i32>,
    rows: Option<PyReadonlyArray1<i32>>,
    l_cols: Option<PyReadonlyArray1<i32>>,
    r_cols: Option<PyReadonlyArray1<i32>>,
    has_missings: bool,
    drop_first: bool,
) -> Bound<'py, PyArray2<f64>> {
    let cat_indices_slice = cat_indices.as_slice().unwrap();
    let d_slice = d.as_slice().unwrap();
    let sparse_data_slice = sparse_data.as_slice().unwrap();
    let sparse_indices_slice = sparse_indices.as_slice().unwrap();
    let sparse_indptr_slice = sparse_indptr.as_slice().unwrap();

    let n_sparse_cols = sparse_indptr_slice.len() - 1;
    let n_total_rows = cat_indices_slice.len();

    // Row restriction lookup
    let (row_included, use_all_rows) = match &rows {
        None => (vec![], true),
        Some(rows_arr) => {
            let rows_slice = rows_arr.as_slice().unwrap();
            let mut included = vec![false; n_total_rows];
            for &r in rows_slice {
                if (r as usize) < n_total_rows {
                    included[r as usize] = true;
                }
            }
            (included, false)
        }
    };

    // Categorical column mapping
    let (cat_col_map, n_out_cat_cols) = match &l_cols {
        None => ((0..cat_ncol as i32).collect::<Vec<_>>(), cat_ncol),
        Some(cols_arr) => {
            let cols_slice = cols_arr.as_slice().unwrap();
            let mut map = vec![-1i32; cat_ncol];
            for (out_idx, &col) in cols_slice.iter().enumerate() {
                if (col as usize) < cat_ncol {
                    map[col as usize] = out_idx as i32;
                }
            }
            (map, cols_slice.len())
        }
    };

    // Sparse column restriction
    let sparse_cols: Vec<usize> = match &r_cols {
        None => (0..n_sparse_cols).collect(),
        Some(cols_arr) => cols_arr.as_slice().unwrap().iter().map(|&c| c as usize).collect(),
    };
    let n_out_sparse_cols = sparse_cols.len();

    if n_out_cat_cols == 0 || n_out_sparse_cols == 0 {
        return PyArray2::zeros_bound(py, [n_out_cat_cols, n_out_sparse_cols], false);
    }

    let needs_bounds_check = drop_first || has_missings;

    // Parallel over sparse columns
    let result: Vec<Vec<f64>> = sparse_cols
        .par_iter()
        .map(|&j| {
            let mut col_result = vec![0.0f64; n_out_cat_cols];
            let ptr_start = sparse_indptr_slice[j] as usize;
            let ptr_end = sparse_indptr_slice[j + 1] as usize;

            for ptr in ptr_start..ptr_end {
                let k = sparse_indices_slice[ptr] as usize;
                let v = sparse_data_slice[ptr];

                if !use_all_rows && !row_included[k] {
                    continue;
                }

                let cat_idx = if drop_first {
                    cat_indices_slice[k] - 1
                } else {
                    cat_indices_slice[k]
                };

                if needs_bounds_check && cat_idx < 0 {
                    continue;
                }

                let cat_col = cat_idx as usize;
                if cat_col >= cat_ncol {
                    continue;
                }

                let out_cat_col = cat_col_map[cat_col];
                if out_cat_col >= 0 {
                    col_result[out_cat_col as usize] += d_slice[k] * v;
                }
            }
            col_result
        })
        .collect();

    // Build output array
    let out = PyArray2::zeros_bound(py, [n_out_cat_cols, n_out_sparse_cols], false);
    {
        let mut out_rw = unsafe { out.as_array_mut() };
        for (j, col) in result.iter().enumerate() {
            for (i, &val) in col.iter().enumerate() {
                out_rw[[i, j]] = val;
            }
        }
    }
    out
}
