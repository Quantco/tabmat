//! Core categorical matrix operations implemented in Rust.
//!
//! These functions implement the categorical matrix operations using simple
//! sequential iteration. The design prioritizes clarity and correctness.

// =============================================================================
// transpose_matvec
// =============================================================================

/// Transpose matrix-vector multiplication: X.T @ other
///
/// Computes X.T @ other where X is a categorical matrix represented by indices.
/// For each row i, X[i, indices[i]] = 1 and all other elements are 0.
/// So (X.T @ other)[j] = sum of other[i] for all i where indices[i] == j.
///
/// When drop_first is true, category 0 is skipped and indices are shifted.
pub fn transpose_matvec(
    indices: &[i32],
    other: &[f64],
    out_size: usize,
    drop_first: bool,
) -> Vec<f64> {
    let mut result = vec![0.0; out_size];
    let offset = if drop_first { 1 } else { 0 };

    for (i, &idx) in indices.iter().enumerate() {
        let col_idx = idx - offset;
        if col_idx >= 0 {
            result[col_idx as usize] += other[i];
        }
    }

    result
}

// =============================================================================
// matvec
// =============================================================================

/// Matrix-vector multiplication: out[i] += other[indices[i] - offset]
///
/// For categorical matrices, this is a simple gather operation.
/// When drop_first is true, category 0 contributes nothing.
pub fn matvec(indices: &[i32], other: &[f64], out: &mut [f64], drop_first: bool) {
    let offset = if drop_first { 1 } else { 0 };

    for (i, &idx) in indices.iter().enumerate() {
        let col_idx = idx - offset;
        if col_idx >= 0 {
            out[i] += other[col_idx as usize];
        }
    }
}

// =============================================================================
// sandwich_categorical
// =============================================================================

/// Sandwich product returning diagonal: X.T @ diag(d) @ X
///
/// For categorical matrices, the result is always diagonal.
/// Result[j] = sum of d[k] for all rows k in `rows` where indices[k] == j.
pub fn sandwich_diagonal(
    indices: &[i32],
    d: &[f64],
    rows: &[i32],
    n_cols: usize,
    drop_first: bool,
) -> Vec<f64> {
    let mut result = vec![0.0; n_cols];
    let offset = if drop_first { 1 } else { 0 };

    for &row in rows {
        let k = row as usize;
        let col_idx = indices[k] - offset;
        if col_idx >= 0 {
            result[col_idx as usize] += d[k];
        }
    }

    result
}

// =============================================================================
// sandwich_cat_cat
// =============================================================================

/// Cat-cat sandwich: X1.T @ diag(d) @ X2
///
/// Result[i, j] = sum of d[k] for rows k where i_indices[k] == i and j_indices[k] == j.
/// Returns a flat vector in row-major order.
pub fn sandwich_cat_cat(
    i_indices: &[i32],
    j_indices: &[i32],
    d: &[f64],
    rows: &[i32],
    i_ncol: usize,
    j_ncol: usize,
    i_drop_first: bool,
    j_drop_first: bool,
) -> Vec<f64> {
    let mut result = vec![0.0; i_ncol * j_ncol];
    let i_offset = if i_drop_first { 1 } else { 0 };
    let j_offset = if j_drop_first { 1 } else { 0 };

    for &row in rows {
        let k = row as usize;
        let i = i_indices[k] - i_offset;
        let j = j_indices[k] - j_offset;

        if i >= 0 && j >= 0 {
            result[i as usize * j_ncol + j as usize] += d[k];
        }
    }

    result
}

// =============================================================================
// sandwich_cat_dense
// =============================================================================

/// Cat-dense sandwich: X_cat.T @ diag(d) @ X_dense
///
/// Result[i, j_idx] = sum over k of (d[k] * X_dense[k, j_cols[j_idx]]) where i_indices[k] == i.
/// Supports both C-contiguous and Fortran-contiguous layouts.
pub fn sandwich_cat_dense(
    i_indices: &[i32],
    d: &[f64],
    mat_j: &[f64],
    mat_j_shape: (usize, usize),
    rows: &[i32],
    j_cols: &[i32],
    i_ncol: usize,
    is_c_contiguous: bool,
    drop_first: bool,
) -> Vec<f64> {
    let (mat_j_nrow, mat_j_ncol) = mat_j_shape;
    let len_j_cols = j_cols.len();
    let mut result = vec![0.0; i_ncol * len_j_cols];
    let offset = if drop_first { 1 } else { 0 };

    for &row in rows {
        let k = row as usize;
        let i = i_indices[k] - offset;

        if i >= 0 {
            let i = i as usize;
            let d_k = d[k];

            for (j_idx, &j_col) in j_cols.iter().enumerate() {
                let j = j_col as usize;
                let mat_val = if is_c_contiguous {
                    mat_j[k * mat_j_ncol + j]
                } else {
                    mat_j[j * mat_j_nrow + k]
                };
                result[i * len_j_cols + j_idx] += d_k * mat_val;
            }
        }
    }

    result
}
