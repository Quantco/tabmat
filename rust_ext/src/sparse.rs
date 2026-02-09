//! Core sparse matrix operations implemented in Rust.
//!
//! These functions implement sparse matrix operations (CSR/CSC format) using
//! simple sequential iteration. The design prioritizes clarity and correctness.

// =============================================================================
// CSR matvec operations
// =============================================================================

/// CSR matrix-vector multiplication: out += X @ v (unrestricted)
///
/// For CSR format:
/// - data[indptr[i]..indptr[i+1]] contains the non-zero values in row i
/// - indices[indptr[i]..indptr[i+1]] contains the column indices for those values
pub fn csr_matvec_unrestricted(
    data: &[f64],
    indices: &[i32],
    indptr: &[i32],
    v: &[f64],
    out: &mut [f64],
) {
    let n_rows = indptr.len() - 1;

    for i in 0..n_rows {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;

        for idx in start..end {
            let j = indices[idx] as usize;
            let x_val = data[idx];
            out[i] += x_val * v[j];
        }
    }
}

/// CSR matrix-vector multiplication with row/column restrictions: out = X[rows, cols] @ v
///
/// Only considers specified rows and columns.
/// Output has length rows.len().
pub fn csr_matvec(
    data: &[f64],
    indices: &[i32],
    indptr: &[i32],
    v: &[f64],
    rows: &[i32],
    cols: &[i32],
    n_cols_total: usize,
) -> Vec<f64> {
    let n = rows.len();
    let mut out = vec![0.0; n];

    // Build column inclusion mask
    let mut col_included = vec![false; n_cols_total];
    for &col in cols {
        col_included[col as usize] = true;
    }

    for (ci, &row) in rows.iter().enumerate() {
        let i = row as usize;
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;

        for idx in start..end {
            let j = indices[idx] as usize;
            if !col_included[j] {
                continue;
            }
            let x_val = data[idx];
            out[ci] += x_val * v[j];
        }
    }

    out
}

// =============================================================================
// CSC rmatvec operations (transpose matvec)
// =============================================================================

/// CSC transpose matrix-vector multiplication: out += XT.T @ v (unrestricted)
///
/// XT is stored in CSC format (which is the same as X.T in CSR format).
/// This computes X.T @ v where XT represents the transpose.
///
/// For CSC format:
/// - data[indptr[j]..indptr[j+1]] contains the non-zero values in column j
/// - indices[indptr[j]..indptr[j+1]] contains the row indices for those values
pub fn csc_rmatvec_unrestricted(
    data: &[f64],
    indices: &[i32],
    indptr: &[i32],
    v: &[f64],
    out: &mut [f64],
) {
    let n_cols = indptr.len() - 1;

    for j in 0..n_cols {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;

        for idx in start..end {
            let i = indices[idx] as usize;
            let xt_val = data[idx];
            out[j] += xt_val * v[i];
        }
    }
}

/// CSC transpose matrix-vector multiplication with restrictions: out = XT[rows, cols].T @ v
///
/// Only considers specified rows (of original matrix) and columns.
/// Output has length cols.len().
pub fn csc_rmatvec(
    data: &[f64],
    indices: &[i32],
    indptr: &[i32],
    v: &[f64],
    rows: &[i32],
    cols: &[i32],
    n_rows_total: usize,
) -> Vec<f64> {
    let m = cols.len();
    let mut out = vec![0.0; m];

    // Build row inclusion mask
    let mut row_included = vec![false; n_rows_total];
    for &row in rows {
        row_included[row as usize] = true;
    }

    for (cj, &col) in cols.iter().enumerate() {
        let j = col as usize;
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;

        for idx in start..end {
            let i = indices[idx] as usize;
            if !row_included[i] {
                continue;
            }
            let xt_val = data[idx];
            out[cj] += xt_val * v[i];
        }
    }

    out
}

// =============================================================================
// sparse_sandwich
// =============================================================================

/// Sparse sandwich product: AT @ diag(d) @ A
///
/// Both A and AT are in CSC format. AT is the transpose of A.
/// This computes the symmetric matrix A.T @ diag(d) @ A.
///
/// Returns the upper triangle filled, then mirrors to lower triangle.
pub fn sparse_sandwich(
    a_data: &[f64],
    a_indices: &[i32],
    a_indptr: &[i32],
    at_data: &[f64],
    at_indices: &[i32],
    at_indptr: &[i32],
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    n_rows_total: usize,
    n_cols_total: usize,
) -> Vec<f64> {
    let m = cols.len();
    let mut out = vec![0.0; m * m];

    // Build row inclusion mask
    let mut row_included = vec![false; n_rows_total];
    for &row in rows {
        row_included[row as usize] = true;
    }

    // Build column map: original col index -> output col index (or -1 if not included)
    let mut col_map = vec![-1i32; n_cols_total];
    for (cj, &col) in cols.iter().enumerate() {
        col_map[col as usize] = cj as i32;
    }

    // For each output column Cj (corresponding to original column j)
    for (cj, &col_j) in cols.iter().enumerate() {
        let j = col_j as usize;

        // Iterate over non-zeros in column j of A
        let a_start = a_indptr[j] as usize;
        let a_end = a_indptr[j + 1] as usize;

        for a_idx in a_start..a_end {
            let k = a_indices[a_idx] as usize;
            if !row_included[k] {
                continue;
            }

            let a_val = a_data[a_idx] * d[k];

            // Now iterate over non-zeros in row k of AT (which is column k of AT in CSC)
            let at_start = at_indptr[k] as usize;
            let at_end = at_indptr[k + 1] as usize;

            for at_idx in at_start..at_end {
                let i = at_indices[at_idx] as usize;

                // Only compute upper triangle (i <= j)
                if i > j {
                    break;
                }

                let ci = col_map[i];
                if ci == -1 {
                    continue;
                }

                let at_val = at_data[at_idx];
                out[cj * m + ci as usize] += at_val * a_val;
            }
        }
    }

    // Mirror upper triangle to lower triangle
    for i in 0..m {
        for j in (i + 1)..m {
            out[i * m + j] = out[j * m + i];
        }
    }

    out
}

// =============================================================================
// csr_dense_sandwich
// =============================================================================

/// CSR-dense sandwich: A.T @ diag(d) @ B
///
/// A is a sparse CSR matrix, B is a dense matrix.
/// Returns a dense matrix of shape (len(A_cols), len(B_cols)).
pub fn csr_dense_sandwich(
    a_data: &[f64],
    a_indices: &[i32],
    a_indptr: &[i32],
    b: &[f64],
    b_shape: (usize, usize),
    d: &[f64],
    rows: &[i32],
    a_cols: &[i32],
    b_cols: &[i32],
    a_ncol: usize,
    is_c_contiguous: bool,
) -> Vec<f64> {
    let n_a_cols = a_cols.len();
    let n_b_cols = b_cols.len();
    let (_b_nrow, b_ncol) = b_shape;

    let mut out = vec![0.0; n_a_cols * n_b_cols];

    if rows.is_empty() || n_a_cols == 0 || n_b_cols == 0 {
        return out;
    }

    // Build A column map
    let mut a_col_map = vec![-1i32; a_ncol];
    for (ci, &col) in a_cols.iter().enumerate() {
        a_col_map[col as usize] = ci as i32;
    }

    // For each row k in the specified rows
    for &row in rows {
        let k = row as usize;
        let d_k = d[k];

        // Iterate over non-zeros in row k of A (CSR)
        let start = a_indptr[k] as usize;
        let end = a_indptr[k + 1] as usize;

        for a_idx in start..end {
            let i = a_indices[a_idx] as usize;
            let ci = a_col_map[i];
            if ci == -1 {
                continue;
            }
            let ci = ci as usize;

            let a_val = a_data[a_idx];
            let q = a_val * d_k;

            // Accumulate contribution for each B column
            for (cj, &b_col) in b_cols.iter().enumerate() {
                let j = b_col as usize;
                let b_val = if is_c_contiguous {
                    b[k * b_ncol + j]
                } else {
                    // Fortran contiguous
                    b[j * b_shape.0 + k]
                };
                out[ci * n_b_cols + cj] += q * b_val;
            }
        }
    }

    out
}

// =============================================================================
// transpose_square_dot_weights
// =============================================================================

/// Compute weighted squared column norms for CSC matrix.
///
/// For each column j: out[j] = sum over non-zeros in column j of (weights[i] * data[idx]^2)
/// where i is the row index.
///
/// This is used for computing column-wise weighted squared norms.
pub fn transpose_square_dot_weights(
    data: &[f64],
    indices: &[i32],
    indptr: &[i32],
    weights: &[f64],
) -> Vec<f64> {
    let n_cols = indptr.len() - 1;
    let mut out = vec![0.0; n_cols];

    for j in 0..n_cols {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;

        for idx in start..end {
            let i = indices[idx] as usize;
            let v = data[idx];
            out[j] += weights[i] * v * v;
        }
    }

    out
}
