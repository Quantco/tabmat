//! Core dense matrix operations implemented in Rust.
//!
//! These functions implement dense matrix operations using simple sequential
//! iteration.

// =============================================================================
// Helper: access dense matrix elements
// =============================================================================

/// Get element from dense matrix at (row, col).
/// Handles both C-contiguous (row-major) and Fortran-contiguous (column-major) layouts.
fn get_element(data: &[f64], row: usize, col: usize, n_rows: usize, n_cols: usize, is_c_contiguous: bool) -> f64 {
    if is_c_contiguous {
        data[row * n_cols + col]
    } else {
        data[col * n_rows + row]
    }
}

// =============================================================================
// dense_sandwich
// =============================================================================

/// Dense sandwich product: X.T @ diag(d) @ X
///
/// Computes the symmetric matrix X[rows, cols].T @ diag(d[rows]) @ X[rows, cols].
/// Returns a flattened array of shape (len(cols), len(cols)) in row-major order.
pub fn dense_sandwich(
    x: &[f64],
    x_shape: (usize, usize),
    d: &[f64],
    rows: &[i32],
    cols: &[i32],
    is_c_contiguous: bool,
) -> Vec<f64> {
    let (n_rows, n_cols) = x_shape;
    let out_m = cols.len();
    let mut out = vec![0.0; out_m * out_m];

    if rows.is_empty() || out_m == 0 {
        return out;
    }

    // Compute upper triangle (including diagonal)
    for (ci, &col_i) in cols.iter().enumerate() {
        let i = col_i as usize;

        for (cj, &col_j) in cols.iter().enumerate().skip(ci) {
            let j = col_j as usize;

            // Sum over restricted rows: d[k] * X[k, i] * X[k, j]
            out[ci * out_m + cj] = rows
                .iter()
                .map(|&row| {
                    let k = row as usize;
                    let x_ki = get_element(x, k, i, n_rows, n_cols, is_c_contiguous);
                    let x_kj = get_element(x, k, j, n_rows, n_cols, is_c_contiguous);
                    d[k] * x_ki * x_kj
                })
                .sum();
        }
    }

    // Mirror upper triangle to lower triangle
    for ci in 0..out_m {
        for cj in (ci + 1)..out_m {
            out[cj * out_m + ci] = out[ci * out_m + cj];
        }
    }

    out
}

// =============================================================================
// dense_rmatvec
// =============================================================================

/// Dense transpose matrix-vector multiplication: X[rows, cols].T @ v[rows]
///
/// Returns a vector of length cols.len().
pub fn dense_rmatvec(
    x: &[f64],
    x_shape: (usize, usize),
    v: &[f64],
    rows: &[i32],
    cols: &[i32],
    is_c_contiguous: bool,
) -> Vec<f64> {
    let (n_rows, n_cols) = x_shape;

    cols.iter()
        .map(|&col| {
            let j = col as usize;
            rows.iter()
                .map(|&row| {
                    let i = row as usize;
                    get_element(x, i, j, n_rows, n_cols, is_c_contiguous) * v[i]
                })
                .sum()
        })
        .collect()
}

// =============================================================================
// dense_matvec
// =============================================================================

/// Dense matrix-vector multiplication: X[rows, cols] @ v[cols]
///
/// Returns a vector of length rows.len().
pub fn dense_matvec(
    x: &[f64],
    x_shape: (usize, usize),
    v: &[f64],
    rows: &[i32],
    cols: &[i32],
    is_c_contiguous: bool,
) -> Vec<f64> {
    let (n_rows, n_cols) = x_shape;

    rows.iter()
        .map(|&row| {
            let i = row as usize;
            cols.iter()
                .map(|&col| {
                    let j = col as usize;
                    get_element(x, i, j, n_rows, n_cols, is_c_contiguous) * v[j]
                })
                .sum()
        })
        .collect()
}

// =============================================================================
// transpose_square_dot_weights
// =============================================================================

/// Compute weighted squared column norms with shift for dense matrix.
///
/// For each column j: out[j] = sum_i(weights[i] * (X[i, j] - shift[j])^2)
pub fn dense_transpose_square_dot_weights(
    x: &[f64],
    x_shape: (usize, usize),
    weights: &[f64],
    shift: &[f64],
    is_c_contiguous: bool,
) -> Vec<f64> {
    let (n_rows, n_cols) = x_shape;

    (0..n_cols)
        .map(|j| {
            (0..n_rows)
                .map(|i| {
                    let x_ij = get_element(x, i, j, n_rows, n_cols, is_c_contiguous);
                    weights[i] * (x_ij - shift[j]).powi(2)
                })
                .sum()
        })
        .collect()
}
