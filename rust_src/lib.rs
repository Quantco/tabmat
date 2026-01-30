//! # tabmat_ext - High-Performance Matrix Operations for tabmat
//!
//! This crate provides Rust implementations of performance-critical matrix operations
//! for the tabmat Python library. These operations are exposed to Python via PyO3 and
//! are designed to accelerate generalized linear model (GLM) fitting.
//!
//! ## Architecture
//!
//! The crate is organized into four modules, each handling a specific matrix type:
//!
//! - [`dense`]: Operations on dense matrices (row-major NumPy arrays)
//! - [`sparse`]: Operations on sparse matrices (CSR/CSC formats)
//! - [`categorical`]: Operations on categorical matrices (one-hot encoded)
//! - [`split`]: Cross-product operations between different matrix types
//!
//! ## Key Operations
//!
//! The primary operations are:
//!
//! - **Sandwich products**: `X.T @ diag(d) @ X` - the core computation in GLM fitting
//! - **Matrix-vector products**: `X @ v` (matvec) and `X.T @ v` (rmatvec)
//! - **Weighted operations**: All operations support row subsets and diagonal weights
//!
//! ## Performance Features
//!
//! - **Parallelization**: Uses Rayon for multi-threaded execution
//! - **SIMD**: Uses `wide` crate for vectorized operations (f64x4)
//! - **Cache optimization**: BLIS-style blocking for better cache utilization
//! - **BLAS integration**: Uses Accelerate framework on macOS via cblas

#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(target_os = "linux")]
extern crate openblas_src;

#[cfg(target_os = "windows")]
extern crate openblas_src;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod categorical;
mod dense;
mod sparse;
mod split;

/// Python module providing high-performance matrix operations for tabmat.
///
/// This module exposes Rust implementations of matrix operations that are
/// performance-critical for GLM fitting. The operations are grouped by
/// the type of matrix they operate on.
#[pymodule]
fn tabmat_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Dense matrix operations
    // These operate on row-major NumPy arrays (C-contiguous)
    m.add_function(wrap_pyfunction!(dense::dense_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(dense::dense_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense::dense_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense::transpose_square_dot_weights, m)?)?;
    m.add_function(wrap_pyfunction!(dense::standardized_sandwich_correction, m)?)?;

    // Sparse matrix operations
    // These operate on CSR (Compressed Sparse Row) or CSC (Compressed Sparse Column) matrices
    m.add_function(wrap_pyfunction!(sparse::sparse_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csr_matvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csr_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csc_rmatvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csc_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::transpose_square_dot_weights, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csr_dense_sandwich, m)?)?;

    // Categorical matrix operations
    // These operate on one-hot encoded categorical variables stored as integer indices
    m.add_function(wrap_pyfunction!(categorical::categorical_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::matvec_fast, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::matvec_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::matvec_restricted, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::transpose_matvec_fast, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::transpose_matvec_fast_rows, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::transpose_matvec_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::transpose_matvec_complex_rows, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::transpose_matvec_restricted, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::sandwich_categorical_fast, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::sandwich_categorical_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::multiply_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::subset_categorical_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::get_col_included, m)?)?;

    // Split matrix operations
    // Cross-product operations between different matrix types (e.g., categorical × dense)
    m.add_function(wrap_pyfunction!(split::is_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(split::sandwich_cat_cat, m)?)?;
    m.add_function(wrap_pyfunction!(split::sandwich_cat_dense, m)?)?;
    m.add_function(wrap_pyfunction!(split::sandwich_cat_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(split::split_col_subsets, m)?)?;

    Ok(())
}
