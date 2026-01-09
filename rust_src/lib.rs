use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod dense;
mod sparse;
mod categorical;
mod split;

/// Tabmat Rust extensions
#[pymodule]
fn tabmat_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Dense matrix operations
    m.add_function(wrap_pyfunction!(dense::dense_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(dense::dense_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense::dense_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense::transpose_square_dot_weights, m)?)?;
    
    // Sparse matrix operations
    m.add_function(wrap_pyfunction!(sparse::sparse_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csr_matvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csr_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csc_rmatvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csc_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::transpose_square_dot_weights, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::csr_dense_sandwich, m)?)?;
    
    // Categorical matrix operations
    m.add_function(wrap_pyfunction!(categorical::categorical_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::matvec_fast, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::matvec_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::transpose_matvec_fast, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::transpose_matvec_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::sandwich_categorical_fast, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::sandwich_categorical_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::multiply_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::subset_categorical_complex, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::get_col_included, m)?)?;
    
    // Split matrix operations
    m.add_function(wrap_pyfunction!(split::is_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(split::sandwich_cat_cat, m)?)?;
    m.add_function(wrap_pyfunction!(split::sandwich_cat_dense, m)?)?;
    m.add_function(wrap_pyfunction!(split::split_col_subsets, m)?)?;
    
    Ok(())
}
