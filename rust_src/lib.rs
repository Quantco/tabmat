use pyo3::prelude::*;

mod dense;
mod sparse;
mod categorical;

/// Tabmat Rust extensions
#[pymodule]
fn tabmat_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Dense matrix operations
    m.add_function(wrap_pyfunction!(dense::dense_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(dense::dense_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense::dense_matvec, m)?)?;
    
    // Sparse matrix operations
    m.add_function(wrap_pyfunction!(sparse::sparse_sandwich, m)?)?;
    
    // Categorical matrix operations
    m.add_function(wrap_pyfunction!(categorical::categorical_sandwich, m)?)?;
    
    Ok(())
}
