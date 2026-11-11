//! The Rust extension module of tabmat, `tabmat.ext._rust`.
//!
//! All Rust kernels are built into this one shared library so that they share
//! a single copy of rayon (and thus one thread pool, sized once at import) and
//! of PyO3. Each kernel family is a submodule of its own (`categorical` for
//! now), and a thin Python module per family (`tabmat.ext.categorical`)
//! re-exports its functions under the names the rest of tabmat has always
//! imported them by.

use pyo3::prelude::*;

mod categorical;
mod common;

/// Creates the submodule `name`, lets `register` fill it, and makes it
/// importable as `tabmat.ext._rust.<name>` (a submodule created from Rust is
/// only an attribute of its parent until it is entered into `sys.modules`).
fn add_submodule(
    parent: &Bound<'_, PyModule>,
    name: &str,
    register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = parent.py();
    let qualname = format!("tabmat.ext._rust.{name}");
    let module = PyModule::new(py, &qualname)?;
    register(&module)?;
    parent.add(name, &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(qualname, &module)?;
    Ok(())
}

#[pymodule(gil_used = false)]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    common::init_thread_pool();
    add_submodule(m, "categorical", categorical::register)?;
    Ok(())
}
