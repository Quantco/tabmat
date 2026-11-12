//! The Rust extension module of tabmat, `tabmat.ext._rust`.
//!
//! All Rust kernels are built into this one shared library so that they share
//! a single copy of rayon (and thus one thread pool, sized once at import) and
//! of PyO3. The kernels themselves are plain Rust in the `kernels` directory,
//! with no PyO3 types; `boundary.rs` holds what the bindings share (dtype
//! dispatch, argument coercion, error conversion), and each kernel family has
//! a binding module of its own (`categorical`, `dense`, `sparse`, `split`)
//! that is a submodule of this one. A thin Python module per family
//! (`tabmat.ext.categorical`, `tabmat.ext.dense`, `tabmat.ext.sparse`,
//! `tabmat.ext.split`) re-exports its functions under the names the rest of tabmat has always
//! imported them by.

use pyo3::prelude::*;

mod boundary;
mod categorical;
mod dense;
mod kernels;
mod sparse;
mod split;

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
    kernels::init_thread_pool();
    add_submodule(m, "categorical", categorical::register)?;
    add_submodule(m, "dense", dense::register)?;
    add_submodule(m, "sparse", sparse::register)?;
    add_submodule(m, "split", split::register)?;
    Ok(())
}
