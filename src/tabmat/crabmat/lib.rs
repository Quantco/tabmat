//! The Rust extension module of tabmat, `tabmat.crabmat`.
//!
//! All Rust kernels are built into this one shared library so that they share
//! a single copy of rayon (and thus one thread pool, sized once at import) and
//! of PyO3. The kernels themselves are plain Rust in the `kernels` directory,
//! with no PyO3 types; `boundary.rs` holds what the bindings share (dtype
//! dispatch, argument coercion, error conversion), and each kernel family has
//! a binding module of its own (so far `categorical` and `dense`) that is a submodule of
//! this one, `tabmat.crabmat.categorical` and so on, which the Python layer
//! imports the kernels from.

use pyo3::prelude::*;

mod boundary;
mod categorical;
mod dense;
mod kernels;

/// Creates the submodule `name` with the docstring `doc`, lets `register`
/// fill it, and makes it importable as `tabmat.crabmat.<name>` (a submodule
/// created from Rust is only an attribute of its parent until it is entered
/// into `sys.modules`).
fn add_submodule(
    parent: &Bound<'_, PyModule>,
    name: &str,
    doc: &str,
    register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = parent.py();
    let qualname = format!("tabmat.crabmat.{name}");
    let module = PyModule::new(py, &qualname)?;
    module.setattr("__doc__", doc)?;
    register(&module)?;
    parent.add(name, &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(qualname, &module)?;
    Ok(())
}

/// The compiled kernels of tabmat that are written in Rust: one submodule
/// per matrix type. These functions are not part of the public API.
#[pymodule(gil_used = false)]
fn crabmat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    kernels::init_thread_pool();
    add_submodule(
        m,
        "categorical",
        "Kernels for :class:`tabmat.CategoricalMatrix` (src/tabmat/crabmat/categorical.rs).",
        categorical::register,
    )?;
    add_submodule(
        m,
        "dense",
        "Kernels for :class:`tabmat.DenseMatrix` (src/tabmat/crabmat/dense.rs).",
        dense::register,
    )?;
    Ok(())
}
