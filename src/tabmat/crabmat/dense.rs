//! Python bindings of the dense kernels (so far the sandwich product),
//! exposed as the submodule `tabmat.crabmat.dense`; the kernels themselves
//! are in `kernels/dense.rs`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::boundary::{extract_center, for_float, ro_slice, Arg};
use crate::kernels::dense as kernels;
use crate::kernels::{check_indices, Centered};

/// `(X[rows, cols] - center[cols]).T @ diag(d[rows]) @ (X[rows, cols] - center[cols])`
/// as a new `(len(cols), len(cols))` array of the dtype of `X`.
///
/// `center` is an optional length-`X.shape[1]` vector (indexed by the
/// original column index, like `X`); `None` means no centering.
#[pyfunction]
#[pyo3(signature = (X, d, rows, cols, thresh1d = 32, kratio = 16, innerblock = 128, center = None))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn dense_sandwich<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    d: &Bound<'py, PyAny>,
    rows: Arg<'py, i32>,
    cols: Arg<'py, i32>,
    thresh1d: usize,
    kratio: usize,
    innerblock: usize,
    center: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = ro_slice(&rows);
    let cols = ro_slice(&cols);
    let (in_n, out_m) = (rows.len(), cols.len());
    for_float!(X, "X", F => {
        let x = X.extract::<PyReadonlyArray2<F>>()?;
        let (n, m) = (x.shape()[0], x.shape()[1]);
        let center = extract_center::<F>(center, m, "center")?;
        let mut out = vec![F::default(); out_m * out_m];
        if in_n > 0 && out_m > 0 {
            if thresh1d == 0 || kratio == 0 || innerblock == 0 {
                return Err(PyValueError::new_err(
                    "thresh1d, kratio and innerblock must be positive",
                ));
            }
            let d = d.extract::<Arg<'py, F>>()?;
            let d = ro_slice(&d);
            check_indices(&rows, n, "row")?;
            check_indices(&cols, m, "column")?;
            if d.len() < n {
                return Err(PyValueError::new_err("d must have one entry per row of X"));
            }
            let center = center.as_ref().map(|c| ro_slice(c));
            let x = Centered { x: x.as_array(), center: center.as_deref() };
            py.detach(|| {
                kernels::sandwich(x, &d, &rows, &cols, &mut out, thresh1d, kratio, innerblock)
            });
        }
        let out = Array2::from_shape_vec((out_m, out_m), out).expect("shape matches length");
        Ok(out.into_pyarray(py).into_any())
    })
}

/// Adds the functions of this module to the submodule
/// `tabmat.crabmat.dense` (see `lib.rs`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dense_sandwich, m)?)?;
    Ok(())
}
