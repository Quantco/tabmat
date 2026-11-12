//! Python bindings of the dense kernels, exposed as the submodule
//! `tabmat.ext.crabmat.dense` and re-exported by `tabmat.ext.dense`; the
//! kernels themselves are in `kernels/dense.rs`.

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

/// Which matrix-vector product [`matvec_like`] computes.
#[derive(Clone, Copy)]
enum MatVec {
    /// `(X - center)[rows, cols] @ v[cols]`
    Plain,
    /// `(X - center)[rows, cols].T @ v[rows]`
    Transposed,
}

/// Shared implementation of `dense_matvec` and `dense_rmatvec`.
fn matvec_like<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    rows: Option<Arg<'py, i32>>,
    cols: Option<Arg<'py, i32>>,
    center: Option<&Bound<'py, PyAny>>,
    which: MatVec,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = rows.as_ref().map(|r| ro_slice(r));
    let cols = cols.as_ref().map(|c| ro_slice(c));
    for_float!(x, "X", F => {
        let x = x.extract::<PyReadonlyArray2<F>>()?;
        let (n, m) = (x.shape()[0], x.shape()[1]);
        let (n_rows, n_cols) = (rows.as_ref().map_or(n, |r| r.len()), cols.as_ref().map_or(m, |c| c.len()));
        let (out_len, v_len) = match which {
            MatVec::Plain => (n_rows, m),
            MatVec::Transposed => (n_cols, n),
        };
        let center = extract_center::<F>(center, m, "center")?;
        let mut out = vec![F::default(); out_len];
        if n_rows > 0 && n_cols > 0 {
            let v = v.extract::<Arg<'py, F>>()?;
            let v = ro_slice(&v);
            if let Some(rows) = &rows {
                check_indices(rows, n, "row")?;
            }
            if let Some(cols) = &cols {
                check_indices(cols, m, "column")?;
            }
            if v.len() < v_len {
                return Err(PyValueError::new_err(format!(
                    "v must have at least {v_len} entries"
                )));
            }
            let center = center.as_ref().map(|c| ro_slice(c));
            let x = Centered { x: x.as_array(), center: center.as_deref() };
            py.detach(|| match which {
                MatVec::Plain => kernels::matvec(x, &v, rows.as_deref(), cols.as_deref(), &mut out),
                MatVec::Transposed => kernels::rmatvec(x, &v, rows.as_deref(), cols.as_deref(), &mut out),
            });
        }
        Ok(out.into_pyarray(py).into_any())
    })
}

/// `(X[rows, cols] - center[cols]).T @ v[rows]` as a new array of
/// `len(cols)` entries.
///
/// `rows=None` and `cols=None` select all rows and columns, which lets the
/// kernel vectorise along them; `center=None` means no centering.
#[pyfunction]
#[pyo3(signature = (X, v, rows = None, cols = None, center = None))]
#[allow(non_snake_case)]
fn dense_rmatvec<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    rows: Option<Arg<'py, i32>>,
    cols: Option<Arg<'py, i32>>,
    center: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    matvec_like(py, X, v, rows, cols, center, MatVec::Transposed)
}

/// `(X[rows, cols] - center[cols]) @ v[cols]` as a new array of `len(rows)`
/// entries, with the same conventions as `dense_rmatvec`.
#[pyfunction]
#[pyo3(signature = (X, v, rows = None, cols = None, center = None))]
#[allow(non_snake_case)]
fn dense_matvec<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    rows: Option<Arg<'py, i32>>,
    cols: Option<Arg<'py, i32>>,
    center: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    matvec_like(py, X, v, rows, cols, center, MatVec::Plain)
}

/// `sum_i weights[i] * (X[i, :] - shift)**2` over the first `len(weights)`
/// rows of `X`, as a new array of `X.shape[1]` entries.
#[pyfunction]
#[allow(non_snake_case)]
fn transpose_square_dot_weights<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    shift: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    for_float!(X, "X", F => {
        let x = X.extract::<PyReadonlyArray2<F>>()?;
        let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);
        let weights = weights.extract::<Arg<'py, F>>()?;
        let weights = ro_slice(&weights);
        let shift = shift.extract::<Arg<'py, F>>()?;
        let shift = ro_slice(&shift);
        if weights.len() > n_rows || shift.len() < n_cols {
            return Err(PyValueError::new_err(
                "weights must have at most one entry per row of X and shift one per column",
            ));
        }
        let mut out = vec![F::default(); n_cols];
        if n_cols > 0 {
            let x = Centered { x: x.as_array(), center: Some(&shift[..n_cols]) };
            py.detach(|| kernels::transpose_square_dot_weights(x, &weights, &mut out));
        }
        Ok(out.into_pyarray(py).into_any())
    })
}

/// Adds the functions of this module to the submodule
/// `tabmat.ext.crabmat.dense` (see `lib.rs`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dense_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(dense_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(dense_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_square_dot_weights, m)?)?;
    Ok(())
}
