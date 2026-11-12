//! Python bindings of the categorical kernels, exposed as the submodule
//! `tabmat.ext.crabmat.categorical` and re-exported by `tabmat.ext.categorical`;
//! the kernels themselves are in `kernels/categorical.rs`.
//!
//! A categorical matrix is passed as its `indices`: one category per row
//! (`-1` for a missing value). Every product takes `drop_first` (the first
//! category is the dropped reference level: the categories are shifted down
//! by one) and `has_missings` (some rows have no category); either selects
//! the kernel that checks for and skips such rows, the plain kernel has no
//! check at all.

use numpy::{Complex32, Complex64, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::boundary::{for_float, ro_slice, rw_array, with_drop, with_dtype, with_rw_slice, Arg};
use crate::kernels::categorical as kernels;

type PyArrayI32<'py> = Bound<'py, PyArray1<i32>>;

/// `out += cat[rows, cols].T @ v[rows]`, in place.
///
/// `indices` are the categories of the rows, `out` has one entry per column
/// of the matrix and the dtype of `v`. `rows` and `cols` restrict the
/// product (`None` for all); the entries of `out` for columns not in `cols`
/// are left as they are.
#[pyfunction]
#[pyo3(signature = (indices, v, out, rows = None, cols = None, drop_first = false, has_missings = false))]
#[allow(clippy::too_many_arguments)]
fn transpose_matvec<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    v: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    rows: Option<Arg<'py, i32>>,
    cols: Option<Arg<'py, i32>>,
    drop_first: bool,
    has_missings: bool,
) -> PyResult<()> {
    let indices = ro_slice(&indices);
    let rows = rows.as_ref().map(|a| ro_slice(a));
    let cols = cols.as_ref().map(|a| ro_slice(a));
    let drop = drop_first || has_missings;
    for_float!(v, "v", F => {
        let v = v.extract::<PyReadonlyArray1<F>>()?;
        let v = ro_slice(&v);
        let mut out = rw_array::<F>(out, "out")?;
        with_rw_slice(&mut out, |out| {
            py.detach(|| {
                let (rows, cols) = (rows.as_deref(), cols.as_deref());
                with_drop!(drop, DROP, drop_first => kernels::transpose_matvec::<F, DROP>(&indices, &v, out, rows, cols, drop_first))
            })
        })?;
        Ok(())
    })
}

/// `out += cat[:, cols] @ v[cols]`, in place: with one-hot data this is
/// `v[indices]`.
///
/// `out` has one entry per row of the matrix and the dtype of `v`; `cols`
/// restricts the product (`None` for all columns).
#[pyfunction]
#[pyo3(signature = (indices, v, out, cols = None, drop_first = false, has_missings = false))]
fn matvec<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    v: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    cols: Option<Arg<'py, i32>>,
    drop_first: bool,
    has_missings: bool,
) -> PyResult<()> {
    let indices = ro_slice(&indices);
    let cols = cols.as_ref().map(|a| ro_slice(a));
    let drop = drop_first || has_missings;
    for_float!(v, "v", F => {
        let v = v.extract::<PyReadonlyArray1<F>>()?;
        let v = ro_slice(&v);
        let mut out = rw_array::<F>(out, "out")?;
        with_rw_slice(&mut out, |out| {
            if out.len() != indices.len() {
                return Err(PyValueError::new_err("out must have one entry per row of the matrix"));
            }
            py.detach(|| {
                with_drop!(drop, DROP, drop_first => kernels::matvec::<F, DROP>(&indices, &v, out, cols.as_deref(), drop_first))
            })?;
            Ok(())
        })
    })
}

/// Diagonal of `cat[rows].T @ diag(d[rows]) @ cat[rows]`: the sum of `d`
/// over the rows of each of the `n_cols` columns, as a new array of the
/// dtype of `d`.
#[pyfunction]
#[pyo3(signature = (indices, d, rows, n_cols, drop_first = false, has_missings = false))]
fn sandwich_categorical<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    d: &Bound<'py, PyAny>,
    rows: Arg<'py, i32>,
    n_cols: usize,
    drop_first: bool,
    has_missings: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let indices = ro_slice(&indices);
    let rows = ro_slice(&rows);
    let drop = drop_first || has_missings;
    for_float!(d, "d", F => {
        let d = d.extract::<PyReadonlyArray1<F>>()?;
        let d = ro_slice(&d);
        let mut res = vec![F::default(); n_cols];
        py.detach(|| {
            with_drop!(drop, DROP, drop_first => kernels::sandwich::<F, DROP>(&indices, &d, &rows, &mut res, drop_first))
        })?;
        Ok(res.into_pyarray(py).into_any())
    })
}

/// Indicator array (`int32`, one entry per column) of the columns listed in
/// `cols`.
#[pyfunction]
fn get_col_included<'py>(
    py: Python<'py>,
    cols: Arg<'py, i32>,
    n_cols: usize,
) -> PyResult<PyArrayI32<'py>> {
    let mask = kernels::col_mask(&ro_slice(&cols), n_cols)?;
    Ok(mask
        .into_iter()
        .map(i32::from)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

/// CSR `(data, indices, indptr)` of `diag(d) @ cat` for a matrix with a
/// dropped reference category or missing values: one entry per row whose
/// (shifted) category is non-negative. `data` has the dtype of `d`, which
/// may be any signed integer, float or complex dtype.
#[pyfunction]
#[pyo3(signature = (indices, d, drop_first = false))]
fn multiply<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    d: &Bound<'py, PyAny>,
    drop_first: bool,
) -> PyResult<(Bound<'py, PyAny>, PyArrayI32<'py>, PyArrayI32<'py>)> {
    let drop_first = drop_first as i32;
    let indices = ro_slice(&indices);
    let (new_indices, indptr) = kernels::csr_structure(&indices, drop_first);
    // The dtypes of Cython's `numeric` fused type plus `long long`.
    let data = with_dtype!(d, "d", T in [f64, f32, i64, i32, i16, Complex64, Complex32] => {
        let d = d.extract::<PyReadonlyArray1<T>>()?;
        Ok(kernels::kept_data(&indices, &ro_slice(&d), drop_first).into_pyarray(py).into_any())
    })?;
    Ok((data, new_indices.into_pyarray(py), indptr.into_pyarray(py)))
}

/// Number of nonzeros and CSR `indices` and `indptr` of a matrix with a
/// dropped reference category or missing values: one entry per row whose
/// (shifted) category is non-negative.
#[pyfunction]
#[pyo3(signature = (indices, drop_first = false))]
fn subset_categorical<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    drop_first: bool,
) -> PyResult<(usize, PyArrayI32<'py>, PyArrayI32<'py>)> {
    let (new_indices, indptr) = kernels::csr_structure(&ro_slice(&indices), drop_first as i32);
    Ok((
        new_indices.len(),
        new_indices.into_pyarray(py),
        indptr.into_pyarray(py),
    ))
}

/// Adds the functions of this module to the submodule
/// `tabmat.ext.crabmat.categorical` (see `lib.rs`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transpose_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(matvec, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_categorical, m)?)?;
    m.add_function(wrap_pyfunction!(get_col_included, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(subset_categorical, m)?)?;
    Ok(())
}
