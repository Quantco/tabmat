//! Python bindings of the categorical kernels, exposed as the submodule
//! `tabmat.ext._rust.categorical` and re-exported by `tabmat.ext.categorical`.
//! Every function keeps the signature of the Cython function it replaced, so
//! the Python side in `tabmat/categorical_matrix.py` is unchanged; the
//! kernels themselves are in `kernels/categorical.rs`.

use numpy::{Complex32, Complex64, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::boundary::{for_float, ro_slice, rw_array, with_dtype, with_rw_slice, Arg};
use crate::kernels::categorical as kernels;

type PyArrayI32<'py> = Bound<'py, PyArray1<i32>>;

/// `drop_first` as the shift to subtract from the indices (`bint` in Cython).
fn drop_shift(drop_first: &Bound<'_, PyAny>) -> PyResult<i32> {
    Ok(drop_first.is_truthy()? as i32)
}

/// Shared implementation of the two `transpose_matvec` functions.
fn transpose_matvec<'py, const DROP: bool>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    other: &Bound<'py, PyAny>,
    rows: Option<Arg<'py, i32>>,
    cols: Option<Arg<'py, i32>>,
    out: &Bound<'py, PyAny>,
    drop_first: i32,
) -> PyResult<()> {
    let indices = ro_slice(&indices);
    let rows = rows.as_ref().map(|a| ro_slice(a));
    let cols = cols.as_ref().map(|a| ro_slice(a));
    for_float!(other, "other", F => {
        let other = other.extract::<PyReadonlyArray1<F>>()?;
        let other = ro_slice(&other);
        let mut out = rw_array::<F>(out, "out")?;
        with_rw_slice(&mut out, |out| {
            py.detach(|| {
                kernels::transpose_matvec::<F, DROP>(
                    &indices,
                    &other,
                    out,
                    rows.as_deref(),
                    cols.as_deref(),
                    drop_first,
                )
            })
        })?;
        Ok(())
    })
}

/// `out += cat[rows, cols].T @ other[rows]` for a matrix without dropped or
/// missing categories. `dtype` is unused and kept for signature compatibility.
#[pyfunction]
#[pyo3(signature = (indices, other, n_cols, dtype, rows, cols, out))]
#[allow(clippy::too_many_arguments)]
fn transpose_matvec_fast<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_cols: usize,
    dtype: &Bound<'py, PyAny>,
    rows: Option<Arg<'py, i32>>,
    cols: Option<Arg<'py, i32>>,
    out: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let _ = (n_cols, dtype);
    transpose_matvec::<false>(py, indices, other, rows, cols, out, 0)
}

/// Like `transpose_matvec_fast`, but the column index is shifted by
/// `drop_first` and negative (dropped or missing) indices are skipped.
#[pyfunction]
#[pyo3(signature = (indices, other, n_cols, dtype, rows, cols, out, drop_first))]
#[allow(clippy::too_many_arguments)]
fn transpose_matvec_complex<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_cols: usize,
    dtype: &Bound<'py, PyAny>,
    rows: Option<Arg<'py, i32>>,
    cols: Option<Arg<'py, i32>>,
    out: &Bound<'py, PyAny>,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let _ = (n_cols, dtype);
    transpose_matvec::<true>(py, indices, other, rows, cols, out, drop_shift(drop_first)?)
}

/// Indicator array of the columns listed in `cols`.
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

/// Shared implementation of the two `matvec` functions.
fn matvec<'py, const DROP: bool>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_rows: usize,
    cols: Option<Arg<'py, i32>>,
    out: &Bound<'py, PyAny>,
    drop_first: i32,
) -> PyResult<()> {
    let indices = ro_slice(&indices);
    let cols = cols.as_ref().map(|a| ro_slice(a));
    let indices = indices
        .get(..n_rows)
        .ok_or_else(|| PyValueError::new_err("n_rows exceeds the length of indices"))?;
    for_float!(other, "other", F => {
        let other = other.extract::<PyReadonlyArray1<F>>()?;
        let other = ro_slice(&other);
        let mut out = rw_array::<F>(out, "out_vec")?;
        with_rw_slice(&mut out, |out| {
            let out = out
                .get_mut(..n_rows)
                .ok_or_else(|| PyValueError::new_err("n_rows exceeds the length of out_vec"))?;
            py.detach(|| kernels::matvec::<F, DROP>(indices, &other, out, cols.as_deref(), drop_first))?;
            Ok(())
        })
    })
}

/// `out_vec += cat[:, cols] @ other` for a matrix without dropped or missing
/// categories: with one-hot data this is `other[indices]`.
#[pyfunction]
#[pyo3(signature = (indices, other, n_rows, cols, n_cols, out_vec))]
fn matvec_fast<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_rows: usize,
    cols: Option<Arg<'py, i32>>,
    n_cols: usize,
    out_vec: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let _ = n_cols;
    matvec::<false>(py, indices, other, n_rows, cols, out_vec, 0)
}

/// Like `matvec_fast`, but the column index is shifted by `drop_first` and
/// negative (dropped or missing) indices are skipped.
#[pyfunction]
#[pyo3(signature = (indices, other, n_rows, cols, n_cols, out_vec, drop_first))]
#[allow(clippy::too_many_arguments)]
fn matvec_complex<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_rows: usize,
    cols: Option<Arg<'py, i32>>,
    n_cols: usize,
    out_vec: &Bound<'py, PyAny>,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let _ = n_cols;
    matvec::<true>(
        py,
        indices,
        other,
        n_rows,
        cols,
        out_vec,
        drop_shift(drop_first)?,
    )
}

/// Shared implementation of the two `sandwich_categorical` functions: the
/// diagonal of the sandwich product restricted to `rows`, as a new array of
/// the dtype of `d`.
fn sandwich<'py, const DROP: bool>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    d: &Bound<'py, PyAny>,
    rows: Arg<'py, i32>,
    n_cols: usize,
    drop_first: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let indices = ro_slice(&indices);
    let rows = ro_slice(&rows);
    for_float!(d, "d", F => {
        let d = d.extract::<PyReadonlyArray1<F>>()?;
        let d = ro_slice(&d);
        let mut res = vec![F::default(); n_cols];
        py.detach(|| kernels::sandwich::<F, DROP>(&indices, &d, &rows, &mut res, drop_first))?;
        Ok(res.into_pyarray(py).into_any())
    })
}

/// Diagonal of `cat[rows].T @ diag(d[rows]) @ cat[rows]` for a matrix without
/// dropped or missing categories. `dtype` is unused and kept for signature
/// compatibility; the result has the dtype of `d`.
#[pyfunction]
#[pyo3(signature = (indices, d, rows, dtype, n_cols))]
fn sandwich_categorical_fast<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    d: &Bound<'py, PyAny>,
    rows: Arg<'py, i32>,
    dtype: &Bound<'py, PyAny>,
    n_cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = dtype;
    sandwich::<false>(py, indices, d, rows, n_cols, 0)
}

/// Like `sandwich_categorical_fast`, but the column index is shifted by
/// `drop_first` and negative (dropped or missing) indices are skipped.
#[pyfunction]
#[pyo3(signature = (indices, d, rows, dtype, n_cols, drop_first))]
fn sandwich_categorical_complex<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    d: &Bound<'py, PyAny>,
    rows: Arg<'py, i32>,
    dtype: &Bound<'py, PyAny>,
    n_cols: usize,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = dtype;
    sandwich::<true>(py, indices, d, rows, n_cols, drop_shift(drop_first)?)
}

/// CSR `(data, indices, indptr)` of `diag(d) @ cat`, with the first
/// `drop_first` categories dropped. `ncols` and `dtype` are unused and kept for
/// signature compatibility; `data` has the dtype of `d`.
#[pyfunction]
#[pyo3(signature = (indices, d, ncols, dtype, drop_first))]
fn multiply_complex<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    d: &Bound<'py, PyAny>,
    ncols: usize,
    dtype: &Bound<'py, PyAny>,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, PyArrayI32<'py>, PyArrayI32<'py>)> {
    let _ = (ncols, dtype);
    let drop_first = drop_shift(drop_first)?;
    let indices = ro_slice(&indices);
    let (new_indices, indptr) = kernels::csr_structure(&indices, drop_first);
    // The dtypes of Cython's `numeric` fused type plus `long long`.
    let data = with_dtype!(d, "d", T in [f64, f32, i64, i32, i16, Complex64, Complex32] => {
        let d = d.extract::<PyReadonlyArray1<T>>()?;
        Ok(kernels::kept_data(&indices, &ro_slice(&d), drop_first).into_pyarray(py).into_any())
    })?;
    Ok((data, new_indices.into_pyarray(py), indptr.into_pyarray(py)))
}

/// Number of nonzeros and CSR `indices` and `indptr` of the matrix with the
/// first `drop_first` categories dropped and missing values skipped. `ncols`
/// is unused and kept for signature compatibility.
#[pyfunction]
#[pyo3(signature = (indices, ncols, drop_first))]
fn subset_categorical_complex<'py>(
    py: Python<'py>,
    indices: Arg<'py, i32>,
    ncols: usize,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<(usize, PyArrayI32<'py>, PyArrayI32<'py>)> {
    let _ = ncols;
    let (new_indices, indptr) =
        kernels::csr_structure(&ro_slice(&indices), drop_shift(drop_first)?);
    Ok((
        new_indices.len(),
        new_indices.into_pyarray(py),
        indptr.into_pyarray(py),
    ))
}

/// Adds the functions of this module to the submodule
/// `tabmat.ext._rust.categorical` (see `lib.rs`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transpose_matvec_fast, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_matvec_complex, m)?)?;
    m.add_function(wrap_pyfunction!(get_col_included, m)?)?;
    m.add_function(wrap_pyfunction!(matvec_fast, m)?)?;
    m.add_function(wrap_pyfunction!(matvec_complex, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_categorical_fast, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_categorical_complex, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_complex, m)?)?;
    m.add_function(wrap_pyfunction!(subset_categorical_complex, m)?)?;
    Ok(())
}
