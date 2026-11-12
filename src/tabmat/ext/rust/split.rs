//! Python bindings of the split kernels (the categorical cross products and
//! the SplitMatrix column bookkeeping), exposed as the submodule
//! `tabmat.ext._rust.split` and re-exported by `tabmat.ext.split`. Every
//! function keeps the signature of the Cython function it replaced
//! (including the `dense_center` argument of #560), so the Python side in
//! `tabmat/categorical_matrix.py` and `tabmat/split_matrix.py` is unchanged;
//! the kernels themselves are in `kernels/split.rs`.

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::boundary::{extract_center, for_float, ro_slice, Arg};
use crate::kernels::split as kernels;
use crate::kernels::{check_indices, Centered};

/// A new 2-d array of the given shape from its row-major data.
fn into_array2<'py, F: numpy::Element>(
    py: Python<'py>,
    shape: (usize, usize),
    data: Vec<F>,
) -> Bound<'py, PyArray2<F>> {
    Array2::from_shape_vec(shape, data)
        .expect("shape matches length")
        .into_pyarray(py)
}

/// A Python object's truth value, as Cython converts a `bint` or `bool`
/// argument (so numpy booleans and integers are accepted as well).
fn truthy(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.is_truthy()
}

/// `cat[rows].T @ diag(d[rows]) @ (mat_j - dense_center)[rows, j_cols]` as a
/// new `(i_ncol, len(j_cols))` array of the dtype of `mat_j`, for a
/// categorical matrix with the row categories `i_indices`. `is_c_contiguous`
/// is unused (`mat_j` is read in place through its strides, whatever its
/// layout) and kept for signature compatibility; `has_missings` and
/// `drop_first` select the complex kernel; `dense_center` is an optional
/// vector with one entry per column of `mat_j` (`None` means no centering).
#[pyfunction]
#[pyo3(signature = (i_indices, i_ncol, d, mat_j, rows, j_cols, is_c_contiguous, has_missings, drop_first, dense_center = None))]
#[allow(clippy::too_many_arguments)]
fn sandwich_cat_dense<'py>(
    py: Python<'py>,
    i_indices: Arg<'py, i32>,
    i_ncol: usize,
    d: &Bound<'py, PyAny>,
    mat_j: &Bound<'py, PyAny>,
    rows: Arg<'py, i32>,
    j_cols: Arg<'py, i32>,
    is_c_contiguous: &Bound<'py, PyAny>,
    has_missings: &Bound<'py, PyAny>,
    drop_first: &Bound<'py, PyAny>,
    dense_center: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = is_c_contiguous;
    let indices = ro_slice(&i_indices);
    let rows = ro_slice(&rows);
    let j_cols = ro_slice(&j_cols);
    let drop_first = truthy(drop_first)?;
    let drop = drop_first || truthy(has_missings)?;
    for_float!(mat_j, "mat_j", F => {
        let mat_j = mat_j.extract::<PyReadonlyArray2<F>>()?;
        let d = d.extract::<Arg<'py, F>>()?;
        let d = ro_slice(&d);
        let center = extract_center::<F>(dense_center, mat_j.shape()[1], "dense_center")?;
        let nj = j_cols.len();
        let mut res = vec![F::default(); i_ncol * nj];
        if !d.is_empty() && !rows.is_empty() && nj > 0 && i_ncol > 0 {
            check_indices(&j_cols, mat_j.shape()[1], "column")?;
            let center = center.as_ref().map(|c| ro_slice(c));
            let mat_j = Centered { x: mat_j.as_array(), center: center.as_deref() };
            py.detach(|| {
                if drop {
                    kernels::sandwich_cat_dense::<F, true>(mat_j, &indices, &d, &rows, &j_cols, drop_first as i32, &mut res)
                } else {
                    kernels::sandwich_cat_dense::<F, false>(mat_j, &indices, &d, &rows, &j_cols, 0, &mut res)
                }
            })?;
        }
        Ok(into_array2(py, (i_ncol, nj), res).into_any())
    })
}

/// `cat_i[rows].T @ diag(d[rows]) @ cat_j[rows]` as a new `(i_ncol, j_ncol)`
/// array of the dtype of `d`, for categorical matrices with the row
/// categories `i_indices` and `j_indices`. `dtype` is unused and kept for
/// signature compatibility; the `drop_first` and `has_missings` flags
/// select the complex kernel.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sandwich_cat_cat<'py>(
    py: Python<'py>,
    i_indices: Arg<'py, i32>,
    j_indices: Arg<'py, i32>,
    i_ncol: usize,
    j_ncol: usize,
    d: &Bound<'py, PyAny>,
    rows: Arg<'py, i32>,
    dtype: &Bound<'py, PyAny>,
    i_drop_first: &Bound<'py, PyAny>,
    j_drop_first: &Bound<'py, PyAny>,
    i_has_missings: &Bound<'py, PyAny>,
    j_has_missings: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = dtype;
    let i_indices = ro_slice(&i_indices);
    let j_indices = ro_slice(&j_indices);
    let rows = ro_slice(&rows);
    let i_drop_first = truthy(i_drop_first)?;
    let j_drop_first = truthy(j_drop_first)?;
    let drop = i_drop_first || j_drop_first || truthy(i_has_missings)? || truthy(j_has_missings)?;
    for_float!(d, "d", F => {
        let d = d.extract::<PyReadonlyArray1<F>>()?;
        let d = ro_slice(&d);
        let mut res = vec![F::default(); i_ncol * j_ncol];
        if !rows.is_empty() && !res.is_empty() {
            py.detach(|| {
                if drop {
                    kernels::sandwich_cat_cat::<F, true>(&i_indices, &j_indices, &d, &rows, j_ncol, i_drop_first as i32, j_drop_first as i32, &mut res)
                } else {
                    kernels::sandwich_cat_cat::<F, false>(&i_indices, &j_indices, &d, &rows, j_ncol, 0, 0, &mut res)
                }
            })?;
        }
        Ok(into_array2(py, (i_ncol, j_ncol), res).into_any())
    })
}

/// `cat_i[rows].T @ diag(d[rows]) @ cat_j[rows]` restricted to the columns
/// `i_cols` and `j_cols`, as a new `(i_ncol, j_ncol)` float64 array. A
/// single-threaded reference implementation kept for the tests; the
/// restriction is applied by masking, so the result keeps the full shape.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _sandwich_cat_cat_limited_rows_cols<'py>(
    py: Python<'py>,
    i_indices: Arg<'py, i32>,
    j_indices: Arg<'py, i32>,
    i_ncol: usize,
    j_ncol: usize,
    d: Arg<'py, f64>,
    rows: Arg<'py, i32>,
    i_cols: Arg<'py, i32>,
    j_cols: Arg<'py, i32>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let i_indices = ro_slice(&i_indices);
    let j_indices = ro_slice(&j_indices);
    let d = ro_slice(&d);
    let rows = ro_slice(&rows);
    let i_cols = ro_slice(&i_cols);
    let j_cols = ro_slice(&j_cols);
    let res = py.detach(|| {
        kernels::sandwich_cat_cat_limited_rows_cols(
            &i_indices, &j_indices, &d, &rows, i_ncol, j_ncol, &i_cols, &j_cols,
        )
    })?;
    Ok(into_array2(py, (i_ncol, j_ncol), res))
}

/// A list of `int32` arrays, one per sub-matrix of a `SplitMatrix`.
type ArrayList<'py> = Vec<Bound<'py, PyArray1<i32>>>;

/// For a `SplitMatrix` `mat` and sorted column subset `cols`, the positions
/// in `cols` of the columns of each sub-matrix (`subset_cols_indices`), the
/// positions of those columns within each sub-matrix (`subset_cols`) and
/// `len(cols)`, such that
/// `mat.indices[i][subset_cols[i]] == cols[subset_cols_indices[i]]`.
#[pyfunction]
#[pyo3(signature = (mat, cols))]
fn split_col_subsets<'py>(
    py: Python<'py>,
    mat: &Bound<'py, PyAny>,
    cols: Arg<'py, i32>,
) -> PyResult<(ArrayList<'py>, ArrayList<'py>, usize)> {
    let indices: Vec<Arg<'py, i64>> = mat
        .getattr("indices")?
        .try_iter()?
        .map(|item| item?.extract())
        .collect::<PyResult<_>>()
        .map_err(|_| PyTypeError::new_err("mat.indices must be a list of 1d integer arrays"))?;
    let indices: Vec<_> = indices.iter().map(|a| ro_slice(a)).collect();
    let indices: Vec<&[i64]> = indices.iter().map(|a| &a[..]).collect();
    let cols = ro_slice(&cols);
    if cols.len() > i32::MAX as usize {
        return Err(PyValueError::new_err("too many columns"));
    }
    let (subset_cols_indices, subset_cols) = kernels::split_col_subsets(&indices, &cols);
    let to_py = |v: Vec<Vec<i32>>| v.into_iter().map(|a| a.into_pyarray(py)).collect();
    Ok((to_py(subset_cols_indices), to_py(subset_cols), cols.len()))
}

/// Whether the integer array `a` is sorted in nondecreasing order.
#[pyfunction]
fn is_sorted(a: Arg<'_, i64>) -> bool {
    kernels::is_sorted(&ro_slice(&a))
}

/// Adds the functions of this module to the submodule
/// `tabmat.ext._rust.split` (see `lib.rs`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sandwich_cat_dense, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich_cat_cat, m)?)?;
    m.add_function(wrap_pyfunction!(_sandwich_cat_cat_limited_rows_cols, m)?)?;
    m.add_function(wrap_pyfunction!(split_col_subsets, m)?)?;
    m.add_function(wrap_pyfunction!(is_sorted, m)?)?;
    Ok(())
}
