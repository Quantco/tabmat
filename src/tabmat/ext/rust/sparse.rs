//! Python bindings of the sparse kernels, exposed as the submodule
//! `tabmat.ext._rust.sparse` and re-exported by `tabmat.ext.sparse`; the
//! kernels themselves are in `kernels/sparse.rs`.
//!
//! The sparse matrices arrive as scipy `csr_matrix`/`csc_matrix` objects
//! (or anything with `data`, `indices`, `indptr` and `shape` attributes).
//! The floating point type (`float32` or `float64`) and the index type
//! (`int32` or `int64`) of a call are those of the matrix, as with Cython's
//! fused types; the vectors and index subsets passed alongside are
//! converted to them if necessary ([`Arg`]), a dense `B` must match.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::boundary::{
    extract_center, for_float, for_index, ro_slice, rw_array, with_rw_slice, Arg, PyFloat,
};
use crate::kernels::sparse::{
    self as kernels, included_mask, position_map, Compressed, FormatError,
};
use crate::kernels::{check_indices, Centered, Index};

/// The arrays of a scipy compressed sparse matrix, borrowed from Python.
struct SparseArrays<'py, I: Index + numpy::Element, F: PyFloat> {
    data: PyReadonlyArray1<'py, F>,
    indices: PyReadonlyArray1<'py, I>,
    indptr: PyReadonlyArray1<'py, I>,
    shape: (usize, usize),
}

impl<'py, I: Index + numpy::Element, F: PyFloat> SparseArrays<'py, I, F> {
    /// Reads `data`, `indices`, `indptr` and `shape` of `obj`, which must
    /// have the dtypes of the call.
    fn extract(obj: &Bound<'py, PyAny>, what: &str) -> PyResult<Self> {
        let attr = |name: &str| obj.getattr(name);
        let data = attr("data")?.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "{what}.data must be a 1d array of the floating point dtype of the call"
            ))
        })?;
        let index_err = |name: &str| {
            PyTypeError::new_err(format!(
                "{what}.{name} must be a 1d array of the index dtype of the call"
            ))
        };
        let indices = attr("indices")?
            .extract()
            .map_err(|_| index_err("indices"))?;
        let indptr = attr("indptr")?.extract().map_err(|_| index_err("indptr"))?;
        let shape = attr("shape")?.extract::<(usize, usize)>()?;
        Ok(Self {
            data,
            indices,
            indptr,
            shape,
        })
    }

    /// The matrix as slices (copies of them if they are strided), with its
    /// structure checked for `n_major` compressed rows or columns.
    fn view(&self, n_major: usize, what: &str) -> PyResult<Owned<'_, I, F>> {
        let owned = Owned {
            data: ro_slice(&self.data),
            indices: ro_slice(&self.indices),
            indptr: ro_slice(&self.indptr),
            shape: self.shape,
        };
        owned.view().check(n_major).map_err(|e| match e {
            FormatError::IndptrLength { expected } => PyValueError::new_err(format!(
                "{what}.indptr has {} entries, expected {expected}",
                owned.indptr.len()
            )),
            FormatError::IndptrEntries => PyValueError::new_err(format!(
                "{what}.indptr must be non-decreasing from 0, with data and indices holding its last entry's worth of values"
            )),
        })?;
        Ok(owned)
    }
}

/// The (possibly copied) arrays behind a [`Compressed`] view.
struct Owned<'a, I: Index, F: Clone> {
    data: std::borrow::Cow<'a, [F]>,
    indices: std::borrow::Cow<'a, [I]>,
    indptr: std::borrow::Cow<'a, [I]>,
    shape: (usize, usize),
}

impl<I: Index, F: Clone> Owned<'_, I, F> {
    fn view(&self) -> Compressed<'_, I, F> {
        Compressed {
            data: &self.data,
            indices: &self.indices,
            indptr: &self.indptr,
            shape: self.shape,
        }
    }
}

/// The index array that determines the index dtype of a call about `obj`.
fn indices_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    obj.getattr("indices")
}

/// The data array that determines the floating point dtype of a call about
/// `obj`.
fn data_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    obj.getattr("data")
}

// ---------------------------------------------------------------------------
// Sandwich products
// ---------------------------------------------------------------------------

/// `A[rows, cols].T @ diag(d[rows]) @ A[rows, cols]` as a new square array
/// of the dtype of `A`, for `A` in CSC and `AT` in CSR format.
#[pyfunction]
#[allow(non_snake_case)]
fn sparse_sandwich<'py>(
    py: Python<'py>,
    A: &Bound<'py, PyAny>,
    AT: &Bound<'py, PyAny>,
    d: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    cols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    for_float!(&data_of(A)?, "A.data", F => for_index!(&indices_of(A)?, "A.indices", I => {
        let d = d.extract::<Arg<'py, F>>()?;
        let rows = rows.extract::<Arg<'py, I>>()?;
        let cols = cols.extract::<Arg<'py, I>>()?;
        let a = SparseArrays::<I, F>::extract(A, "A")?;
        let at = SparseArrays::<I, F>::extract(AT, "AT")?;
        let (n, n_cols) = a.shape;
        let a = a.view(n_cols, "A")?;
        let at = at.view(n, "AT")?;
        let (a, at) = (a.view(), at.view());
        let d = ro_slice(&d);
        let rows = ro_slice(&rows);
        let cols = ro_slice(&cols);
        let m = cols.len();
        let mut out = vec![F::default(); m * m];
        if m > 0 && !rows.is_empty() {
            if at.shape != a.shape {
                return Err(PyValueError::new_err("A and AT must have the same shape"));
            }
            if d.len() < n {
                return Err(PyValueError::new_err("d must have one entry per row of A"));
            }
            check_indices(&a.indices[..a.nnz()], n, "A.indices")?;
            check_indices(&at.indices[..at.nnz()], n_cols, "AT.indices")?;
            check_indices(&rows, n, "row")?;
            check_indices(&cols, n_cols, "column")?;
            let row_included = included_mask(&rows, n);
            let col_map = position_map(&cols, n_cols);
            py.detach(|| kernels::sandwich(a, at, &d, &row_included, &cols, &col_map, &mut out));
        }
        let out = Array2::from_shape_vec((m, m), out).expect("shape matches length");
        Ok(out.into_pyarray(py).into_any())
    }))
}

/// `A[rows, A_cols].T @ diag(d[rows]) @ (B - B_center)[rows, B_cols]` as a
/// new `(len(A_cols), len(B_cols))` array of the dtype of `A`, for `A` in
/// CSR format and `B` a dense array of the same dtype in any layout.
/// `B_center` is an optional vector with one entry per column of `B`
/// (`None` means no centering).
#[pyfunction]
#[pyo3(signature = (A, B, d, rows, A_cols, B_cols, B_center = None))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn csr_dense_sandwich<'py>(
    py: Python<'py>,
    A: &Bound<'py, PyAny>,
    B: &Bound<'py, PyAny>,
    d: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    A_cols: &Bound<'py, PyAny>,
    B_cols: &Bound<'py, PyAny>,
    B_center: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    for_float!(&data_of(A)?, "A.data", F => for_index!(&indices_of(A)?, "A.indices", I => {
        let d = d.extract::<Arg<'py, F>>()?;
        let rows = rows.extract::<Arg<'py, I>>()?;
        let a_cols = A_cols.extract::<Arg<'py, I>>()?;
        let b_cols = B_cols.extract::<Arg<'py, I>>()?;
        let a = SparseArrays::<I, F>::extract(A, "A")?;
        let b = B
            .extract::<PyReadonlyArray2<F>>()
            .map_err(|_| PyTypeError::new_err("B must be a 2d array of the dtype of A"))?;
        let center = extract_center::<F>(B_center, b.shape()[1], "B_center")?;
        let (n, m) = a.shape;
        let a = a.view(n, "A")?;
        let a = a.view();
        let d = ro_slice(&d);
        let rows = ro_slice(&rows);
        let a_cols = ro_slice(&a_cols);
        let b_cols = ro_slice(&b_cols);
        let (nac, nbc) = (a_cols.len(), b_cols.len());
        let mut out = vec![F::default(); nac * nbc];
        if !rows.is_empty() && nac > 0 && nbc > 0 && a.nnz() > 0 {
            let n_rows = n.min(b.shape()[0]);
            if d.len() < n_rows {
                return Err(PyValueError::new_err("d must have one entry per row of A"));
            }
            check_indices(&rows, n_rows, "row")?;
            check_indices(&a_cols, m, "A column")?;
            check_indices(&b_cols, b.shape()[1], "B column")?;
            check_indices(&a.indices[..a.nnz()], m, "A.indices")?;
            let acol_map = position_map(&a_cols, m);
            let center = center.as_ref().map(|c| ro_slice(c));
            let b = Centered { x: b.as_array(), center: center.as_deref() };
            py.detach(|| {
                kernels::csr_dense_sandwich(a, b, &d, &rows, &acol_map, &b_cols, nac, &mut out)
            });
        }
        let out = Array2::from_shape_vec((nac, nbc), out).expect("shape matches length");
        Ok(out.into_pyarray(py).into_any())
    }))
}

// ---------------------------------------------------------------------------
// Matrix-vector products
// ---------------------------------------------------------------------------

/// Which product the matrix-vector functions compute.
#[derive(Clone, Copy)]
enum MatVec {
    /// `X @ v` for `X` in CSR format: one output per row.
    Csr,
    /// `XT.T @ v` for `XT` in CSC format: one output per column.
    Csc,
}

/// Shared implementation of `csr_matvec_unrestricted` and
/// `csc_rmatvec_unrestricted`: accumulates into `out` if it is given (and
/// returns it), otherwise into a new zeroed array.
fn matvec_unrestricted<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
    which: MatVec,
) -> PyResult<Bound<'py, PyAny>> {
    for_float!(&data_of(x)?, "X.data", F => for_index!(&indices_of(x)?, "X.indices", I => {
        let v = v.extract::<Arg<'py, F>>()?;
        let x = SparseArrays::<I, F>::extract(x, "X")?;
        let (n_rows, n_cols) = x.shape;
        let (n_major, n_minor) = match which {
            MatVec::Csr => (n_rows, n_cols),
            MatVec::Csc => (n_cols, n_rows),
        };
        let x = x.view(n_major, "X")?;
        let x = x.view();
        let v = ro_slice(&v);
        if v.len() < n_minor {
            return Err(PyValueError::new_err(format!(
                "v must have at least {n_minor} entries"
            )));
        }
        let kernel = |out: &mut [F]| -> PyResult<()> {
            if out.len() > n_major {
                return Err(PyValueError::new_err(format!(
                    "out must have at most {n_major} entries"
                )));
            }
            py.detach(|| kernels::matvec::<I, F, false>(x, &v, &[], &[], out))?;
            Ok(())
        };
        match out {
            None => {
                let mut out = vec![F::default(); n_major];
                kernel(&mut out)?;
                Ok(out.into_pyarray(py).into_any())
            }
            Some(out) => {
                let mut out_arr = rw_array::<F>(out, "out")?;
                with_rw_slice(&mut out_arr, kernel)?;
                Ok(out.clone())
            }
        }
    }))
}

/// `out += X @ v` for `X` in CSR format, into `out` if it is given (it is
/// returned) and a new zeroed array of the dtype of `X` otherwise.
#[pyfunction]
#[pyo3(signature = (X, v, out = None))]
#[allow(non_snake_case)]
fn csr_matvec_unrestricted<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    matvec_unrestricted(py, X, v, out, MatVec::Csr)
}

/// `out += XT.T @ v` for `XT` in CSC format, into `out` if it is given (it
/// is returned) and a new zeroed array of the dtype of `XT` otherwise.
#[pyfunction]
#[pyo3(signature = (XT, v, out = None))]
#[allow(non_snake_case)]
fn csc_rmatvec_unrestricted<'py>(
    py: Python<'py>,
    XT: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    matvec_unrestricted(py, XT, v, out, MatVec::Csc)
}

/// Shared implementation of `csr_matvec` and `csc_rmatvec`: a new array
/// with one entry per selected row (CSR) or column (CSC).
fn matvec_restricted<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    cols: &Bound<'py, PyAny>,
    which: MatVec,
) -> PyResult<Bound<'py, PyAny>> {
    for_float!(&data_of(x)?, "X.data", F => for_index!(&indices_of(x)?, "X.indices", I => {
        let v = v.extract::<Arg<'py, F>>()?;
        let rows = rows.extract::<Arg<'py, I>>()?;
        let cols = cols.extract::<Arg<'py, I>>()?;
        let x = SparseArrays::<I, F>::extract(x, "X")?;
        let (n_rows, n_cols) = x.shape;
        let v = ro_slice(&v);
        let rows = ro_slice(&rows);
        let cols = ro_slice(&cols);
        check_indices(&rows, n_rows, "row")?;
        check_indices(&cols, n_cols, "column")?;
        let out = match which {
            MatVec::Csr => {
                let x = x.view(n_rows, "X")?;
                let col_included = included_mask(&cols, n_cols);
                let mut out = vec![F::default(); rows.len()];
                py.detach(|| kernels::matvec::<I, F, true>(x.view(), &v, &rows, &col_included, &mut out))?;
                out
            }
            MatVec::Csc => {
                let x = x.view(n_cols, "X")?;
                let row_included = included_mask(&rows, n_rows);
                let mut out = vec![F::default(); cols.len()];
                py.detach(|| kernels::matvec::<I, F, true>(x.view(), &v, &cols, &row_included, &mut out))?;
                out
            }
        };
        Ok(out.into_pyarray(py).into_any())
    }))
}

/// `X[rows, cols] @ v[cols]` for `X` in CSR format, as a new array of
/// `len(rows)` entries.
#[pyfunction]
#[allow(non_snake_case)]
fn csr_matvec<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    cols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    matvec_restricted(py, X, v, rows, cols, MatVec::Csr)
}

/// `XT[rows, cols].T @ v[rows]` for `XT` in CSC format, as a new array of
/// `len(cols)` entries.
#[pyfunction]
#[allow(non_snake_case)]
fn csc_rmatvec<'py>(
    py: Python<'py>,
    XT: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    cols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    matvec_restricted(py, XT, v, rows, cols, MatVec::Csc)
}

// ---------------------------------------------------------------------------
// transpose_square_dot_weights
// ---------------------------------------------------------------------------

/// `out[j] = sum_k weights[indices[k]] * data[k]**2` over the entries `k`
/// of column `j` of a CSC matrix given by its arrays, as a new array of
/// `len(indptr) - 1` entries of the dtype of `data`.
#[pyfunction]
fn transpose_square_dot_weights<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    for_float!(data, "data", F => for_index!(indices, "indices", I => {
        let data = data.extract::<PyReadonlyArray1<F>>()?;
        let indices = indices.extract::<PyReadonlyArray1<I>>()?;
        let indptr = indptr.extract::<Arg<'py, I>>()?;
        let weights = weights.extract::<Arg<'py, F>>()?;
        let weights = ro_slice(&weights);
        let n_cols = indptr.len().saturating_sub(1);
        let x = Owned {
            data: ro_slice(&data),
            indices: ro_slice(&indices),
            indptr: ro_slice(&indptr),
            shape: (weights.len(), n_cols),
        };
        let x = x.view();
        x.check(n_cols).map_err(|_| PyValueError::new_err("indptr, indices and data do not form a valid CSC matrix"))?;
        let mut out = vec![F::default(); n_cols];
        py.detach(|| kernels::transpose_square_dot_weights(x, &weights, &mut out))?;
        Ok(out.into_pyarray(py).into_any())
    }))
}

/// Adds the functions of this module to the submodule
/// `tabmat.ext._rust.sparse` (see `lib.rs`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(csr_dense_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(csr_matvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(csr_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(csc_rmatvec_unrestricted, m)?)?;
    m.add_function(wrap_pyfunction!(csc_rmatvec, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_square_dot_weights, m)?)?;
    Ok(())
}
