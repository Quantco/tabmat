//! Kernels for `tabmat.CategoricalMatrix`, exposed as the submodule
//! `tabmat.ext._rust.categorical` and re-exported by `tabmat.ext.categorical`.
//!
//! A categorical matrix is stored as one `int32` index per row, giving the
//! column of that row's single nonzero entry (`-1` for a missing value). Every
//! function here keeps the signature of the Cython function it replaced, so the
//! Python side in `tabmat/categorical_matrix.py` is unchanged.
//!
//! The "fast" and "complex" variants of each operation differ only in the
//! handling of the index: the complex one subtracts `drop_first` and skips
//! rows whose adjusted column index is negative (dropped reference category or
//! missing value). The variants share one kernel, specialised at compile time
//! through the `DROP` const generic so that the fast path has no check at all.
//!
//! Floating point arrays are accepted as either `float32` or `float64`; the
//! dtype is dispatched at runtime, mirroring Cython's `floating` fused type.

use std::ops::Range;

use numpy::ndarray::ArrayView1;
use numpy::{
    Complex32, Complex64, Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::common::{ro_slice, Float};

/// Below this many rows the kernels run single-threaded: the per-thread
/// result copies and the thread wake-up would cost more than they save.
const PARALLEL_MIN_ROWS: usize = 1 << 14;

/// Chunk length for the parallel reduction over result columns.
const REDUCE_BLOCK: usize = 4096;

type PyArrayI32<'py> = Bound<'py, PyArray1<i32>>;

// ---------------------------------------------------------------------------
// Array access helpers
// ---------------------------------------------------------------------------

/// Runs `f` on the contents of a writable array. A strided array is copied
/// out, updated, and written back.
fn with_rw_slice<T: Element + Copy, R>(
    arr: &mut PyReadwriteArray1<'_, T>,
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    if let Ok(s) = arr.as_slice_mut() {
        return f(s);
    }
    let mut v = arr.as_array().to_vec();
    let r = f(&mut v);
    arr.as_array_mut().assign(&ArrayView1::from(&v));
    r
}

/// A generic callback over the float type, since closures cannot be generic.
trait FloatCall<R> {
    fn call<F: Float>(
        self,
        other: PyReadonlyArray1<'_, F>,
        out: PyReadwriteArray1<'_, F>,
    ) -> PyResult<R>;
}

/// Dispatches on the dtype of `other` and calls `f` with `other` and `out`
/// extracted as float32 or float64 arrays. `out` must have the same dtype.
fn dispatch_float<R>(
    other: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    f: impl FloatCall<R>,
) -> PyResult<R> {
    if other.is_instance_of::<PyArray1<f64>>() {
        f.call(
            other.extract::<PyReadonlyArray1<f64>>()?,
            extract_rw::<f64>(out)?,
        )
    } else if other.is_instance_of::<PyArray1<f32>>() {
        f.call(
            other.extract::<PyReadonlyArray1<f32>>()?,
            extract_rw::<f32>(out)?,
        )
    } else {
        Err(PyTypeError::new_err(
            "expected a 1d float32 or float64 numpy array",
        ))
    }
}

fn extract_rw<'py, T: Element>(out: &Bound<'py, PyAny>) -> PyResult<PyReadwriteArray1<'py, T>> {
    out.extract::<PyReadwriteArray1<T>>().map_err(|_| {
        PyTypeError::new_err("`out` must be a 1d numpy array with the dtype of the input")
    })
}

/// Indicator of the columns listed in `cols` (`get_col_included` in Cython).
fn col_mask(cols: &[i32], n_cols: usize) -> Vec<bool> {
    let mut mask = vec![false; n_cols];
    for &c in cols {
        mask[c as usize] = true;
    }
    mask
}

/// The column a row with category `index` contributes to, or `None` if the
/// row is to be skipped. Without `DROP` the index is the column.
#[inline(always)]
fn column<const DROP: bool>(index: i32, drop_first: i32) -> Option<usize> {
    if DROP {
        let col = index - drop_first;
        (col >= 0).then_some(col as usize)
    } else {
        Some(index as usize)
    }
}

/// `drop_first` as the shift to subtract from the indices (`bint` in Cython).
fn drop_shift(drop_first: &Bound<'_, PyAny>) -> PyResult<i32> {
    Ok(drop_first.is_truthy()? as i32)
}

// ---------------------------------------------------------------------------
// Parallel scatter-add
// ---------------------------------------------------------------------------

/// Calls `scatter(range, res)`, which must do `res[col] += val` for the rows
/// in `range`, over sub-ranges of `0..n` that together cover it once.
///
/// Each thread scatters a contiguous range of rows into its own copy of `res`;
/// the copies are then summed block by block. The summation order, and hence
/// the result, only depends on the number of threads.
fn scatter_add<F: Float>(n: usize, res: &mut [F], scatter: impl Fn(Range<usize>, &mut [F]) + Sync) {
    let n_threads = rayon::current_num_threads();
    if n < PARALLEL_MIN_ROWS || n_threads == 1 {
        scatter(0..n, res);
        return;
    }

    let chunk = n.div_ceil(n_threads);
    let partials: Vec<Vec<F>> = (0..n)
        .into_par_iter()
        .step_by(chunk)
        .map(|start| {
            let mut local = vec![F::default(); res.len()];
            scatter(start..(start + chunk).min(n), &mut local);
            local
        })
        .collect();

    res.par_chunks_mut(REDUCE_BLOCK)
        .enumerate()
        .for_each(|(b, block)| {
            let offset = b * REDUCE_BLOCK;
            for partial in &partials {
                for (r, p) in block.iter_mut().zip(&partial[offset..]) {
                    *r += *p;
                }
            }
        });
}

// ---------------------------------------------------------------------------
// transpose_matvec: out[indices[row]] += other[row]
// ---------------------------------------------------------------------------

/// `out[indices[row] - drop_first] += other[row]` over `rows`, restricted to
/// the columns in `cols`.
///
/// `rows` of `None` (or of full length) means all rows and `cols` of `None`
/// (or of full length) means all columns, as in the Cython original.
fn transpose_matvec_kernel<F: Float, const DROP: bool>(
    indices: &[i32],
    other: &[F],
    out: &mut [F],
    rows: Option<&[i32]>,
    cols: Option<&[i32]>,
    n_cols: usize,
    drop_first: i32,
) {
    let rows = rows.filter(|r| r.len() != indices.len());
    let mask = cols
        .filter(|c| c.len() != n_cols)
        .map(|c| col_mask(c, n_cols));
    let mask = mask.as_deref();

    match rows {
        None => scatter_add(indices.len(), out, |r, res| {
            for (&index, &val) in indices[r.clone()].iter().zip(&other[r]) {
                if let Some(col) = column::<DROP>(index, drop_first) {
                    if mask.is_none_or(|m| m[col]) {
                        res[col] += val;
                    }
                }
            }
        }),
        Some(rows) => scatter_add(rows.len(), out, |r, res| {
            for &row in &rows[r] {
                let row = row as usize;
                if let Some(col) = column::<DROP>(indices[row], drop_first) {
                    if mask.is_none_or(|m| m[col]) {
                        res[col] += other[row];
                    }
                }
            }
        }),
    }
}

struct TransposeMatvec<'a, const DROP: bool> {
    py: Python<'a>,
    indices: PyReadonlyArray1<'a, i32>,
    rows: Option<PyReadonlyArray1<'a, i32>>,
    cols: Option<PyReadonlyArray1<'a, i32>>,
    n_cols: usize,
    drop_first: i32,
}

impl<const DROP: bool> FloatCall<()> for TransposeMatvec<'_, DROP> {
    fn call<F: Float>(
        self,
        other: PyReadonlyArray1<'_, F>,
        mut out: PyReadwriteArray1<'_, F>,
    ) -> PyResult<()> {
        let indices = ro_slice(&self.indices);
        let other = ro_slice(&other);
        let rows = self.rows.as_ref().map(ro_slice);
        let cols = self.cols.as_ref().map(ro_slice);
        with_rw_slice(&mut out, |out| {
            self.py.detach(|| {
                transpose_matvec_kernel::<F, DROP>(
                    &indices,
                    &other,
                    out,
                    rows.as_deref(),
                    cols.as_deref(),
                    self.n_cols,
                    self.drop_first,
                )
            })
        });
        Ok(())
    }
}

/// `out += cat[rows, cols].T @ other[rows]` for a matrix without dropped or
/// missing categories. `dtype` is unused and kept for signature compatibility.
#[pyfunction]
#[pyo3(signature = (indices, other, n_cols, dtype, rows, cols, out))]
#[allow(clippy::too_many_arguments)]
fn transpose_matvec_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_cols: usize,
    dtype: &Bound<'py, PyAny>,
    rows: Option<PyReadonlyArray1<'py, i32>>,
    cols: Option<PyReadonlyArray1<'py, i32>>,
    out: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let _ = dtype;
    dispatch_float(
        other,
        out,
        TransposeMatvec::<false> {
            py,
            indices,
            rows,
            cols,
            n_cols,
            drop_first: 0,
        },
    )
}

/// Like `transpose_matvec_fast`, but the column index is shifted by
/// `drop_first` and negative (dropped or missing) indices are skipped.
#[pyfunction]
#[pyo3(signature = (indices, other, n_cols, dtype, rows, cols, out, drop_first))]
#[allow(clippy::too_many_arguments)]
fn transpose_matvec_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_cols: usize,
    dtype: &Bound<'py, PyAny>,
    rows: Option<PyReadonlyArray1<'py, i32>>,
    cols: Option<PyReadonlyArray1<'py, i32>>,
    out: &Bound<'py, PyAny>,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let _ = dtype;
    dispatch_float(
        other,
        out,
        TransposeMatvec::<true> {
            py,
            indices,
            rows,
            cols,
            n_cols,
            drop_first: drop_shift(drop_first)?,
        },
    )
}

/// Indicator array of the columns listed in `cols`.
#[pyfunction]
fn get_col_included<'py>(
    py: Python<'py>,
    cols: PyReadonlyArray1<'py, i32>,
    n_cols: usize,
) -> PyArrayI32<'py> {
    let mask = col_mask(&ro_slice(&cols), n_cols);
    mask.into_iter()
        .map(i32::from)
        .collect::<Vec<_>>()
        .into_pyarray(py)
}

// ---------------------------------------------------------------------------
// matvec: out[row] += other[indices[row]]
// ---------------------------------------------------------------------------

/// `out[i] += other[indices[i] - drop_first]` for the first `n_rows` rows,
/// skipping negative indices and, if `cols` is given, columns not in it.
fn matvec_kernel<F: Float, const DROP: bool>(
    indices: &[i32],
    other: &[F],
    out: &mut [F],
    n_rows: usize,
    cols: Option<&[i32]>,
    n_cols: usize,
    drop_first: i32,
) {
    let mask = cols.map(|c| col_mask(c, n_cols));
    let mask = mask.as_deref();
    out[..n_rows]
        .par_iter_mut()
        .zip(&indices[..n_rows])
        .with_min_len(PARALLEL_MIN_ROWS)
        .for_each(|(o, &index)| {
            if let Some(col) = column::<DROP>(index, drop_first) {
                if mask.is_none_or(|m| m[col]) {
                    *o += other[col];
                }
            }
        });
}

struct Matvec<'a, const DROP: bool> {
    py: Python<'a>,
    indices: PyReadonlyArray1<'a, i32>,
    n_rows: usize,
    cols: Option<PyReadonlyArray1<'a, i32>>,
    n_cols: usize,
    drop_first: i32,
}

impl<const DROP: bool> FloatCall<()> for Matvec<'_, DROP> {
    fn call<F: Float>(
        self,
        other: PyReadonlyArray1<'_, F>,
        mut out: PyReadwriteArray1<'_, F>,
    ) -> PyResult<()> {
        let indices = ro_slice(&self.indices);
        let other = ro_slice(&other);
        let cols = self.cols.as_ref().map(ro_slice);
        with_rw_slice(&mut out, |out| {
            self.py.detach(|| {
                matvec_kernel::<F, DROP>(
                    &indices,
                    &other,
                    out,
                    self.n_rows,
                    cols.as_deref(),
                    self.n_cols,
                    self.drop_first,
                )
            })
        });
        Ok(())
    }
}

/// `out_vec += cat[:, cols] @ other` for a matrix without dropped or missing
/// categories: with one-hot data this is `other[indices]`.
#[pyfunction]
#[pyo3(signature = (indices, other, n_rows, cols, n_cols, out_vec))]
fn matvec_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_rows: usize,
    cols: Option<PyReadonlyArray1<'py, i32>>,
    n_cols: usize,
    out_vec: &Bound<'py, PyAny>,
) -> PyResult<()> {
    dispatch_float(
        other,
        out_vec,
        Matvec::<false> {
            py,
            indices,
            n_rows,
            cols,
            n_cols,
            drop_first: 0,
        },
    )
}

/// Like `matvec_fast`, but the column index is shifted by `drop_first` and
/// negative (dropped or missing) indices are skipped.
#[pyfunction]
#[pyo3(signature = (indices, other, n_rows, cols, n_cols, out_vec, drop_first))]
#[allow(clippy::too_many_arguments)]
fn matvec_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    other: &Bound<'py, PyAny>,
    n_rows: usize,
    cols: Option<PyReadonlyArray1<'py, i32>>,
    n_cols: usize,
    out_vec: &Bound<'py, PyAny>,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<()> {
    dispatch_float(
        other,
        out_vec,
        Matvec::<true> {
            py,
            indices,
            n_rows,
            cols,
            n_cols,
            drop_first: drop_shift(drop_first)?,
        },
    )
}

// ---------------------------------------------------------------------------
// sandwich: diag(cat.T @ diag(d) @ cat)[col] = sum of d over rows in column col
// ---------------------------------------------------------------------------

fn sandwich_kernel<F: Float, const DROP: bool>(
    indices: &[i32],
    d: &[F],
    rows: &[i32],
    n_cols: usize,
    drop_first: i32,
) -> Vec<F> {
    let mut res = vec![F::default(); n_cols];
    scatter_add(rows.len(), &mut res, |r, res| {
        for &k in &rows[r] {
            let k = k as usize;
            if let Some(col) = column::<DROP>(indices[k], drop_first) {
                res[col] += d[k];
            }
        }
    });
    res
}

/// Diagonal of the sandwich product restricted to `rows`, as a float array.
fn sandwich<'py, F: Float, const DROP: bool>(
    py: Python<'py>,
    indices: &PyReadonlyArray1<'py, i32>,
    d: PyReadonlyArray1<'py, F>,
    rows: &PyReadonlyArray1<'py, i32>,
    n_cols: usize,
    drop_first: i32,
) -> Bound<'py, PyArray1<F>> {
    let indices = ro_slice(indices);
    let d = ro_slice(&d);
    let rows = ro_slice(rows);
    py.detach(|| sandwich_kernel::<F, DROP>(&indices, &d, &rows, n_cols, drop_first))
        .into_pyarray(py)
}

fn dispatch_sandwich<'py, const DROP: bool>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    d: &Bound<'py, PyAny>,
    rows: PyReadonlyArray1<'py, i32>,
    n_cols: usize,
    drop_first: i32,
) -> PyResult<Bound<'py, PyAny>> {
    if d.is_instance_of::<PyArray1<f64>>() {
        let d = d.extract::<PyReadonlyArray1<f64>>()?;
        Ok(sandwich::<f64, DROP>(py, &indices, d, &rows, n_cols, drop_first).into_any())
    } else if d.is_instance_of::<PyArray1<f32>>() {
        let d = d.extract::<PyReadonlyArray1<f32>>()?;
        Ok(sandwich::<f32, DROP>(py, &indices, d, &rows, n_cols, drop_first).into_any())
    } else {
        Err(PyTypeError::new_err(
            "expected a 1d float32 or float64 numpy array",
        ))
    }
}

/// Diagonal of `cat[rows].T @ diag(d[rows]) @ cat[rows]` for a matrix without
/// dropped or missing categories. `dtype` is unused and kept for signature
/// compatibility; the result has the dtype of `d`.
#[pyfunction]
#[pyo3(signature = (indices, d, rows, dtype, n_cols))]
fn sandwich_categorical_fast<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    d: &Bound<'py, PyAny>,
    rows: PyReadonlyArray1<'py, i32>,
    dtype: &Bound<'py, PyAny>,
    n_cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = dtype;
    dispatch_sandwich::<false>(py, indices, d, rows, n_cols, 0)
}

/// Like `sandwich_categorical_fast`, but the column index is shifted by
/// `drop_first` and negative (dropped or missing) indices are skipped.
#[pyfunction]
#[pyo3(signature = (indices, d, rows, dtype, n_cols, drop_first))]
fn sandwich_categorical_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    d: &Bound<'py, PyAny>,
    rows: PyReadonlyArray1<'py, i32>,
    dtype: &Bound<'py, PyAny>,
    n_cols: usize,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = dtype;
    dispatch_sandwich::<true>(py, indices, d, rows, n_cols, drop_shift(drop_first)?)
}

// ---------------------------------------------------------------------------
// CSR conversion helpers
// ---------------------------------------------------------------------------

/// CSR `indices` and `indptr` of the matrix with the first `drop_first`
/// categories dropped and missing values skipped: one entry per row whose
/// shifted column index is non-negative.
fn csr_structure(indices: &[i32], drop_first: i32) -> (Vec<i32>, Vec<i32>) {
    let mut new_indices = Vec::with_capacity(indices.len());
    let mut indptr = Vec::with_capacity(indices.len() + 1);
    for &index in indices {
        indptr.push(new_indices.len() as i32);
        if let Some(col) = column::<true>(index, drop_first) {
            new_indices.push(col as i32);
        }
    }
    indptr.push(new_indices.len() as i32);
    (new_indices, indptr)
}

/// CSR `(data, indices, indptr)` of `diag(d) @ cat`, with the first
/// `drop_first` categories dropped. `ncols` and `dtype` are unused and kept for
/// signature compatibility; `data` has the dtype of `d`.
#[pyfunction]
#[pyo3(signature = (indices, d, ncols, dtype, drop_first))]
fn multiply_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    d: &Bound<'py, PyAny>,
    ncols: usize,
    dtype: &Bound<'py, PyAny>,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, PyArrayI32<'py>, PyArrayI32<'py>)> {
    let _ = (ncols, dtype);
    let drop_first = drop_shift(drop_first)?;
    let indices = ro_slice(&indices);

    /// The kept entries of `d`, as a numpy array of the same dtype.
    fn kept_data<'py, T: Element + Copy>(
        py: Python<'py>,
        indices: &[i32],
        d: PyReadonlyArray1<'py, T>,
        drop_first: i32,
    ) -> Bound<'py, PyAny> {
        let d = ro_slice(&d);
        let data: Vec<T> = indices
            .iter()
            .zip(d.iter())
            .filter(|(&index, _)| column::<true>(index, drop_first).is_some())
            .map(|(_, &v)| v)
            .collect();
        data.into_pyarray(py).into_any()
    }

    macro_rules! dispatch {
        ($($t:ty),*) => {
            $(
                if d.is_instance_of::<PyArray1<$t>>() {
                    let data = kept_data(py, &indices, d.extract::<PyReadonlyArray1<$t>>()?, drop_first);
                    let (new_indices, indptr) = csr_structure(&indices, drop_first);
                    return Ok((data, new_indices.into_pyarray(py), indptr.into_pyarray(py)));
                }
            )*
        };
    }
    // The dtypes of Cython's `numeric` fused type plus `long long`.
    dispatch!(f64, f32, i64, i32, i16, Complex64, Complex32);
    Err(PyTypeError::new_err(
        "expected a 1d numpy array of a signed integer, float or complex dtype",
    ))
}

/// Number of nonzeros and CSR `indices` and `indptr` of the matrix with the
/// first `drop_first` categories dropped and missing values skipped. `ncols`
/// is unused and kept for signature compatibility.
#[pyfunction]
#[pyo3(signature = (indices, ncols, drop_first))]
fn subset_categorical_complex<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<'py, i32>,
    ncols: usize,
    drop_first: &Bound<'py, PyAny>,
) -> PyResult<(usize, PyArrayI32<'py>, PyArrayI32<'py>)> {
    let _ = ncols;
    let (new_indices, indptr) = csr_structure(&ro_slice(&indices), drop_shift(drop_first)?);
    Ok((
        new_indices.len(),
        new_indices.into_pyarray(py),
        indptr.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

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
