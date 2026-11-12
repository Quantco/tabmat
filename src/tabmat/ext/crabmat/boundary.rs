//! The Python boundary shared by the binding modules: dtype dispatch,
//! argument coercion and the conversion of kernel errors into Python
//! exceptions. Nothing in here computes anything; the kernels live in
//! [`crate::kernels`] and know nothing about Python.
//!
//! Every function dispatches on the dtype of one array (the matrix, or the
//! weights `d` where there is no matrix) with [`for_float!`] (and
//! [`for_index!`] where the index type matters), so each kernel is compiled
//! once per element type like Cython's fused types. The other one-dimensional
//! arguments are accepted as anything numpy can convert to that dtype
//! ([`Arg`]): the Python layer passes matching dtypes on every hot path, so
//! the conversion only ever costs something for callers that do not, and
//! never raises. Output arrays, two-dimensional arrays and sparse matrices
//! must match the dtype exactly, since converting them would silently copy
//! the data.

use std::borrow::Cow;

use numpy::ndarray::ArrayView1;
use numpy::{
    AllowTypeChange, Element, PyArrayDescr, PyArrayLike1, PyReadonlyArray1, PyReadwriteArray1,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::kernels::{Float, IndexError};

/// A kernel element type as seen from Python: a numpy element that PyO3
/// can extract.
pub trait PyFloat: Float + Element + for<'a, 'py> FromPyObject<'a, 'py> {}
impl PyFloat for f32 {}
impl PyFloat for f64 {}

/// The dtype of the numpy array `arr`, for dispatching on it.
pub fn dtype_of<'py>(arr: &Bound<'py, PyAny>, what: &str) -> PyResult<Bound<'py, PyArrayDescr>> {
    let arr = arr
        .cast::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err(format!("{what} must be a numpy array")))?;
    Ok(arr.dtype())
}

/// Evaluates `$body` with the type alias `$T` bound to the Rust type of the
/// dtype of the numpy array `$arr`, which must be one of `$types`; raises
/// `TypeError` otherwise. `$body` is compiled once per listed type, like a
/// Cython function over a fused type, and must evaluate to a `PyResult`.
macro_rules! with_dtype {
    ($arr:expr, $what:expr, $T:ident in [$($t:ty),+ $(,)?] => $body:expr) => {{
        let arr: &::pyo3::Bound<'_, ::pyo3::PyAny> = $arr;
        let dtype = $crate::boundary::dtype_of(arr, $what)?;
        $(
            if ::numpy::PyArrayDescrMethods::is_equiv_to(&dtype, &::numpy::dtype::<$t>(arr.py())) {
                #[allow(non_camel_case_types, dead_code)]
                type $T = $t;
                $body
            } else
        )+
        {
            Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                "{} must have dtype {}, not {}",
                $what,
                stringify!($($t),+).replace(", ", " or "),
                dtype
            )))
        }
    }};
}
pub(crate) use with_dtype;

/// Evaluates `$body` with `$F` bound to `f64` or `f32`, the dtype of the
/// floating point array `$arr`.
macro_rules! for_float {
    ($arr:expr, $what:expr, $F:ident => $body:expr) => {
        $crate::boundary::with_dtype!($arr, $what, $F in [f64, f32] => $body)
    };
}
pub(crate) use for_float;

/// Evaluates `$body` with the const `$DROP` bound to the value of `$drop`:
/// the dispatch of the categorical kernels to their plain (`false`) or
/// `drop_first`/missing-value-aware (`true`) instantiation. The `$shift`
/// flags are rebound as `i32` in the `true` branch and as the literal `0`
/// in the `false` one, which ignores them: passing the runtime value to
/// the plain kernel too measured 1.2x slower at one thread (the extra live
/// value spilled the loop's invariants).
macro_rules! with_drop {
    ($drop:expr, $DROP:ident, $($shift:ident),+ => $body:expr) => {
        if $drop {
            const $DROP: bool = true;
            $(let $shift = $shift as i32;)+
            $body
        } else {
            const $DROP: bool = false;
            $(let $shift = 0i32;)+
            $body
        }
    };
}
pub(crate) use with_drop;

/// Evaluates `$body` with `$I` bound to `i32` or `i64`, the dtype of the
/// index array `$arr`.
macro_rules! for_index {
    ($arr:expr, $what:expr, $I:ident => $body:expr) => {
        $crate::boundary::with_dtype!($arr, $what, $I in [i32, i64] => $body)
    };
}
pub(crate) use for_index;

/// A one-dimensional argument array of element type `T`: any array (or
/// array-like) numpy can convert to that dtype is accepted. The conversion
/// is a copy, but the Python layer passes the right dtype already.
pub type Arg<'py, T> = PyArrayLike1<'py, T, AllowTypeChange>;

/// A contiguous view of a read-only array, copying only if it is strided.
pub fn ro_slice<'a, T: Element + Copy>(arr: &'a PyReadonlyArray1<'_, T>) -> Cow<'a, [T]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().to_vec()),
    }
}

/// Extracts the output array `out`, which must be a one-dimensional array of
/// the dtype `T` of the call (an output cannot be converted).
pub fn rw_array<'py, T: Element>(
    out: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadwriteArray1<'py, T>> {
    out.extract::<PyReadwriteArray1<T>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be a writable 1d numpy array of the dtype of the call"
        ))
    })
}

/// Runs `f` on the contents of a writable array. A strided array is copied
/// out, updated, and written back.
pub fn with_rw_slice<T: Element + Copy, R>(
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

/// Extracts the optional `center` argument of a kernel as a vector with
/// one entry per column of the dense operand (`None` stays `None`).
pub fn extract_center<'py, F: PyFloat>(
    center: Option<&Bound<'py, PyAny>>,
    n_cols: usize,
    what: &str,
) -> PyResult<Option<Arg<'py, F>>> {
    let Some(center) = center.filter(|c| !c.is_none()) else {
        return Ok(None);
    };
    let center = center.extract::<Arg<'py, F>>()?;
    if center.len() != n_cols {
        return Err(PyValueError::new_err(format!(
            "{what} must have length {n_cols}, got {}.",
            center.len()
        )));
    }
    Ok(Some(center))
}

impl From<IndexError> for PyErr {
    fn from(e: IndexError) -> PyErr {
        PyIndexError::new_err(format!(
            "{} index out of bounds for axis of length {}",
            e.what, e.len
        ))
    }
}
