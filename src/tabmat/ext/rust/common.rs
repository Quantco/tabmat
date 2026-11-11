//! Helpers shared by the kernel submodules: the floating point element
//! trait the kernels are instantiated for, the thread pool setup and numpy
//! array access.

use std::borrow::Cow;
use std::ops::{Add, AddAssign, Mul, Sub};

use numpy::{Element, PyReadonlyArray1};

/// Element types the floating point kernels are instantiated for, mirroring
/// Cython's `floating` fused type.
pub trait Float:
    Element
    + Copy
    + Default
    + Send
    + Sync
    + AddAssign
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + PartialOrd
    + private::Sealed
{
}

impl Float for f32 {}
impl Float for f64 {}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A contiguous view of a read-only array, copying only if it is strided.
pub fn ro_slice<'a, T: Element + Copy>(arr: &'a PyReadonlyArray1<'_, T>) -> Cow<'a, [T]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().to_vec()),
    }
}

/// Sizes rayon's global thread pool like OpenMP would: `RAYON_NUM_THREADS`
/// wins, then `OMP_NUM_THREADS`, then one thread per core. Called once, when
/// the extension module is imported; a pool that already exists (for example
/// because another Rust extension in the process built one first) is left
/// alone.
pub fn init_thread_pool() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let Some(n) = std::env::var("OMP_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    else {
        return;
    };
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
}
