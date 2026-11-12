//! The kernels: plain Rust over slices and `ndarray` views, with no PyO3 or
//! numpy types, so that they can be read, tested and benchmarked on their
//! own. The binding modules next to this directory extract the Python
//! arguments and call them. This module holds what the kernel families share:
//! the element type traits and SIMD helpers, index validation, the dense
//! operand with an optional center, the categorical index helper, the thread
//! pool setup and the deterministic parallel reductions.
//!
//! # Conventions
//!
//! Every index is checked, so an index outside its axis is reported as an
//! [`IndexError`] (Python's `IndexError` at the boundary) instead of
//! corrupting memory; the C++ kernels did not check at all. The check happens
//! where it is cheapest for the kernel at hand: up front with a vectorised
//! pass ([`check_indices`]) when the kernel does a whole row or column of
//! work per index entry, so that the pass is negligible and the inner loops
//! can read without checks; in L1-sized sub-chunks of the index array when
//! the kernel does only a few operations per entry, where a separate pass over
//! the whole array was measured to cost 20% of the memory traffic; and as a
//! bounds check in the loop when the index is part of the data stream the loop
//! reads anyway (the indices of a sparse matrix, the categories of a
//! categorical matrix). `unsafe` is used only for the unchecked loads that
//! follow such a validation, each with a `SAFETY` comment naming it.
//!
//! The kernels run on rayon's global thread pool (the bindings release the
//! GIL around them). Where threads accumulate into the same output, each gets
//! a private copy that is summed afterwards in thread order
//! ([`partial_sums`]): the result then only depends on the number of threads,
//! not on their timing (the C++ kernels used atomic adds). Measured against
//! rayon's own `fold`/`reduce` this costs nothing and is deterministic by
//! construction.
//!
//! Each kernel module carries a `#[cfg(test)]` reference implementation
//! written as the plainest possible loops, and `cargo test` checks the
//! kernels against it over random shapes, subsets, layouts and thread
//! counts, and checks that repeated runs agree bit for bit.

pub mod categorical;
pub mod dense;
pub mod sparse;

use std::borrow::Cow;
use std::ops::{Add, AddAssign, Mul, Range, Sub};

use ndarray::{ArrayView2, ArrayViewMut2, Axis, Zip};
use pulp::Simd;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------

/// Element types the floating point kernels are instantiated for, mirroring
/// Cython's `floating` fused type.
pub trait Float:
    Copy
    + Default
    + Send
    + Sync
    + AddAssign
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + PartialOrd
    + private::Sealed
    + 'static
{
}

impl Float for f32 {}
impl Float for f64 {}

/// Index types of index arrays (the `indices` and `indptr` of a compressed
/// sparse matrix, row and column subsets), mirroring Cython's `win_integral`
/// fused type: scipy uses `int32` or `int64`.
pub trait Index: Copy + Default + Send + Sync + Ord + private::Sealed + 'static {
    /// The index as a `usize`; a negative index becomes a huge value, which
    /// the bounds checks of the kernels reject.
    fn as_usize(self) -> usize;
}

impl Index for i32 {
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl Index for i64 {
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
}

/// What the SIMD kernels need from the element type: its vector type for the
/// instruction set `S`, and splat, arithmetic, fused multiply-add and
/// horizontal sum on it. `pulp` names these per element type; this trait
/// lets one kernel be written for both `f32` and `f64`.
pub trait SimdFloat: Float {
    type Vec<S: Simd>: Copy + Send + Sync;

    /// Number of lanes of `Vec<S>`.
    fn lanes<S: Simd>() -> usize {
        std::mem::size_of::<Self::Vec<S>>() / std::mem::size_of::<Self>()
    }
    fn splat<S: Simd>(simd: S, v: Self) -> Self::Vec<S>;
    fn zero<S: Simd>(simd: S) -> Self::Vec<S> {
        Self::splat(simd, Self::default())
    }
    /// Splits a slice into whole vectors and a scalar tail.
    fn as_simd<S: Simd>(slice: &[Self]) -> (&[Self::Vec<S>], &[Self]);
    /// Splits a mutable slice into whole vectors and a scalar tail.
    fn as_mut_simd<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Vec<S>], &mut [Self]);
    fn vadd<S: Simd>(simd: S, a: Self::Vec<S>, b: Self::Vec<S>) -> Self::Vec<S>;
    fn vsub<S: Simd>(simd: S, a: Self::Vec<S>, b: Self::Vec<S>) -> Self::Vec<S>;
    /// `a * b + c`, fused where the hardware has FMA.
    fn mul_add<S: Simd>(simd: S, a: Self::Vec<S>, b: Self::Vec<S>, c: Self::Vec<S>)
        -> Self::Vec<S>;
    fn reduce_sum<S: Simd>(simd: S, v: Self::Vec<S>) -> Self;
}

macro_rules! impl_simd_float {
    ($t:ty, $vec:ident, $splat:ident, $as_simd:ident, $as_mut_simd:ident, $add:ident, $sub:ident, $mul_add:ident, $reduce:ident) => {
        impl SimdFloat for $t {
            type Vec<S: Simd> = S::$vec;

            #[inline(always)]
            fn splat<S: Simd>(simd: S, v: $t) -> S::$vec {
                simd.$splat(v)
            }
            #[inline(always)]
            fn as_simd<S: Simd>(slice: &[$t]) -> (&[S::$vec], &[$t]) {
                S::$as_simd(slice)
            }
            #[inline(always)]
            fn as_mut_simd<S: Simd>(slice: &mut [$t]) -> (&mut [S::$vec], &mut [$t]) {
                S::$as_mut_simd(slice)
            }
            #[inline(always)]
            fn vadd<S: Simd>(simd: S, a: S::$vec, b: S::$vec) -> S::$vec {
                simd.$add(a, b)
            }
            #[inline(always)]
            fn vsub<S: Simd>(simd: S, a: S::$vec, b: S::$vec) -> S::$vec {
                simd.$sub(a, b)
            }
            #[inline(always)]
            fn mul_add<S: Simd>(simd: S, a: S::$vec, b: S::$vec, c: S::$vec) -> S::$vec {
                simd.$mul_add(a, b, c)
            }
            #[inline(always)]
            fn reduce_sum<S: Simd>(simd: S, v: S::$vec) -> $t {
                simd.$reduce(v)
            }
        }
    };
}

impl_simd_float!(
    f64,
    f64s,
    splat_f64s,
    as_simd_f64s,
    as_mut_simd_f64s,
    add_f64s,
    sub_f64s,
    mul_add_e_f64s,
    reduce_sum_f64s
);
impl_simd_float!(
    f32,
    f32s,
    splat_f32s,
    as_simd_f32s,
    as_mut_simd_f32s,
    add_f32s,
    sub_f32s,
    mul_add_e_f32s,
    reduce_sum_f32s
);

// ---------------------------------------------------------------------------
// SIMD vector kernels
// ---------------------------------------------------------------------------
//
// The inner loops of the matrix-vector products and of the sparse x dense
// sandwich, written once against `pulp` so that they run with the vector
// width and FMA of the instruction set detected at run time. The `centered`
// variants subtract a per-column center from `x` as they read it, which is
// how the kernels compute with `X - center` without copying `X` (expanding
// into `X @ v - center @ v` would cancel for a large center). Within a
// variant the summation order is fixed, so results are deterministic.

/// `y += a * x` over the whole slices.
#[inline(always)]
pub fn axpy<S: Simd, F: SimdFloat>(simd: S, a: F, x: &[F], y: &mut [F]) {
    let (xv, xt) = F::as_simd::<S>(x);
    let (yv, yt) = F::as_mut_simd::<S>(y);
    let av = F::splat(simd, a);
    for (y, &x) in yv.iter_mut().zip(xv) {
        *y = F::mul_add(simd, av, x, *y);
    }
    for (y, &x) in yt.iter_mut().zip(xt) {
        *y += a * x;
    }
}

/// `y[k] += (x[k] - c) * a` for a scalar center `c`.
#[inline(always)]
pub fn centered_axpy<S: Simd, F: SimdFloat>(simd: S, x: &[F], c: F, a: F, y: &mut [F]) {
    let (xv, xt) = F::as_simd::<S>(x);
    let (yv, yt) = F::as_mut_simd::<S>(y);
    let (av, cv) = (F::splat(simd, a), F::splat(simd, c));
    for (y, &x) in yv.iter_mut().zip(xv) {
        *y = F::mul_add(simd, F::vsub(simd, x, cv), av, *y);
    }
    for (y, &x) in yt.iter_mut().zip(xt) {
        *y += (x - c) * a;
    }
}

/// `y[k] += (x[k] - c[k]) * a` for a per-element center `c`.
#[inline(always)]
pub fn centered_axpy_vc<S: Simd, F: SimdFloat>(simd: S, x: &[F], c: &[F], a: F, y: &mut [F]) {
    let (xv, xt) = F::as_simd::<S>(x);
    let (cv, ct) = F::as_simd::<S>(c);
    let (yv, yt) = F::as_mut_simd::<S>(y);
    let av = F::splat(simd, a);
    for ((y, &x), &c) in yv.iter_mut().zip(xv).zip(cv) {
        *y = F::mul_add(simd, F::vsub(simd, x, c), av, *y);
    }
    for ((y, &x), &c) in yt.iter_mut().zip(xt).zip(ct) {
        *y += (x - c) * a;
    }
}

/// `y[k] += sum_t (x[t][k] - c[t]) * a[t]` for four columns at once, which
/// reads and writes `y` once for the four.
#[inline(always)]
pub fn centered_axpy4<S: Simd, F: SimdFloat>(
    simd: S,
    x: [&[F]; 4],
    c: [F; 4],
    a: [F; 4],
    y: &mut [F],
) {
    let xs = x.map(|x| F::as_simd::<S>(x));
    let (yv, yt) = F::as_mut_simd::<S>(y);
    let cv = c.map(|c| F::splat(simd, c));
    let av = a.map(|a| F::splat(simd, a));
    let nvec = yv.len();
    let xv = xs.map(|(v, _)| &v[..nvec]);
    for (k, y) in yv.iter_mut().enumerate() {
        let mut yk = *y;
        for t in 0..4 {
            yk = F::mul_add(simd, F::vsub(simd, xv[t][k], cv[t]), av[t], yk);
        }
        *y = yk;
    }
    let xt = xs.map(|(_, t)| t);
    for (k, y) in yt.iter_mut().enumerate() {
        let mut yk = *y;
        for t in 0..4 {
            yk += (xt[t][k] - c[t]) * a[t];
        }
        *y = yk;
    }
}

/// `sum_k (x[k] - c) * y[k]` for a scalar center `c`, with two independent
/// accumulators.
#[inline(always)]
pub fn centered_dot<S: Simd, F: SimdFloat>(simd: S, x: &[F], c: F, y: &[F]) -> F {
    let (xv, xt) = F::as_simd::<S>(x);
    let (yv, yt) = F::as_simd::<S>(y);
    let cv = F::splat(simd, c);
    let (mut acc0, mut acc1) = (F::zero(simd), F::zero(simd));
    let mut xp = xv.chunks_exact(2);
    let mut yp = yv.chunks_exact(2);
    for (x, y) in (&mut xp).zip(&mut yp) {
        acc0 = F::mul_add(simd, F::vsub(simd, x[0], cv), y[0], acc0);
        acc1 = F::mul_add(simd, F::vsub(simd, x[1], cv), y[1], acc1);
    }
    for (&x, &y) in xp.remainder().iter().zip(yp.remainder()) {
        acc0 = F::mul_add(simd, F::vsub(simd, x, cv), y, acc0);
    }
    let mut total = F::reduce_sum(simd, F::vadd(simd, acc0, acc1));
    for (&x, &y) in xt.iter().zip(yt) {
        total += (x - c) * y;
    }
    total
}

/// `sum_k (x[k] - c[k]) * y[k]` for a per-element center `c`, with two
/// independent accumulators.
#[inline(always)]
pub fn centered_dot_vc<S: Simd, F: SimdFloat>(simd: S, x: &[F], c: &[F], y: &[F]) -> F {
    let (xv, xt) = F::as_simd::<S>(x);
    let (cv, ct) = F::as_simd::<S>(c);
    let (yv, yt) = F::as_simd::<S>(y);
    let (mut acc0, mut acc1) = (F::zero(simd), F::zero(simd));
    let mut xp = xv.chunks_exact(2);
    let mut cp = cv.chunks_exact(2);
    let mut yp = yv.chunks_exact(2);
    for ((x, c), y) in (&mut xp).zip(&mut cp).zip(&mut yp) {
        acc0 = F::mul_add(simd, F::vsub(simd, x[0], c[0]), y[0], acc0);
        acc1 = F::mul_add(simd, F::vsub(simd, x[1], c[1]), y[1], acc1);
    }
    for ((&x, &c), &y) in xp
        .remainder()
        .iter()
        .zip(cp.remainder())
        .zip(yp.remainder())
    {
        acc0 = F::mul_add(simd, F::vsub(simd, x, c), y, acc0);
    }
    let mut total = F::reduce_sum(simd, F::vadd(simd, acc0, acc1));
    for ((&x, &c), &y) in xt.iter().zip(ct).zip(yt) {
        total += (x - c) * y;
    }
    total
}

// ---------------------------------------------------------------------------
// Index validation
// ---------------------------------------------------------------------------

/// An index outside its axis, found by a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexError {
    /// What was indexed ("row", "column", ...).
    pub what: &'static str,
    /// The length of the axis.
    pub len: usize,
}

/// The smallest and largest entry of `idx` (zeros if it is empty): a
/// min/max reduction rather than an early-exit scan, written with eight
/// independent accumulators so that the compiler vectorises it.
#[inline(always)]
pub fn index_range<I: Index>(idx: &[I]) -> (I, I) {
    const LANES: usize = 8;
    let mut lo = [I::default(); LANES];
    let mut hi = [I::default(); LANES];
    let mut chunks = idx.chunks_exact(LANES);
    for chunk in &mut chunks {
        for l in 0..LANES {
            lo[l] = lo[l].min(chunk[l]);
            hi[l] = hi[l].max(chunk[l]);
        }
    }
    for &i in chunks.remainder() {
        lo[0] = lo[0].min(i);
        hi[0] = hi[0].max(i);
    }
    let lo = lo.iter().fold(I::default(), |a, &b| a.min(b));
    let hi = hi.iter().fold(I::default(), |a, &b| a.max(b));
    (lo, hi)
}

/// Checks that every index in `idx` is non-negative and below `len`, with
/// one vectorised pass; `what` names the axis in the error.
#[inline(always)]
pub fn check_indices<I: Index>(
    idx: &[I],
    len: usize,
    what: &'static str,
) -> Result<(), IndexError> {
    let (lo, hi) = index_range(idx);
    if idx.is_empty() || (lo >= I::default() && hi.as_usize() < len) {
        Ok(())
    } else {
        Err(IndexError { what, len })
    }
}

// ---------------------------------------------------------------------------
// Categorical indices
// ---------------------------------------------------------------------------

/// The column a row with category `index` contributes to, or `None` if the
/// row is to be skipped. With `DROP` the index is shifted by `drop_first`
/// (the dropped reference category) and a negative result (the dropped
/// category, or `-1` for a missing value) skips the row; without `DROP` the
/// index is the column and there is no check at all, which is what makes
/// the plain kernels fast.
#[inline(always)]
pub fn category_column<const DROP: bool>(index: i32, drop_first: i32) -> Option<usize> {
    if DROP {
        let col = index - drop_first;
        (col >= 0).then_some(col as usize)
    } else {
        Some(index as usize)
    }
}

// ---------------------------------------------------------------------------
// Dense operands with an optional center
// ---------------------------------------------------------------------------

/// A dense matrix operand `X - center`: a view of `X` with any strides (C
/// order, Fortran order or non-contiguous, read in place) and an optional
/// per-column center, indexed by the original column index like `X`. The
/// kernels that read dense data subtract the center as they go, so that
/// `StandardizedMatrix` can compute with centered data without copying it.
/// Without a center the operand is `X` itself; subtracting a zero center
/// gives exactly the same values.
#[derive(Clone, Copy)]
pub struct Centered<'a, F> {
    pub x: ArrayView2<'a, F>,
    pub center: Option<&'a [F]>,
}

impl<'a, F: Float> Centered<'a, F> {
    pub fn nrows(&self) -> usize {
        self.x.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.x.ncols()
    }

    /// The center of column `col`: zero without a center.
    #[inline(always)]
    pub fn center_of(&self, col: usize) -> F {
        self.center.map_or(F::default(), |c| c[col])
    }

    /// The center as a vector with one entry per column: zeros without a
    /// center.
    pub fn center_vec(&self) -> Cow<'a, [F]> {
        match self.center {
            Some(c) => Cow::Borrowed(c),
            None => Cow::Owned(vec![F::default(); self.ncols()]),
        }
    }

    /// Element `(row, col)` of `X` without a bounds check.
    ///
    /// # Safety
    /// `row < nrows()` and `col < ncols()`. The kernels rely on having
    /// validated their index arrays ([`check_indices`]) so that their inner
    /// loops are plain loads.
    #[inline(always)]
    pub unsafe fn get(&self, row: usize, col: usize) -> F {
        debug_assert!(row < self.nrows() && col < self.ncols());
        unsafe { *self.x.uget((row, col)) }
    }

    /// Packs `d[rows[k]] * (X[rows[k], cols[j]] - mu[cols[j]])` into
    /// `dst[k * stride + j]`, row-major, for `mu` the center as a vector
    /// ([`Self::center_vec`]): the block packing of the sparse x dense
    /// sandwich.
    ///
    /// # Safety
    /// As for [`Self::gather_column`], for every row and column index, and
    /// `mu` has one entry per column.
    #[inline(always)]
    pub unsafe fn pack_block<I: Index>(
        &self,
        rows: &[I],
        cols: &[I],
        d: &[F],
        mu: &[F],
        dst: &mut [F],
        stride: usize,
    ) {
        for (drow, &k) in dst.chunks_mut(stride).zip(rows) {
            let k = k.as_usize();
            // SAFETY: the caller's contract.
            let dk = unsafe { *d.get_unchecked(k) };
            for (dst, &j) in drow.iter_mut().zip(cols) {
                let j = j.as_usize();
                *dst = dk * (unsafe { self.get(k, j) - *mu.get_unchecked(j) });
            }
        }
    }

    /// Gathers `X[rows[k], col] - center[col]` into `dst[k]`, times
    /// `d[rows[k]]` when `SCALE`: the column packing of the sandwich panels.
    ///
    /// # Safety
    /// `col < ncols()`, every row index is below `nrows()`, and `d` has one
    /// entry per row of `X` when `SCALE`.
    #[inline(always)]
    pub unsafe fn gather_column<I: Index, const SCALE: bool>(
        &self,
        col: usize,
        rows: &[I],
        d: &[F],
        dst: &mut [F],
    ) {
        let mu = self.center_of(col);
        let column = self.x.column(col);
        let dst = &mut dst[..rows.len()];
        // SAFETY (both branches): the caller's contract.
        match column.as_slice() {
            // a contiguous column: let the compiler see the unit stride
            Some(column) => {
                gather::<F, I, SCALE>(dst, rows, d, mu, |k| unsafe { *column.get_unchecked(k) })
            }
            None => gather::<F, I, SCALE>(dst, rows, d, mu, |k| unsafe { *column.uget(k) }),
        }
    }
}

/// `dst[k] = get(rows[k]) - mu`, times `d[rows[k]]` when `SCALE`; `get` is
/// the element accessor of one column of `X`.
#[inline(always)]
fn gather<F: Float, I: Index, const SCALE: bool>(
    dst: &mut [F],
    rows: &[I],
    d: &[F],
    mu: F,
    get: impl Fn(usize) -> F,
) {
    for (dst, &k) in dst.iter_mut().zip(rows) {
        let k = k.as_usize();
        let v = get(k) - mu;
        *dst = if SCALE {
            // SAFETY: the caller's contract, `d` has an entry per row.
            unsafe { *d.get_unchecked(k) * v }
        } else {
            v
        };
    }
}

/// Copies the lower triangle of the square matrix `a` into its upper one.
///
/// The matrix is split into quadrants: the upper right one is the
/// transpose of the lower left one (copied with a parallel `Zip`), and the
/// two diagonal ones are mirrored recursively under `rayon::join`, down to
/// `LEAF`-sized blocks done by one task.
pub fn mirror_lower_triangle<F: Float>(a: ArrayViewMut2<'_, F>) {
    /// Below this edge length a block is mirrored by one task.
    const LEAF: usize = 64;
    let n = a.nrows();
    if n <= LEAF {
        let mut a = a;
        for i in 0..n {
            for j in i + 1..n {
                a[[i, j]] = a[[j, i]];
            }
        }
        return;
    }
    let mid = n / 2;
    let (top, bottom) = a.split_at(Axis(0), mid);
    let (top_left, mut top_right) = top.split_at(Axis(1), mid);
    let (bottom_left, bottom_right) = bottom.split_at(Axis(1), mid);
    rayon::join(
        || {
            Zip::from(top_right.rows_mut())
                .and(bottom_left.columns())
                .par_for_each(|mut row, col| row.assign(&col));
        },
        || {
            rayon::join(
                || mirror_lower_triangle(top_left),
                || mirror_lower_triangle(bottom_right),
            );
        },
    );
}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Deterministic parallel reductions
// ---------------------------------------------------------------------------

/// Chunk length of the parallel reduction over the entries of a result.
pub const REDUCE_BLOCK: usize = 4096;

/// Runs `work(w, partial)` for every worker `w` in `0..n_workers` in
/// parallel, each on its own zeroed copy of `out`, and adds the copies to
/// `out` afterwards, entry by entry in worker order (block-parallel over the
/// entries). `work` must only accumulate (`+=`) into its buffer: with a
/// single worker it runs on `out` itself, and the sum of the partials is
/// then what it would have produced on `out` directly.
///
/// The summation order, and hence the result, only depends on
/// `n_workers`, never on thread timing. This is the one way the kernels
/// combine per-thread results: a static split of the work over the workers
/// and this reduction in a fixed order.
pub fn partial_sums<F: Float, E: Send>(
    n_workers: usize,
    out: &mut [F],
    work: impl Fn(usize, &mut [F]) -> Result<(), E> + Sync,
) -> Result<(), E> {
    if n_workers <= 1 {
        return work(0, out);
    }
    // Allocated (and zeroed) in parallel: faulting in the pages of the
    // copies is a visible part of the call for mid-sized results.
    let partials: Vec<Vec<F>> = (0..n_workers)
        .into_par_iter()
        .map(|w| {
            let mut partial = vec![F::default(); out.len()];
            work(w, &mut partial)?;
            Ok(partial)
        })
        .collect::<Result<_, E>>()?;
    out.par_chunks_mut(REDUCE_BLOCK)
        .enumerate()
        .for_each(|(b, block)| {
            let offset = b * REDUCE_BLOCK;
            for partial in &partials {
                for (o, &p) in block.iter_mut().zip(&partial[offset..]) {
                    *o += p;
                }
            }
        });
    Ok(())
}

/// Below this many rows the scatter kernels run single-threaded: the
/// per-thread result copies and the thread wake-up would cost more than
/// they save.
pub const PARALLEL_MIN_ROWS: usize = 1 << 14;

/// Calls `scatter(range, res)`, which must do `res[col] += val` for the rows
/// in `range`, over sub-ranges of `0..n` that together cover it once.
///
/// Each thread scatters a contiguous range of rows into its own copy of
/// `res` ([`partial_sums`]). The scatter runs on one thread when there are
/// few rows, or when `res` has more entries than there are rows to scatter:
/// summing the per-thread copies would then cost more than the scatter
/// itself.
pub fn scatter_add<F: Float, E: Send>(
    n: usize,
    res: &mut [F],
    scatter: impl Fn(Range<usize>, &mut [F]) -> Result<(), E> + Sync,
) -> Result<(), E> {
    let n_threads = rayon::current_num_threads();
    if n < PARALLEL_MIN_ROWS || n_threads == 1 || res.len() > n {
        return scatter(0..n, res);
    }
    let chunk = n.div_ceil(n_threads);
    partial_sums(n.div_ceil(chunk), res, |w, local| {
        scatter(w * chunk..((w + 1) * chunk).min(n), local)
    })
}

// ---------------------------------------------------------------------------
// Test support
// ---------------------------------------------------------------------------

/// Helpers for the reference tests of the kernel modules: a small
/// deterministic random number generator (no dependency needed) and a way
/// to run a kernel on a pool of a given size.
#[cfg(test)]
pub mod testing {
    /// A xorshift64* generator, seeded explicitly so that failures are
    /// reproducible.
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Rng(seed.max(1))
        }

        pub fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }

        /// Uniform in `[0, 1)`.
        pub fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Uniform in `[-1, 1)` plus `offset`.
        pub fn value(&mut self, offset: f64) -> f64 {
            2.0 * self.uniform() - 1.0 + offset
        }

        /// Uniform in `0..n`.
        pub fn below(&mut self, n: usize) -> usize {
            (self.uniform() * n as f64) as usize
        }

        /// `n` values in `[-1, 1) + offset`.
        pub fn values(&mut self, n: usize, offset: f64) -> Vec<f64> {
            (0..n).map(|_| self.value(offset)).collect()
        }

        /// A sorted random subset of `0..n` of about half its size.
        pub fn subset(&mut self, n: usize) -> Vec<i32> {
            let mut v: Vec<i32> = (0..n as i32).filter(|_| self.uniform() < 0.5).collect();
            if v.is_empty() && n > 0 {
                v.push(self.below(n) as i32);
            }
            v.sort_unstable();
            v
        }
    }

    /// Runs `f` on a rayon pool of `n` threads.
    pub fn with_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(f)
    }

    /// The thread counts the reference tests run on.
    pub const THREADS: [usize; 3] = [1, 3, 8];

    /// Asserts that `a` and `b` agree to a relative tolerance of `tol` of
    /// the largest magnitude in `b`.
    pub fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        let scale = b.iter().fold(1.0f64, |m, &v| m.max(v.abs()));
        for (k, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * scale,
                "{what}: entry {k} differs: {x} vs {y} (scale {scale})"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_range_and_check() {
        assert_eq!(index_range::<i32>(&[]), (0, 0));
        assert_eq!(index_range(&[3i64, -1, 7, 2, 9, 9, 0, 1, 5, 4]), (-1, 9));
        assert!(check_indices(&[0i32, 1, 2], 3, "row").is_ok());
        assert_eq!(
            check_indices(&[0i32, 3], 3, "row"),
            Err(IndexError {
                what: "row",
                len: 3
            })
        );
        assert!(check_indices(&[-1i32], 3, "row").is_err());
        assert!(check_indices::<i32>(&[], 0, "row").is_ok());
    }

    #[test]
    fn category_column_shifts_and_skips() {
        assert_eq!(category_column::<false>(3, 0), Some(3));
        assert_eq!(category_column::<true>(3, 1), Some(2));
        assert_eq!(category_column::<true>(0, 1), None);
        assert_eq!(category_column::<true>(-1, 0), None);
    }

    #[test]
    fn partial_sums_accumulate_in_worker_order() {
        let mut out = vec![1.0f64; 10_000];
        partial_sums::<_, ()>(5, &mut out, |w, partial| {
            for (i, p) in partial.iter_mut().enumerate() {
                *p += (w * 10_000 + i) as f64;
            }
            Ok(())
        })
        .unwrap();
        for (i, &o) in out.iter().enumerate() {
            let expected: f64 = 1.0 + (0..5).map(|w| (w * 10_000 + i) as f64).sum::<f64>();
            assert_eq!(o, expected);
        }
        assert_eq!(
            partial_sums::<f64, &str>(3, &mut out, |w, _| if w == 1 { Err("bad") } else { Ok(()) }),
            Err("bad")
        );
    }

    #[test]
    fn simd_kernels_match_scalar() {
        use pulp::Arch;
        struct Check;
        impl pulp::WithSimd for Check {
            type Output = ();
            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) {
                let n = 37;
                let x: Vec<f64> = (0..n).map(|k| k as f64 * 0.5 + 1.0).collect();
                let y: Vec<f64> = (0..n).map(|k| (k % 7) as f64 - 3.0).collect();
                let c: Vec<f64> = (0..n).map(|k| (k % 3) as f64).collect();
                let dot_ref: f64 = (0..n).map(|k| (x[k] - 2.0) * y[k]).sum();
                assert!((centered_dot(simd, &x, 2.0, &y) - dot_ref).abs() < 1e-9);
                let dot_vc_ref: f64 = (0..n).map(|k| (x[k] - c[k]) * y[k]).sum();
                assert!((centered_dot_vc(simd, &x, &c, &y) - dot_vc_ref).abs() < 1e-9);
                let mut out = y.clone();
                axpy(simd, 1.5, &x, &mut out);
                assert!((0..n).all(|k| (out[k] - (y[k] + 1.5 * x[k])).abs() < 1e-12));
                let mut out = y.clone();
                centered_axpy(simd, &x, 2.0, 1.5, &mut out);
                assert!((0..n).all(|k| (out[k] - (y[k] + (x[k] - 2.0) * 1.5)).abs() < 1e-12));
                let mut out = y.clone();
                centered_axpy_vc(simd, &x, &c, 1.5, &mut out);
                assert!((0..n).all(|k| (out[k] - (y[k] + (x[k] - c[k]) * 1.5)).abs() < 1e-12));
                let mut out = y.clone();
                centered_axpy4(
                    simd,
                    [&x, &y, &c, &x],
                    [1.0, 0.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0, 4.0],
                    &mut out,
                );
                assert!((0..n).all(|k| {
                    let e =
                        y[k] + (x[k] - 1.0) + 2.0 * y[k] + 3.0 * (c[k] - 2.0) + 4.0 * (x[k] - 3.0);
                    (out[k] - e).abs() < 1e-12
                }));
            }
        }
        Arch::new().dispatch(Check);
    }

    #[test]
    fn centered_gather_and_pack() {
        let data: Vec<f64> = (0..12).map(|v| v as f64).collect();
        let x = ArrayView2::from_shape((4, 3), &data).unwrap();
        let center = [1.0, 2.0, 3.0];
        let d = [1.0, 10.0, 100.0, 1000.0];
        let op = Centered {
            x,
            center: Some(&center),
        };
        let mut dst = [0.0; 2];
        unsafe { op.gather_column::<i32, true>(1, &[3, 0], &d, &mut dst) };
        assert_eq!(dst, [1000.0 * (10.0 - 2.0), 1.0 * (1.0 - 2.0)]);
        let mut block = [0.0; 4];
        unsafe { op.pack_block(&[1i64, 2], &[0i64, 2], &d, &center, &mut block, 2) };
        assert_eq!(block, [10.0 * 2.0, 10.0 * 2.0, 100.0 * 5.0, 100.0 * 5.0]);
        let plain = Centered { x, center: None };
        assert_eq!(plain.center_of(2), 0.0);
        assert_eq!(&*plain.center_vec(), &[0.0; 3]);
    }

    #[test]
    fn mirror_fills_upper_triangle() {
        for m in [3usize, 150] {
            let mut out: Vec<f64> = (0..m * m)
                .map(|k| if k / m >= k % m { k as f64 } else { 0.0 })
                .collect();
            mirror_lower_triangle(ArrayViewMut2::from_shape((m, m), &mut out[..]).unwrap());
            for i in 0..m {
                for j in 0..m {
                    assert_eq!(out[i * m + j], out[j * m + i]);
                }
            }
        }
    }

    #[test]
    fn scatter_add_matches_serial() {
        let n = 100_000;
        let idx: Vec<usize> = (0..n).map(|i| (i * 7919) % 1000).collect();
        let mut serial = vec![0.0f64; 1000];
        for &i in &idx {
            serial[i] += 1.0;
        }
        for threads in testing::THREADS {
            let mut res = vec![0.0f64; 1000];
            testing::with_threads(threads, || {
                scatter_add::<_, ()>(n, &mut res, |r, res| {
                    for &i in &idx[r] {
                        res[i] += 1.0;
                    }
                    Ok(())
                })
                .unwrap()
            });
            assert_eq!(res, serial);
        }
    }
}
