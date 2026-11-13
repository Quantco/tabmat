//! The kernels: plain Rust over slices and `ndarray` views, with no PyO3 or
//! numpy types, so that they can be read, tested and benchmarked on their
//! own. The binding modules next to this directory extract the Python
//! arguments and call them. This module holds what the kernel families share:
//! the element type traits, index validation, the categorical index helper,
//! the thread pool setup and the deterministic parallel reductions.
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
//! reads anyway (the categories of a categorical matrix). `unsafe` is used
//! only for the unchecked loads that follow such a validation, each with a
//! `SAFETY` comment naming it.
//!
//! The kernels run on rayon's global thread pool (the bindings release the
//! GIL around them). Where threads accumulate into the same output, each gets
//! a private copy that is summed afterwards in thread order
//! ([`try_partial_sums`], and [`row_partials`] over row ranges): the result then only depends on the number of threads,
//! not on their timing (the C++ kernels used atomic adds). Measured against
//! rayon's own `fold`/`reduce` this costs nothing and is deterministic by
//! construction.
//!
//! Each kernel module carries a `#[cfg(test)]` reference implementation
//! written as the plainest possible loops, and `cargo test` checks the
//! kernels against it over random shapes, subsets, layouts and thread
//! counts, and checks that repeated runs agree bit for bit.

pub mod categorical;

use std::ops::{Add, AddAssign, Mul, Range, Sub};

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

impl IndexError {
    /// The error for an index into `what`, an axis of length `len`.
    pub const fn new(what: &'static str, len: usize) -> Self {
        Self { what, len }
    }
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
        Err(IndexError::new(what, len))
    }
}

/// Length of the sub-chunks in which a kernel validates its index array
/// before using it unchecked: small enough to stay in L1, so that the
/// validation pass costs no memory traffic, large enough to amortise its
/// setup.
pub const VALIDATE_CHUNK: usize = 1024;

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

/// The column a row with category `index` contributes to: `Ok(None)` for
/// a skipped row ([`category_column`]), `Err(err)` for a column at or
/// beyond `n_cols`. This is the one check the scatter kernels do per row,
/// and the comparison against `n_cols` is also what lets the compiler drop
/// the bounds check of the scatter into a result sliced to `n_cols`
/// beforehand (`&mut res[..n_cols]` in the callers, which is why those
/// reslices are there).
///
/// `err` is an [`IndexError`] built once outside the loop and passed in by
/// value: this inlines to the same dozen instructions per row as the
/// hand-written compare (checked against the assembly). Building the error
/// in the loop instead, as `res.get(col).ok_or(IndexError { .. })?`,
/// measured up to 30% slower on cache-resident results, and validating
/// gathered chunks of categories up front up to 2x slower.
#[inline(always)]
pub fn checked_category<const DROP: bool>(
    index: i32,
    drop_first: i32,
    n_cols: usize,
    err: IndexError,
) -> Result<Option<usize>, IndexError> {
    match category_column::<DROP>(index, drop_first) {
        Some(col) if col >= n_cols => Err(err),
        col => Ok(col),
    }
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
/// then what it would have produced on `out` directly. The first error in
/// worker order is returned and `out` is left untouched.
///
/// The summation order, and hence the result, only depends on
/// `n_workers`, never on thread timing. This is the one way the kernels
/// combine per-thread results: a static split of the work over the workers
/// and this reduction in a fixed order.
pub fn try_partial_sums<F: Float, E: Send>(
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

/// Below this many rows [`row_partials`] runs on one thread: the per-thread
/// result copies and the thread wake-up would cost more than they save.
pub const PARALLEL_MIN_ROWS: usize = 1 << 14;

/// [`try_partial_sums`] over the rows `0..n`: calls `accumulate(rows, res)`,
/// which must only add into `res` what the rows in `rows` contribute, on
/// sub-ranges that together cover `0..n` once, one contiguous range per
/// thread, each into a private copy of `res` that is summed back in range
/// order. This is how the kernels that scatter row contributions into a
/// result (the categorical products) run in parallel without atomics and
/// with a result that only depends on the thread count.
///
/// It runs on one thread, directly on `res`, when there are fewer than
/// [`PARALLEL_MIN_ROWS`] rows, when the pool has one thread, or when `res`
/// has more entries than there are rows: zeroing and summing a copy of
/// `res` per thread would then cost more than the rows themselves (measured
/// 30x on a 1e5 x 1e3 categorical cross product over 1e6 rows).
///
/// `accumulate` should capture by value (a `move` closure): with captured
/// references the compiler reloaded every invariant (lengths, the data
/// pointers) from memory on each row, which measured 1.3-1.8x slower at
/// one thread.
pub fn row_partials<F: Float, E: Send>(
    n: usize,
    res: &mut [F],
    accumulate: impl Fn(Range<usize>, &mut [F]) -> Result<(), E> + Sync,
) -> Result<(), E> {
    let n_threads = rayon::current_num_threads();
    if n < PARALLEL_MIN_ROWS || n_threads == 1 || res.len() > n {
        return accumulate(0..n, res);
    }
    let chunk = n.div_ceil(n_threads);
    try_partial_sums(n.div_ceil(chunk), res, |w, local| {
        accumulate(w * chunk..((w + 1) * chunk).min(n), local)
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
            Err(IndexError::new("row", 3))
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
        try_partial_sums::<f64, ()>(5, &mut out, |w, partial| {
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
            try_partial_sums::<f64, &str>(3, &mut out, |w, _| if w == 1 {
                Err("bad")
            } else {
                Ok(())
            }),
            Err("bad")
        );
    }

    #[test]
    fn row_partials_match_serial() {
        let n = 100_000;
        let idx: Vec<usize> = (0..n).map(|i| (i * 7919) % 1000).collect();
        let mut serial = vec![0.0f64; 1000];
        for &i in &idx {
            serial[i] += 1.0;
        }
        for threads in testing::THREADS {
            let mut res = vec![0.0f64; 1000];
            testing::with_threads(threads, || {
                row_partials::<_, ()>(n, &mut res, |r, res| {
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
