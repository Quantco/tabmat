//! Kernels for `tabmat.DenseMatrix`: so far the sandwich product
//! `X.T @ diag(d) @ X`.
//!
//! `X` is read in place whatever its layout: the kernels index it through
//! an `ndarray` view with runtime strides, so C order, Fortran order and
//! non-contiguous arrays are all accepted without a copy. It may be
//! `float32` or `float64`, with `rows` and `cols` index arrays selecting a
//! row and column subset. The product takes an optional per-column
//! `center` and computes with `X - center` ([`Centered`]), subtracting it
//! as the data is read: `StandardizedMatrix` relies on this to stay accurate
//! for columns whose mean is large relative to their spread. A zero center
//! gives the uncentered product exactly.
//!
//! # The sandwich product
//!
//! `dense_sandwich` is a BLIS/GotoBLAS-style blocked kernel for
//! `out = X[rows, cols].T @ diag(d[rows]) @ X[rows, cols]`, computing only the
//! lower triangle of `out` and mirroring it at the end. With the defaults
//! `thresh1d = 32`, `kratio = 16`, `innerblock = 128` and
//! `kstep = kratio * thresh1d = 512`:
//!
//! - for each block of `kstep` columns of `out` (`Cj`) and each block of
//!   `kstep` selected rows of `X` (`Rk`), the d-scaled, centered columns of
//!   that block are packed k-contiguously into the `R` panel (one
//!   `kstep`-strided column per `out` column);
//! - for each block of `thresh1d` rows of `out` at or below the diagonal
//!   (`Ci >= Cj`) the centered columns are packed the same way into an `L`
//!   panel, and `dense_base` multiplies the panels in `innerblock`-sized
//!   cache blocks;
//! - each cache block is tiled with register tiles of `IB x JB` accumulators
//!   (4x4 where it fits, 2 and 1 for the remainders), see [`dense_tile`]. The
//!   sweep over tile columns is clipped at `i + IB` so that only the lower
//!   triangle is accumulated.
//!
//! The kernel is written as three stages: the block walk ([`walk`] over a
//! [`SandwichPlan`] of the `(Cj, Rk, Ci)` blocks), the packing of the
//! panels ([`PanelSource`]) and the arithmetic ([`dense_base`] and the
//! tile). The walk is the same whichever level the threads split
//! ([`Split`]), which depends on the shape: when `X` is tall and narrow
//! (`in_n / kstep > out_m / thresh1d`, "k-parallel") the threads split the
//! `Rk` blocks statically, each walking its share with its own `R` and `L`
//! panels into its own partial `out`, and the partials are summed at the end
//! in worker order ([`partial_sums`]); otherwise ("column parallel") the
//! `R` panel is shared and packed in parallel, and the threads split the row
//! blocks `Ci`, each with its own `L` panel from a pool. The C++ version
//! that this replaces accumulated the k-parallel partial sums into one
//! `out` with atomic adds; per-thread partials make the result independent
//! of thread timing at the cost of one `out`-sized buffer per thread.
//!
//! The register tile is written against the `pulp` crate: portable SIMD
//! with runtime dispatch (NEON on aarch64, SSE/AVX2 chosen at run time on
//! x86-64, scalar otherwise) on stable Rust, so one tile serves every target
//! like the xsimd tile did. Scratch panels are plain `Vec`s: the vector
//! loads need no alignment, and a paired measurement showed 64-byte aligned
//! buffers to make no difference.

use std::ops::Range;
use std::sync::Mutex;

use ndarray::ArrayViewMut2;
use pulp::{Arch, Simd, WithSimd};
use rayon::prelude::*;

use super::{mirror_lower_triangle, partial_sums, Blocks, Centered, Float, PanelSource, SimdFloat};

// ---------------------------------------------------------------------------
// Sandwich: the register tile
// ---------------------------------------------------------------------------

/// The packed panels of one `(Rk, Cj, Ci)` block.
///
/// `l` holds the centered `X` columns for the rows of `out` from `imin2`,
/// `r` the d-scaled centered columns for the columns of `out` from `jmin2`.
/// Both are k-contiguous with `kstep` elements between columns, of which
/// the first `klen` are in use.
#[derive(Clone, Copy)]
struct Panels<'a, F> {
    r: &'a [F],
    l: &'a [F],
    imin2: usize,
    jmin2: usize,
    klen: usize,
    kstep: usize,
}

impl<F> Panels<'_, F> {
    #[inline(always)]
    fn lcol(&self, i: usize) -> &[F] {
        &self.l[(i - self.imin2) * self.kstep..][..self.klen]
    }
    #[inline(always)]
    fn rcol(&self, j: usize) -> &[F] {
        &self.r[(j - self.jmin2) * self.kstep..][..self.klen]
    }
}

/// The rows `[row0, row0 + data.len() / out_m)` of a row-major `out`
/// matrix with `out_m` columns.
struct OutBlock<'a, F> {
    data: &'a mut [F],
    out_m: usize,
    row0: usize,
}

impl<F: Float> OutBlock<'_, F> {
    #[inline(always)]
    fn add(&mut self, i: usize, j: usize, v: F) {
        self.data[(i - self.row0) * self.out_m + j] += v;
    }
}

/// The micro-kernel: accumulates the `IB x JB` tile of
/// `out[i:i+IB, j:j+JB]` over the `klen` packed rows of the panels.
///
/// This is `dense_tile` of the C++ version, instruction for instruction:
/// `acc[ir][jr]` is the `IB x JB` array of `accumsimd` vector registers;
/// the main loop loads one vector of each of the `IB` `L` columns and the
/// `JB` `R` columns per step and issues `IB * JB` fused multiply-adds (16
/// for the 4x4 tile); the vectors are then summed horizontally, the scalar
/// remainder of `klen` that does not fill a vector is added, and the tile is
/// added to `out`. Unlike the C++, the add into `out` is never atomic: the
/// k-parallel driver gives every thread its own `out`.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // index loops on purpose: they mirror the C++ tile
fn dense_tile<S: Simd, F: SimdFloat, const IB: usize, const JB: usize>(
    simd: S,
    p: Panels<'_, F>,
    out: &mut OutBlock<'_, F>,
    i: usize,
    j: usize,
) {
    let lcols: [&[F]; IB] = std::array::from_fn(|ir| p.lcol(i + ir));
    let rcols: [&[F]; JB] = std::array::from_fn(|jr| p.rcol(j + jr));

    // The columns all have `klen` elements, so the vector parts all have the
    // same length; slicing them to it lets the compiler drop the bounds
    // checks in the loop below.
    let lv: [&[F::Vec<S>]; IB] = lcols.map(|c| F::as_simd::<S>(c).0);
    let rv: [&[F::Vec<S>]; JB] = rcols.map(|c| F::as_simd::<S>(c).0);
    let nvec = lv[0].len();
    let lv = lv.map(|v| &v[..nvec]);
    let rv = rv.map(|v| &v[..nvec]);

    // main simd inner loop
    let mut acc = [[F::zero(simd); JB]; IB];
    for k in 0..nvec {
        for ir in 0..IB {
            let xl = lv[ir][k];
            for jr in 0..JB {
                acc[ir][jr] = F::mul_add(simd, xl, rv[jr][k], acc[ir][jr]);
            }
        }
    }

    // horizontal sum of the simd blocks
    let mut sums = [[F::default(); JB]; IB];
    for ir in 0..IB {
        for jr in 0..JB {
            sums[ir][jr] = F::reduce_sum(simd, acc[ir][jr]);
        }
    }

    // remainder loop handling the entries that can't be handled in a
    // simd_size stride
    for k in nvec * F::lanes::<S>()..p.klen {
        for ir in 0..IB {
            let xl = lcols[ir][k];
            for jr in 0..JB {
                sums[ir][jr] += xl * rcols[jr][k];
            }
        }
    }

    // add to the output array
    for ir in 0..IB {
        for jr in 0..JB {
            out.add(i + ir, j + jr, sums[ir][jr]);
        }
    }
}

/// Sweeps `j` over `[jmin, jmaxinner)` in steps of `JB` for the rows
/// `i..i+IB`, starting from the current `j` and advancing it. The remainder
/// that does not fit a `JB`-wide tile is left for the caller's next,
/// narrower `JB`.
#[inline(always)]
fn dense_tile_cols<S: Simd, F: SimdFloat, const IB: usize, const JB: usize>(
    simd: S,
    p: Panels<'_, F>,
    out: &mut OutBlock<'_, F>,
    i: usize,
    j: &mut usize,
    jmin: usize,
    jmaxinner: usize,
) {
    let jmaxblock = jmin + ((jmaxinner - jmin) / JB) * JB;
    while *j < jmaxblock {
        dense_tile::<S, F, IB, JB>(simd, p, out, i, *j);
        *j += JB;
    }
}

/// Sweeps `i` over `[imin, imax)` in steps of `IB`, starting from the
/// current `i` and advancing it. Only the lower triangle (`j <= i`) of `out`
/// is computed, so the `j` sweep is clipped at `i + IB`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dense_tile_rows<S: Simd, F: SimdFloat, const IB: usize>(
    simd: S,
    p: Panels<'_, F>,
    out: &mut OutBlock<'_, F>,
    i: &mut usize,
    imin: usize,
    imax: usize,
    jmin: usize,
    jmax: usize,
) {
    let imaxblock = imin + ((imax - imin) / IB) * IB;
    while *i < imaxblock {
        let jmaxinner = jmax.min(*i + IB);
        let mut j = jmin;
        dense_tile_cols::<S, F, IB, 4>(simd, p, out, *i, &mut j, jmin, jmaxinner);
        dense_tile_cols::<S, F, IB, 2>(simd, p, out, *i, &mut j, jmin, jmaxinner);
        dense_tile_cols::<S, F, IB, 1>(simd, p, out, *i, &mut j, jmin, jmaxinner);
        *i += IB;
    }
}

/// Computes the lower-triangular part of the `out` block
/// `[imin2, imax2) x [jmin2, jmax2)` from the packed panels, in
/// `innerblock`-sized cache blocks, each tiled with the widest register tile
/// that fits.
#[inline(always)]
fn dense_base<S: Simd, F: SimdFloat>(
    simd: S,
    p: Panels<'_, F>,
    out: &mut OutBlock<'_, F>,
    rows: Range<usize>,
    cols: Range<usize>,
    innerblock: usize,
) {
    for imin in rows.clone().step_by(innerblock) {
        let imax = (imin + innerblock).min(rows.end);
        for jmin in cols.clone().step_by(innerblock) {
            let jmax = (jmin + innerblock).min(cols.end);
            let mut i = imin;
            dense_tile_rows::<S, F, 4>(simd, p, out, &mut i, imin, imax, jmin, jmax);
            dense_tile_rows::<S, F, 2>(simd, p, out, &mut i, imin, imax, jmin, jmax);
            dense_tile_rows::<S, F, 1>(simd, p, out, &mut i, imin, imax, jmin, jmax);
        }
    }
}

/// The arguments of [`dense_base`], as the operation that `pulp` dispatches
/// to the instruction set detected at run time. Everything below the
/// dispatch is `#[inline(always)]`, so the tile is compiled once per
/// instruction set with that set's vector width and FMA.
struct DenseBase<'a, 'o, F> {
    p: Panels<'a, F>,
    out: &'a mut OutBlock<'o, F>,
    rows: Range<usize>,
    cols: Range<usize>,
    innerblock: usize,
}

impl<F: SimdFloat> WithSimd for DenseBase<'_, '_, F> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        dense_base(
            simd,
            self.p,
            self.out,
            self.rows,
            self.cols,
            self.innerblock,
        );
    }
}

// ---------------------------------------------------------------------------
// Sandwich: the block walk
// ---------------------------------------------------------------------------

/// The block plan of the sandwich product: the column blocks `Cj` of `out`
/// and the blocks `Rk` of the selected rows of `X`, both `kstep` long, and,
/// for each `Cj`, the row blocks `Ci` of `out` of `thresh1d` rows from the
/// diagonal down ([`Self::out_rows`]). The walk visits the
/// `(Cj, Rk, Ci)` triples with `Cj` outermost.
#[derive(Clone, Copy)]
struct SandwichPlan {
    out_cols: Blocks,
    x_rows: Blocks,
    out_m: usize,
    thresh1d: usize,
    kstep: usize,
    innerblock: usize,
}

impl SandwichPlan {
    /// The row blocks `Ci` of `out` at or below the diagonal block of `cj`;
    /// `kstep` is a multiple of `thresh1d`, so they are the global
    /// `thresh1d`-blocks of the rows of `out` from `cj.start` on.
    fn out_rows(&self, cj: &Range<usize>) -> Blocks {
        Blocks::new(cj.start..self.out_m, self.thresh1d)
    }
}

/// Which level of the sandwich walk the threads split.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Split {
    /// "k-parallel", for a tall and narrow `X`: every worker takes a
    /// static share of the `Rk` blocks ([`Blocks::part`]) and walks the
    /// rest of the plan alone, with its own panels and into its own copy
    /// of `out`; the copies are summed in worker order ([`partial_sums`]).
    XRows,
    /// "column parallel": one walk, in which the columns of each `R` panel
    /// are packed in parallel and the `Ci` blocks of each `(Cj, Rk)` step,
    /// which own disjoint rows of `out`, are split between the threads,
    /// each taking an `L` panel source from a pool.
    OutRows,
}

/// A free list of scratch objects for a parallel loop: an iteration takes
/// one, uses it and gives it back, so that one object per concurrently
/// running thread is made (by `make`) per call and reused. The sandwich
/// keeps its `L` panel sources here.
struct PanelPool<T, M> {
    free: Mutex<Vec<T>>,
    make: M,
}

impl<T, M: Fn() -> T> PanelPool<T, M> {
    fn new(make: M) -> Self {
        Self {
            free: Mutex::new(Vec::new()),
            make,
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let popped = self.free.lock().unwrap().pop();
        let mut item = popped.unwrap_or_else(&self.make);
        let result = f(&mut item);
        self.free.lock().unwrap().push(item);
        result
    }
}

/// The walk of the sandwich over the `Rk` blocks `x_rows` (all of them, or
/// one worker's share): for each `Cj` and each `Rk`, pack the d-scaled `R`
/// panel of `(Rk, Cj)`, then for each `Ci` from `Cj` down pack the `L`
/// panel of `(Rk, Ci)` and accumulate the lower triangle of `out[Ci, Cj]`
/// over `Rk` with [`dense_base`]. With [`Split::OutRows`] the `R` packing
/// and the `Ci` loop run in parallel; the order in which the `Rk` blocks
/// contribute to an entry of `out` is the same either way.
fn walk<'x, 'a, F: SimdFloat, M>(
    arch: Arch,
    plan: &SandwichPlan,
    x_rows: Blocks,
    split: Split,
    r: &mut PanelSource<'x, 'a, i32, F>,
    l_pool: &PanelPool<PanelSource<'x, 'a, i32, F>, M>,
    out: &mut [F],
) where
    M: Fn() -> PanelSource<'x, 'a, i32, F> + Sync,
{
    let SandwichPlan {
        out_m,
        thresh1d,
        kstep,
        innerblock,
        ..
    } = *plan;
    let parallel = split == Split::OutRows;
    for cj in plan.out_cols.iter() {
        let out_rows = plan.out_rows(&cj);
        for rk in x_rows.iter() {
            let r = if parallel {
                r.fill_par(rk.clone(), cj.clone())
            } else {
                r.fill(rk.clone(), cj.clone())
            };
            let row_block = |(ci, data): (Range<usize>, &mut [F])| {
                l_pool.with(|l| {
                    let l = l.fill(rk.clone(), ci.clone());
                    arch.dispatch(DenseBase {
                        p: Panels {
                            r,
                            l,
                            imin2: ci.start,
                            jmin2: cj.start,
                            klen: rk.len(),
                            kstep,
                        },
                        out: &mut OutBlock {
                            data,
                            out_m,
                            row0: ci.start,
                        },
                        rows: ci,
                        cols: cj.clone(),
                        innerblock,
                    });
                });
            };
            // The `Ci` blocks own consecutive `thresh1d`-row slices of `out`
            // from row `cj.start` on.
            let out = &mut out[cj.start * out_m..];
            if parallel {
                out_rows
                    .par_iter()
                    .zip(out.par_chunks_mut(thresh1d * out_m))
                    .for_each(row_block);
            } else {
                out_rows
                    .iter()
                    .zip(out.chunks_mut(thresh1d * out_m))
                    .for_each(row_block);
            }
        }
    }
}

/// `out[cols, cols] = (X - center)[rows, cols].T @ diag(d[rows]) @ (X - center)[rows, cols]`.
/// `out` is `out_m x out_m`, row-major and zeroed; the indices have been
/// checked against the shape of `X`.
#[allow(clippy::too_many_arguments)]
pub fn sandwich<F: SimdFloat>(
    x: Centered<'_, F>,
    d: &[F],
    rows: &[i32],
    cols: &[i32],
    out: &mut [F],
    thresh1d: usize,
    kratio: usize,
    innerblock: usize,
) {
    let (in_n, out_m) = (rows.len(), cols.len());
    let kstep = kratio * thresh1d;
    let plan = SandwichPlan {
        out_cols: Blocks::new(0..out_m, kstep),
        x_rows: Blocks::new(0..in_n, kstep),
        out_m,
        thresh1d,
        kstep,
        innerblock,
    };
    let split = if in_n / kstep > out_m / thresh1d {
        Split::XRows
    } else {
        Split::OutRows
    };
    let arch = Arch::new();
    // One walk over the given Rk blocks with its own panel sources: an R
    // panel holds the columns of one Cj block, an L panel those of one Ci
    // block.
    let run = move |x_rows: Blocks, out: &mut [F]| {
        // SAFETY: `rows` and `cols` were checked against the shape of `X` at
        // the Python boundary, and `d` has an entry per row.
        let mut r =
            unsafe { PanelSource::columns(x, rows, cols, Some(d), kstep, out_m.min(kstep)) };
        // SAFETY: as for `r`.
        let l_pool = PanelPool::new(|| unsafe {
            PanelSource::columns(x, rows, cols, None, kstep, thresh1d)
        });
        walk(arch, &plan, x_rows, split, &mut r, &l_pool, out);
    };
    match split {
        Split::XRows => {
            let n_workers = plan.x_rows.len().min(rayon::current_num_threads()).max(1);
            partial_sums(n_workers, out, |w, out| {
                run(plan.x_rows.part(w, n_workers), out)
            });
        }
        Split::OutRows => run(plan.x_rows, out),
    }

    mirror_lower_triangle(ArrayViewMut2::from_shape((out_m, out_m), out).expect("square"));
}

#[cfg(test)]
mod tests {
    //! The kernel against a plain reference loop, over random shapes, both
    //! layouts and a non-contiguous view, row and column subsets, with and
    //! without a center, and several thread counts.

    use ndarray::{Array2, ArrayView2};

    use super::super::testing::{assert_close, with_threads, Rng, THREADS};
    use super::*;

    fn reference_sandwich(
        x: ArrayView2<'_, f64>,
        d: &[f64],
        rows: &[i32],
        cols: &[i32],
        center: Option<&[f64]>,
    ) -> Vec<f64> {
        let m = cols.len();
        let mut out = vec![0.0; m * m];
        let mu = |j: usize| center.map_or(0.0, |c| c[j]);
        for (ci, &i) in cols.iter().enumerate() {
            for (cj, &j) in cols.iter().enumerate() {
                let mut sum = 0.0;
                for &k in rows {
                    let k = k as usize;
                    sum += d[k]
                        * (x[[k, i as usize]] - mu(i as usize))
                        * (x[[k, j as usize]] - mu(j as usize));
                }
                out[ci * m + cj] = sum;
            }
        }
        out
    }

    /// The layouts a kernel is tested in: C order, Fortran order, and a
    /// view with a stride in both directions.
    fn layouts(data: &Array2<f64>) -> Vec<(&'static str, Array2<f64>)> {
        let f = data.t().as_standard_layout().t().to_owned();
        let mut padded = Array2::<f64>::zeros((data.nrows() * 2, data.ncols() * 3));
        padded.slice_mut(ndarray::s![..;2, ..;3]).assign(data);
        vec![("C", data.clone()), ("F", f), ("strided", padded)]
    }

    fn view<'a>(name: &str, a: &'a Array2<f64>) -> ArrayView2<'a, f64> {
        if name == "strided" {
            a.slice(ndarray::s![..;2, ..;3])
        } else {
            a.view()
        }
    }

    #[test]
    fn sandwich_matches_reference() {
        let mut rng = Rng::new(11);
        // (n, m): a tall k-parallel shape, a column-parallel one, and odd sizes
        // that exercise the 2x2 and 1x1 remainder tiles.
        for &(n, m) in &[(3000usize, 7usize), (600, 70), (131, 37)] {
            let data = Array2::from_shape_vec((n, m), rng.values(n * m, 5.0)).unwrap();
            let d = rng.values(n, 1.5);
            let center: Vec<f64> = (0..m).map(|j| data.column(j).mean().unwrap()).collect();
            let all_rows: Vec<i32> = (0..n as i32).collect();
            let all_cols: Vec<i32> = (0..m as i32).collect();
            let sub_rows = rng.subset(n);
            let sub_cols = rng.subset(m);
            for (name, arr) in layouts(&data) {
                let x = view(name, &arr);
                for (rows, cols) in [(&all_rows, &all_cols), (&sub_rows, &sub_cols)] {
                    for center in [None, Some(&center[..])] {
                        let expected = reference_sandwich(x, &d, rows, cols, center);
                        let op = Centered { x, center };
                        for threads in THREADS {
                            let mut first = None;
                            for _ in 0..2 {
                                let mut out = vec![0.0; cols.len() * cols.len()];
                                with_threads(threads, || {
                                    sandwich(op, &d, rows, cols, &mut out, 32, 16, 128)
                                });
                                assert_close(
                                    &out,
                                    &expected,
                                    1e-11,
                                    &format!("sandwich {name} {n}x{m}"),
                                );
                                match &first {
                                    None => first = Some(out),
                                    Some(f) => assert!(
                                        *f == out,
                                        "run-to-run difference at {threads} threads"
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
