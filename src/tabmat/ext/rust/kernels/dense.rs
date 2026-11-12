//! Kernels for `tabmat.DenseMatrix`: the sandwich product
//! `X.T @ diag(d) @ X`, the matrix-vector products `X[rows, cols] @ v` and
//! `X[rows, cols].T @ v`, and the weighted column sums of squares used for
//! standardisation.
//!
//! `X` is read in place whatever its layout: the kernels index it through
//! an `ndarray` view with runtime strides, so C order, Fortran order and
//! non-contiguous arrays are all accepted without a copy. It may be
//! `float32` or `float64`, with `rows` and `cols` index arrays selecting a
//! row and column subset. All three products take an optional per-column
//! `center` and compute with `X - center` ([`Centered`]), subtracting it
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
//! Which loop the threads split depends on the shape: when `X` is tall and
//! narrow (`in_n / kstep > out_m / thresh1d`, "k-parallel") the threads split
//! the `Rk` blocks statically, each with its own `R` and `L` panels and its
//! own partial `out`, and the partials are summed at the end in worker order
//! ([`partial_sums`]); otherwise ("column parallel") the `R` panel is shared
//! and packed in parallel, and the threads split the row blocks `Ci`, each
//! with its own `L` panel from a pool. The C++ version that this replaces
//! accumulated the k-parallel partial sums into one `out` with atomic adds;
//! per-thread partials make the result independent of thread timing at the
//! cost of one `out`-sized buffer per thread.
//!
//! The register tile is written against the `pulp` crate: portable SIMD
//! with runtime dispatch (NEON on aarch64, SSE/AVX2 chosen at run time on
//! x86-64, scalar otherwise) on stable Rust, so one tile serves every target
//! like the xsimd tile did. Scratch panels are plain `Vec`s: the vector
//! loads need no alignment, and a paired measurement showed 64-byte aligned
//! buffers to make no difference.
//!
//! # The matrix-vector products
//!
//! `dense_matvec` and `dense_rmatvec` take `rows` and `cols` as `None` for
//! "all of them". The inner loops run along the contiguous direction of `X`
//! (down a column for Fortran order, along a row for C order) and are
//! vectorised with the shared centered dot and axpy kernels when that
//! direction is not restricted; otherwise they fall back to indexed loads.
//! Row blocks are split between the threads (with partial results summed in
//! worker order for `rmatvec`, whose output is shared); small products run
//! on one thread, since starting the threads would cost more than the
//! product.

use std::ops::Range;
use std::sync::Mutex;

use ndarray::ArrayViewMut2;
use pulp::{Arch, Simd, WithSimd};
use rayon::prelude::*;

use super::{
    centered_axpy, centered_axpy4, centered_axpy_vc, centered_dot, centered_dot_vc,
    mirror_lower_triangle, partial_sums, Centered, Float, SimdFloat,
};

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
// Sandwich: packing and the blocked drivers
// ---------------------------------------------------------------------------

/// Packs the `R` panel for the `out` columns `cols` and the rows `rk`.
#[inline(always)]
fn pack_r<F: Float>(
    x: Centered<'_, F>,
    d: &[F],
    rows: &[i32],
    cols: &[i32],
    rk: Range<usize>,
    r: &mut [F],
    kstep: usize,
) {
    for (rcol, &jj) in r.chunks_mut(kstep).zip(cols) {
        // SAFETY: `rows` and `cols` were checked against the shape of `X`
        // at the Python boundary, and `d` has an entry per row.
        unsafe { x.gather_column::<i32, true>(jj as usize, &rows[rk.clone()], d, rcol) };
    }
}

/// The shape of the sandwich product and its blocking parameters.
#[derive(Clone, Copy)]
struct Blocking {
    in_n: usize,
    out_m: usize,
    thresh1d: usize,
    innerblock: usize,
    kstep: usize,
}

/// Packs the `L` panel for the `out` rows `[ci, ci + thresh1d)` into `l` and
/// multiplies it against the `R` panel of the `out` columns `[cj, cjmax2)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dense_row_block<F: SimdFloat>(
    arch: Arch,
    x: Centered<'_, F>,
    rows: &[i32],
    cols: &[i32],
    b: Blocking,
    r: &[F],
    l: &mut [F],
    out: &mut OutBlock<'_, F>,
    ci: usize,
    cj: usize,
    cjmax2: usize,
    rk: Range<usize>,
) {
    let cimax2 = (ci + b.thresh1d).min(b.out_m);
    for (lcol, &ii) in l.chunks_mut(b.kstep).zip(&cols[ci..cimax2]) {
        // SAFETY: as in `pack_r`.
        unsafe { x.gather_column::<i32, false>(ii as usize, &rows[rk.clone()], &[], lcol) };
    }
    arch.dispatch(DenseBase {
        p: Panels {
            r,
            l,
            imin2: ci,
            jmin2: cj,
            klen: rk.len(),
            kstep: b.kstep,
        },
        out,
        rows: ci..cimax2,
        cols: cj..cjmax2,
        innerblock: b.innerblock,
    });
}

/// A free list of `L` panels for the column-parallel row-block loop: a row
/// block takes a panel, uses it, and gives it back, so one panel per
/// concurrently running thread is allocated per call and reused.
struct PanelPool<F: Float> {
    free: Mutex<Vec<Vec<F>>>,
    len: usize,
}

impl<F: Float> PanelPool<F> {
    fn new(len: usize) -> Self {
        Self {
            free: Mutex::new(Vec::new()),
            len,
        }
    }

    fn with(&self, f: impl FnOnce(&mut [F])) {
        let mut buf = self
            .free
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| vec![F::default(); self.len]);
        f(&mut buf);
        self.free.lock().unwrap().push(buf);
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
    let b = Blocking {
        in_n: rows.len(),
        out_m: cols.len(),
        thresh1d,
        innerblock,
        kstep: kratio * thresh1d,
    };
    let Blocking {
        in_n, out_m, kstep, ..
    } = b;
    let arch = Arch::new();
    // An R panel holds the columns of one kstep-wide block of out.
    let r_len = out_m.min(kstep) * kstep;
    let l_len = thresh1d * kstep;

    let kparallel = in_n / kstep > out_m / thresh1d;
    if kparallel {
        // Each worker takes a contiguous range of the Rk blocks (OpenMP's
        // static schedule) and accumulates into its own copy of out.
        let n_blocks = in_n.div_ceil(kstep);
        let n_workers = n_blocks.min(rayon::current_num_threads()).max(1);
        // Worker w gets the blocks [w * n_blocks / n_workers, (w + 1) * ...),
        // i.e. floor or ceil of the average, every worker busy.
        let worker_blocks = |w: usize| w * n_blocks / n_workers..(w + 1) * n_blocks / n_workers;
        partial_sums::<F, ()>(n_workers, out, |w, out| {
            let mut r = vec![F::default(); r_len];
            let mut l = vec![F::default(); l_len];
            let mut out = OutBlock {
                data: out,
                out_m,
                row0: 0,
            };
            for cj in (0..out_m).step_by(kstep) {
                let cjmax2 = (cj + kstep).min(out_m);
                for block in worker_blocks(w) {
                    let rk = block * kstep..((block + 1) * kstep).min(in_n);
                    pack_r(x, d, rows, &cols[cj..cjmax2], rk.clone(), &mut r, kstep);
                    for ci in (cj..out_m).step_by(thresh1d) {
                        dense_row_block(
                            arch,
                            x,
                            rows,
                            cols,
                            b,
                            &r,
                            &mut l,
                            &mut out,
                            ci,
                            cj,
                            cjmax2,
                            rk.clone(),
                        );
                    }
                }
            }
            Ok(())
        })
        .expect("the workers cannot fail");
    } else {
        // One shared R panel, packed in parallel over its columns; the
        // threads split the row blocks of out, each with its own L panel.
        let mut r = vec![F::default(); r_len];
        let l_pool = PanelPool::new(l_len);
        for cj in (0..out_m).step_by(kstep) {
            let cjmax2 = (cj + kstep).min(out_m);
            for rk_start in (0..in_n).step_by(kstep) {
                let rk = rk_start..(rk_start + kstep).min(in_n);
                r.par_chunks_mut(kstep)
                    .zip(cols[cj..cjmax2].par_iter())
                    .for_each(|(rcol, &jj)| {
                        // SAFETY: as in `pack_r`.
                        unsafe {
                            x.gather_column::<i32, true>(jj as usize, &rows[rk.clone()], d, rcol)
                        };
                    });
                let r = &r[..];
                out[cj * out_m..]
                    .par_chunks_mut(thresh1d * out_m)
                    .enumerate()
                    .for_each(|(block, data)| {
                        let ci = cj + block * thresh1d;
                        let mut out = OutBlock {
                            data,
                            out_m,
                            row0: ci,
                        };
                        l_pool.with(|l| {
                            dense_row_block(
                                arch,
                                x,
                                rows,
                                cols,
                                b,
                                r,
                                l,
                                &mut out,
                                ci,
                                cj,
                                cjmax2,
                                rk.clone(),
                            );
                        });
                    });
            }
        }
    }

    mirror_lower_triangle(ArrayViewMut2::from_shape((out_m, out_m), out).expect("square"));
}

// ---------------------------------------------------------------------------
// Matrix-vector products
// ---------------------------------------------------------------------------

/// Products of more entries than this run in parallel; below, starting the
/// threads costs more than the product.
const PARALLEL_MIN_WORK: usize = 100_000;

/// Rows per block of the matrix-vector kernels when the columns of `X` are
/// contiguous: longer blocks mean fewer column switches per block.
const ROW_BLOCK_F: usize = 4096;

/// Rows per block when the rows of `X` are contiguous.
const ROW_BLOCK_C: usize = 1024;

/// The index of the `k`-th selected row or column: `k` itself without a
/// selection.
#[inline(always)]
fn select(sel: Option<&[i32]>, k: usize) -> usize {
    sel.map_or(k, |s| s[k] as usize)
}

/// The shape of a matrix-vector product: the selected rows and columns
/// (`None` for all), and the blocking that follows from the layout of `X`.
#[derive(Clone, Copy)]
struct MatvecShape<'a> {
    rows: Option<&'a [i32]>,
    cols: Option<&'a [i32]>,
    n_rows: usize,
    n_cols: usize,
    /// Whether the columns of `X` are contiguous (Fortran order); otherwise
    /// the kernels walk along the rows.
    col_contiguous: bool,
    row_block: usize,
}

impl<'a> MatvecShape<'a> {
    fn new<F: Float>(x: Centered<'_, F>, rows: Option<&'a [i32]>, cols: Option<&'a [i32]>) -> Self {
        let col_contiguous = x.x.strides()[0] == 1 && x.nrows() > 1;
        Self {
            rows,
            cols,
            n_rows: rows.map_or(x.nrows(), |r| r.len()),
            n_cols: cols.map_or(x.ncols(), |c| c.len()),
            col_contiguous,
            row_block: if col_contiguous {
                ROW_BLOCK_F
            } else {
                ROW_BLOCK_C
            },
        }
    }

    fn n_blocks(&self) -> usize {
        self.n_rows.div_ceil(self.row_block)
    }

    fn block(&self, b: usize) -> Range<usize> {
        b * self.row_block..((b + 1) * self.row_block).min(self.n_rows)
    }

    /// Number of workers: one when the product is small.
    fn n_workers(&self) -> usize {
        if self.n_rows * self.n_cols > PARALLEL_MIN_WORK {
            self.n_blocks().min(rayon::current_num_threads()).max(1)
        } else {
            1
        }
    }

    /// The row blocks of worker `w` of `n_workers` (a static split).
    fn worker_blocks(&self, w: usize, n_workers: usize) -> Range<usize> {
        let n_blocks = self.n_blocks();
        w * n_blocks / n_workers..(w + 1) * n_blocks / n_workers
    }
}

/// `out[Cj] += sum_{Ci in block} (X - center)[rows[Ci], cols[Cj]] * v[rows[Ci]]`
/// for one row block, as the operation `pulp` dispatches.
struct RmatvecBlock<'x, 'v, 'o, F> {
    x: Centered<'x, F>,
    v: &'v [F],
    shape: MatvecShape<'v>,
    block: Range<usize>,
    out: &'o mut [F],
}

impl<F: SimdFloat> WithSimd for RmatvecBlock<'_, '_, '_, F> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let RmatvecBlock {
            x,
            v,
            shape,
            block,
            out,
        } = self;
        // SAFETY (all unchecked loads below): `rows`, `cols` and the lengths
        // of `v` and `center` were checked at the Python boundary.
        if shape.col_contiguous {
            // One (centered) dot product per column; a contiguous column
            // over all rows is vectorised.
            for (cj, o) in out.iter_mut().enumerate() {
                let j = select(shape.cols, cj);
                let mu = x.center_of(j);
                let column = x.x.column(j);
                match (shape.rows, column.as_slice()) {
                    (None, Some(column)) => {
                        *o += centered_dot(simd, &column[block.clone()], mu, &v[block.clone()]);
                    }
                    _ => {
                        let mut sum = F::default();
                        for ci in block.clone() {
                            let i = select(shape.rows, ci);
                            sum +=
                                (unsafe { *column.uget(i) } - mu) * unsafe { *v.get_unchecked(i) };
                        }
                        *o += sum;
                    }
                }
            }
        } else {
            // Scale each row by its v entry and add it to out; a contiguous
            // row over all columns is vectorised.
            let center = x.center_vec();
            for ci in block {
                let i = select(shape.rows, ci);
                let vi = unsafe { *v.get_unchecked(i) };
                let row = x.x.row(i);
                match (shape.cols, row.as_slice()) {
                    (None, Some(row)) => centered_axpy_vc(simd, row, &center, vi, out),
                    _ => {
                        for (cj, o) in out.iter_mut().enumerate() {
                            let j = select(shape.cols, cj);
                            *o += (unsafe { *row.uget(j) } - unsafe { *center.get_unchecked(j) })
                                * vi;
                        }
                    }
                }
            }
        }
    }
}

/// `out = (X - center)[rows, cols].T @ v[rows]`; `out` has `n_cols` zeroed
/// entries. The row blocks are split statically between the workers, each
/// accumulating into its own copy of `out`.
pub fn rmatvec<F: SimdFloat>(
    x: Centered<'_, F>,
    v: &[F],
    rows: Option<&[i32]>,
    cols: Option<&[i32]>,
    out: &mut [F],
) {
    let shape = MatvecShape::new(x, rows, cols);
    let arch = Arch::new();
    let n_workers = shape.n_workers();
    partial_sums::<F, ()>(n_workers, out, |w, out| {
        for b in shape.worker_blocks(w, n_workers) {
            arch.dispatch(RmatvecBlock {
                x,
                v,
                shape,
                block: shape.block(b),
                out,
            });
        }
        Ok(())
    })
    .expect("the workers cannot fail");
}

/// `out[Ci] = sum_Cj (X - center)[rows[Ci], cols[Cj]] * v[cols[Cj]]` for
/// one row block (`out` is that block's slice), as the operation `pulp`
/// dispatches.
struct MatvecBlock<'x, 'v, 'o, F> {
    x: Centered<'x, F>,
    v: &'v [F],
    shape: MatvecShape<'v>,
    block: Range<usize>,
    out: &'o mut [F],
}

impl<F: SimdFloat> WithSimd for MatvecBlock<'_, '_, '_, F> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let MatvecBlock {
            x,
            v,
            shape,
            block,
            out,
        } = self;
        // SAFETY (all unchecked loads below): as in `RmatvecBlock`.
        if shape.col_contiguous {
            // Add the columns to out, four at a time where all rows are
            // used and the columns are contiguous.
            let column = |cj: usize| {
                let j = select(shape.cols, cj);
                (j, x.x.column(j), x.center_of(j), unsafe {
                    *v.get_unchecked(j)
                })
            };
            let mut cj = 0;
            if shape.rows.is_none() {
                while cj + 4 <= shape.n_cols {
                    let cols: [_; 4] = std::array::from_fn(|t| column(cj + t));
                    let slices = cols.each_ref().map(|(_, c, _, _)| c.as_slice());
                    if slices.iter().all(|s| s.is_some()) {
                        let xs = slices.map(|s| &s.unwrap()[block.clone()]);
                        centered_axpy4(simd, xs, cols.map(|c| c.2), cols.map(|c| c.3), out);
                        cj += 4;
                    } else {
                        break;
                    }
                }
            }
            for cj in cj..shape.n_cols {
                let (_, column, mu, vj) = column(cj);
                match (shape.rows, column.as_slice()) {
                    (None, Some(column)) => {
                        centered_axpy(simd, &column[block.clone()], mu, vj, out)
                    }
                    _ => {
                        for (ci, o) in block.clone().zip(out.iter_mut()) {
                            let i = select(shape.rows, ci);
                            *o += (unsafe { *column.uget(i) } - mu) * vj;
                        }
                    }
                }
            }
        } else {
            // One (centered) dot product per row; a contiguous row over all
            // columns is vectorised.
            let center = x.center_vec();
            for (ci, o) in block.zip(out.iter_mut()) {
                let i = select(shape.rows, ci);
                let row = x.x.row(i);
                *o = match (shape.cols, row.as_slice()) {
                    (None, Some(row)) => centered_dot_vc(simd, row, &center, v),
                    _ => {
                        let mut sum = F::default();
                        for cj in 0..shape.n_cols {
                            let j = select(shape.cols, cj);
                            sum += (unsafe { *row.uget(j) } - unsafe { *center.get_unchecked(j) })
                                * unsafe { *v.get_unchecked(j) };
                        }
                        sum
                    }
                };
            }
        }
    }
}

/// `out = (X - center)[rows, cols] @ v[cols]`; `out` has `n_rows` zeroed
/// entries. Every row block owns its slice of `out`, so the blocks are
/// simply split between the threads.
pub fn matvec<F: SimdFloat>(
    x: Centered<'_, F>,
    v: &[F],
    rows: Option<&[i32]>,
    cols: Option<&[i32]>,
    out: &mut [F],
) {
    let shape = MatvecShape::new(x, rows, cols);
    let arch = Arch::new();
    let run = |b: usize, out: &mut [F]| {
        arch.dispatch(MatvecBlock {
            x,
            v,
            shape,
            block: shape.block(b),
            out,
        })
    };
    if shape.n_workers() > 1 {
        out.par_chunks_mut(shape.row_block)
            .enumerate()
            .for_each(|(b, out)| run(b, out));
    } else {
        out.chunks_mut(shape.row_block)
            .enumerate()
            .for_each(|(b, out)| run(b, out));
    }
}

/// `out[j] = sum_i weights[i] * (X[i, j] - shift[j])**2` over the first
/// `weights.len()` rows of `X`; `out` and `shift` have one entry per
/// column of `X`, `weights` at most one per row.
pub fn transpose_square_dot_weights<F: Float>(x: Centered<'_, F>, weights: &[F], out: &mut [F]) {
    out.par_iter_mut().enumerate().for_each(|(j, out)| {
        let shift = x.center_of(j);
        let column = x.x.column(j);
        let mut sum = F::default();
        for (i, &w) in weights.iter().enumerate() {
            // SAFETY: `i < weights.len() <= x.nrows()` and `j < out.len() == x.ncols()`.
            let centred = unsafe { *column.uget(i) } - shift;
            sum += w * centred * centred;
        }
        *out = sum;
    });
}

#[cfg(test)]
mod tests {
    //! The kernels against plain reference loops, over random shapes, both
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

    fn reference_matvec(
        x: ArrayView2<'_, f64>,
        v: &[f64],
        rows: Option<&[i32]>,
        cols: Option<&[i32]>,
        center: Option<&[f64]>,
    ) -> Vec<f64> {
        let all_rows: Vec<i32> = (0..x.nrows() as i32).collect();
        let all_cols: Vec<i32> = (0..x.ncols() as i32).collect();
        let mu = |j: usize| center.map_or(0.0, |c| c[j]);
        rows.unwrap_or(&all_rows)
            .iter()
            .map(|&i| {
                cols.unwrap_or(&all_cols)
                    .iter()
                    .map(|&j| (x[[i as usize, j as usize]] - mu(j as usize)) * v[j as usize])
                    .sum()
            })
            .collect()
    }

    fn reference_rmatvec(
        x: ArrayView2<'_, f64>,
        v: &[f64],
        rows: Option<&[i32]>,
        cols: Option<&[i32]>,
        center: Option<&[f64]>,
    ) -> Vec<f64> {
        let all_rows: Vec<i32> = (0..x.nrows() as i32).collect();
        let all_cols: Vec<i32> = (0..x.ncols() as i32).collect();
        let mu = |j: usize| center.map_or(0.0, |c| c[j]);
        cols.unwrap_or(&all_cols)
            .iter()
            .map(|&j| {
                rows.unwrap_or(&all_rows)
                    .iter()
                    .map(|&i| (x[[i as usize, j as usize]] - mu(j as usize)) * v[i as usize])
                    .sum()
            })
            .collect()
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

    #[test]
    fn matvec_kernels_match_reference() {
        let mut rng = Rng::new(12);
        for &(n, m) in &[(5000usize, 30usize), (300, 9), (2500, 257)] {
            let data = Array2::from_shape_vec((n, m), rng.values(n * m, 5.0)).unwrap();
            let v_rows = rng.values(n, 0.0);
            let v_cols = rng.values(m, 0.0);
            let center: Vec<f64> = (0..m).map(|j| data.column(j).mean().unwrap()).collect();
            let sub_rows = rng.subset(n);
            let sub_cols = rng.subset(m);
            for (name, arr) in layouts(&data) {
                let x = view(name, &arr);
                for rows in [None, Some(&sub_rows[..])] {
                    for cols in [None, Some(&sub_cols[..])] {
                        for center in [None, Some(&center[..])] {
                            let op = Centered { x, center };
                            let what = format!(
                                "{name} {n}x{m} rows={} cols={}",
                                rows.is_some(),
                                cols.is_some()
                            );
                            let exp_m = reference_matvec(x, &v_cols, rows, cols, center);
                            let exp_r = reference_rmatvec(x, &v_rows, rows, cols, center);
                            for threads in THREADS {
                                let mut first = None;
                                for _ in 0..2 {
                                    let mut out_m = vec![0.0; exp_m.len()];
                                    let mut out_r = vec![0.0; exp_r.len()];
                                    with_threads(threads, || {
                                        matvec(op, &v_cols, rows, cols, &mut out_m);
                                        rmatvec(op, &v_rows, rows, cols, &mut out_r);
                                    });
                                    assert_close(&out_m, &exp_m, 1e-12, &format!("matvec {what}"));
                                    assert_close(&out_r, &exp_r, 1e-12, &format!("rmatvec {what}"));
                                    match &first {
                                        None => first = Some((out_m, out_r)),
                                        Some(f) => assert!(
                                            *f == (out_m, out_r),
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

    #[test]
    fn transpose_square_dot_weights_matches_reference() {
        let mut rng = Rng::new(13);
        let (n, m) = (700, 11);
        let data = Array2::from_shape_vec((n, m), rng.values(n * m, 2.0)).unwrap();
        let w = rng.values(n, 1.0);
        let shift = rng.values(m, 2.0);
        let expected: Vec<f64> = (0..m)
            .map(|j| {
                (0..n)
                    .map(|i| w[i] * (data[[i, j]] - shift[j]).powi(2))
                    .sum()
            })
            .collect();
        for (name, arr) in layouts(&data) {
            let x = view(name, &arr);
            let mut out = vec![0.0; m];
            transpose_square_dot_weights(
                Centered {
                    x,
                    center: Some(&shift),
                },
                &w,
                &mut out,
            );
            assert_close(&out, &expected, 1e-12, &format!("tsdw {name}"));
        }
    }
}
