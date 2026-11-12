//! Kernels for `tabmat.SparseMatrix`: the sandwich product
//! `A.T @ diag(d) @ A` of a sparse matrix, the cross sandwich
//! `A.T @ diag(d) @ (B - center)` with a dense `B`, the matrix-vector
//! products of the CSR and CSC representations and the weighted column sums
//! of squares used for standardisation.
//!
//! The sparse matrices are [`Compressed`]: the `data`, `indices` and
//! `indptr` arrays of scipy's CSR/CSC formats, with the structure of
//! `indptr` checked once ([`Compressed::check`]). The sandwich kernels have
//! their index arrays validated up front by the caller (a vectorised pass,
//! negligible next to their quadratic work) and index without checks in
//! their inner loops. The matrix-vector products are a single pass over the
//! nonzeros themselves, where every form of separate validation pass
//! measured slower than the bounds check of the gather it would remove, so
//! they check inside the loop.
//!
//! # `csr_dense_sandwich`
//!
//! `out[A_cols, B_cols] = A[rows, A_cols].T @ diag(d[rows]) @ (B - center)[rows, B_cols]`
//! is the one kernel with a SIMD inner loop. The rows are processed in
//! blocks of [`KBLOCK`] and the dense columns in blocks of [`JBLOCK`]; for
//! each block pair the d-scaled, centered dense block is packed into a
//! row-major panel `R` ([`Centered::pack_block`]), then every nonzero
//! `A[k, i]` of the block adds `A[k, i] * R[k, :]` into row `i` of the
//! worker's private copy of `out`, an axpy over the dense columns written
//! with `pulp` (portable SIMD with runtime dispatch). The workers split the
//! row blocks statically, like the OpenMP loop of the C++ version, and their
//! copies are summed in worker order ([`partial_sums`]) instead of with
//! atomic adds, so the result does not depend on thread timing.

use std::ops::Range;

use ndarray::ArrayViewMut2;
use pulp::{Arch, Simd, WithSimd};
use rayon::prelude::*;

use super::{
    axpy, mirror_lower_triangle, partial_sums, Centered, Float, Index, IndexError, SimdFloat,
};

/// Rows per block of `csr_dense_sandwich`.
pub const KBLOCK: usize = 128;

/// Dense columns per block of `csr_dense_sandwich`.
pub const JBLOCK: usize = 128;

/// Nonzeros per parallel task of the matrix-vector kernels: the rows (or
/// columns) of a task are sized from the average number of nonzeros per
/// row, so that a matrix with few long columns is still split between the
/// threads.
const TASK_NNZ: usize = 1 << 14;

/// "Not selected" marker in the column maps.
pub const NONE: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Sparse matrix access
// ---------------------------------------------------------------------------

/// A compressed sparse matrix: `indptr[p]..indptr[p + 1]` are the entries
/// of the compressed row (CSR) or column (CSC) `p`, with `indices` their
/// column (CSR) or row (CSC) and `data` their value. `shape` is the shape of
/// the matrix as `(n_rows, n_cols)`.
#[derive(Clone, Copy)]
pub struct Compressed<'a, I, F> {
    pub data: &'a [F],
    pub indices: &'a [I],
    pub indptr: &'a [I],
    pub shape: (usize, usize),
}

/// Why a compressed matrix is malformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatError {
    /// `indptr` does not have `n_major + 1` entries.
    IndptrLength { expected: usize },
    /// `indptr` is not non-decreasing from zero, or its last entry exceeds
    /// the length of `data` or `indices`.
    IndptrEntries,
}

impl<I: Index, F: Float> Compressed<'_, I, F> {
    /// Checks that `indptr` has `n_major + 1` non-decreasing entries from
    /// zero and that `data` and `indices` hold the `indptr[-1]` nonzeros, so
    /// that [`Self::entries`] cannot fail afterwards. (Slicing without the
    /// checks after this measured no faster.)
    pub fn check(&self, n_major: usize) -> Result<(), FormatError> {
        if self.indptr.len() != n_major + 1 {
            return Err(FormatError::IndptrLength {
                expected: n_major + 1,
            });
        }
        let monotone = self.indptr.windows(2).all(|w| w[0] <= w[1]);
        let nnz = self.nnz();
        if !monotone
            || self.indptr[0] < I::default()
            || nnz > self.indices.len()
            || nnz > self.data.len()
        {
            return Err(FormatError::IndptrEntries);
        }
        Ok(())
    }

    /// The stored indices (the first `nnz()` entries of the `indices` array).
    pub fn indices(&self) -> &[I] {
        &self.indices[..self.nnz()]
    }

    /// The indices and values of compressed row or column `p`.
    #[inline(always)]
    pub fn entries(&self, p: usize) -> (&[I], &[F]) {
        let start = self.indptr[p].as_usize();
        let end = self.indptr[p + 1].as_usize();
        (&self.indices[start..end], &self.data[start..end])
    }

    /// Number of stored entries, `indptr[-1]`.
    pub fn nnz(&self) -> usize {
        self.indptr[self.indptr.len() - 1].as_usize()
    }

    /// Number of compressed rows (CSR) or columns (CSC).
    pub fn n_major(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Rows (or columns) per parallel task: about [`TASK_NNZ`] nonzeros.
    fn task_len(&self) -> usize {
        let n_major = self.n_major();
        (TASK_NNZ * n_major / self.nnz().max(1)).clamp(1, n_major.max(1))
    }
}

/// Map from an index to its position in `idx` (`NONE` if absent); `idx` has
/// been checked against `len`.
pub fn position_map<I: Index>(idx: &[I], len: usize) -> Vec<u32> {
    let mut map = vec![NONE; len];
    for (p, &i) in idx.iter().enumerate() {
        map[i.as_usize()] = p as u32;
    }
    map
}

/// Indicator of the entries listed in `idx` out of `len`; `idx` has been
/// checked against `len`.
pub fn included_mask<I: Index>(idx: &[I], len: usize) -> Vec<bool> {
    let mut mask = vec![false; len];
    for &i in idx {
        mask[i.as_usize()] = true;
    }
    mask
}

// ---------------------------------------------------------------------------
// sparse_sandwich: A.T @ diag(d) @ A
// ---------------------------------------------------------------------------

/// `out[cols, cols] = A[rows, cols].T @ diag(d[rows]) @ A[rows, cols]` for
/// `a` the CSC and `at` the CSR representation of the same matrix. `out` is
/// `len(cols)` square, row-major and zeroed. The indices of `a` have been
/// checked against the rows and those of `at` against the columns of the
/// matrix, `row_included` and `d` have one entry per row and `col_map` maps
/// each column to its position in `cols` (or `NONE`).
///
/// For every selected column `j` (a row of `out`) the entries `A[k, j]` of
/// that column are walked, and for every selected row `k` the entries
/// `A[k, i]` of that row with `i <= j` (the CSR indices are sorted) are
/// accumulated into `out[j, i]`: the lower triangle, which is then mirrored.
/// The rows of `out` are independent, so they are computed in parallel.
pub fn sandwich<I: Index, F: Float>(
    a: Compressed<'_, I, F>,
    at: Compressed<'_, I, F>,
    d: &[F],
    row_included: &[bool],
    cols: &[I],
    col_map: &[u32],
    out: &mut [F],
) {
    let m = cols.len();
    assert!(row_included.len() == a.shape.0 && d.len() >= a.shape.0 && col_map.len() == a.shape.1);
    out.par_chunks_mut(m).enumerate().for_each(|(cj, out_row)| {
        let j = cols[cj].as_usize();
        let (a_rows, a_vals) = a.entries(j);
        for (&k, &a_val) in a_rows.iter().zip(a_vals) {
            let k = k.as_usize();
            // SAFETY: `k` is a checked row index of the matrix, and
            // `row_included` and `d` have an entry per row (asserted above).
            if !unsafe { *row_included.get_unchecked(k) } {
                continue;
            }
            let a_val = a_val * unsafe { *d.get_unchecked(k) };
            let (at_cols, at_vals) = at.entries(k);
            for (&i, &at_val) in at_cols.iter().zip(at_vals) {
                let i = i.as_usize();
                if i > j {
                    break;
                }
                // SAFETY: `i` is a checked column index and `col_map` has an
                // entry per column (asserted above).
                let ci = unsafe { *col_map.get_unchecked(i) };
                if ci == NONE {
                    continue;
                }
                // SAFETY: `position_map` only stores positions below
                // `cols.len() == m`, the length of `out_row`.
                unsafe { *out_row.get_unchecked_mut(ci as usize) += at_val * a_val };
            }
        }
    });
    mirror_lower_triangle(ArrayViewMut2::from_shape((m, m), out).expect("square"));
}

// ---------------------------------------------------------------------------
// csr_dense_sandwich: A.T @ diag(d) @ (B - center)
// ---------------------------------------------------------------------------

/// One `(row block, dense column block)` step of `csr_dense_sandwich`, as
/// the operation `pulp` dispatches to the instruction set detected at run
/// time: pack the d-scaled, centered dense block into `r`, then do the axpy
/// of every nonzero of the block into `outtemp`.
struct CsrDenseBlock<'a, 'b, I, F> {
    a: Compressed<'a, I, F>,
    b: Centered<'b, F>,
    mu: &'a [F],
    d: &'a [F],
    rows: &'a [I],
    b_cols: &'a [I],
    acol_map: &'a [u32],
    r: &'a mut [F],
    outtemp: &'a mut [F],
    out_stride: usize,
    ck: Range<usize>,
    cj: Range<usize>,
}

impl<I: Index, F: SimdFloat> WithSimd for CsrDenseBlock<'_, '_, I, F> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let CsrDenseBlock {
            a,
            b,
            mu,
            d,
            rows,
            b_cols,
            acol_map,
            r,
            outtemp,
            out_stride,
            ck,
            cj,
        } = self;
        let jlen = cj.len();
        let block_rows = &rows[ck];
        let block_cols = &b_cols[cj.clone()];

        // Pack the d-scaled, centered dense block row-major through the
        // view's strides (for an F-ordered b this reads one element of each
        // of the block's columns per row, which the cache handles better than
        // packing column by column did).
        // SAFETY: `rows` was checked against the rows of `b` and `d`,
        // `b_cols` against its columns, by the caller.
        unsafe { b.pack_block(block_rows, block_cols, d, mu, r, JBLOCK) };

        for (rrow, &k) in r.chunks(JBLOCK).zip(block_rows) {
            let rrow = &rrow[..jlen];
            let (a_cols, a_vals) = a.entries(k.as_usize());
            for (&i, &q) in a_cols.iter().zip(a_vals) {
                let ci = acol_map[i.as_usize()];
                if ci == NONE {
                    continue;
                }
                let outrow = &mut outtemp[ci as usize * out_stride + cj.start..][..jlen];
                axpy(simd, q, rrow, outrow);
            }
        }
    }
}

/// `out[A_cols, B_cols] = A[rows, A_cols].T @ diag(d[rows]) @ (B - center)[rows, B_cols]`
/// for `a` in CSR format and `b` any strided view with an optional center.
/// `out` is `len(A_cols) x len(B_cols)`, row-major and zeroed; `rows` has
/// been checked against the rows of `a`, `b` and `d`, `b_cols` against the
/// columns of `b`, and the indices of `a` against `acol_map`.
#[allow(clippy::too_many_arguments)]
pub fn csr_dense_sandwich<I: Index, F: SimdFloat>(
    a: Compressed<'_, I, F>,
    b: Centered<'_, F>,
    d: &[F],
    rows: &[I],
    acol_map: &[u32],
    b_cols: &[I],
    n_a_cols: usize,
    out: &mut [F],
) {
    let (nr, nbc) = (rows.len(), b_cols.len());
    // the per-thread partial results have the layout of out
    let out_stride = nbc;
    let arch = Arch::new();
    let mu = b.center_vec();

    // Each worker takes a contiguous range of the row blocks (OpenMP's
    // static schedule) and accumulates into its own copy of out.
    let n_blocks = nr.div_ceil(KBLOCK);
    let n_workers = n_blocks.min(rayon::current_num_threads()).max(1);
    let worker_blocks = |w: usize| w * n_blocks / n_workers..(w + 1) * n_blocks / n_workers;
    partial_sums::<F, ()>(n_workers, out, |w, outtemp| {
        let mut r = vec![F::default(); KBLOCK * JBLOCK];
        for block in worker_blocks(w) {
            let ck = block * KBLOCK..((block + 1) * KBLOCK).min(nr);
            for cjj in (0..nbc).step_by(JBLOCK) {
                let cj = cjj..(cjj + JBLOCK).min(nbc);
                arch.dispatch(CsrDenseBlock {
                    a,
                    b,
                    mu: &mu,
                    d,
                    rows,
                    b_cols,
                    acol_map,
                    r: &mut r,
                    outtemp,
                    out_stride,
                    ck: ck.clone(),
                    cj,
                });
            }
        }
        debug_assert_eq!(n_a_cols * out_stride, outtemp.len());
        Ok(())
    })
    .expect("the workers cannot fail");
}

// ---------------------------------------------------------------------------
// Matrix-vector products
// ---------------------------------------------------------------------------

/// `out[p] += x[major[p], :] . v` over the entries of each compressed row
/// (CSR: `X @ v`) or column (CSC: `XT.T @ v`), restricted to the minor
/// indices flagged in `included`. Without `RESTRICTED` all compressed rows
/// (in order) and all entries take part and `major` and `included` are
/// unused, so that the unrestricted products compile to the plain loop.
/// `major` has been checked against the compressed dimension; the stored
/// indices are checked against `v` (and `included`) in the loop.
pub fn matvec<I: Index, F: Float, const RESTRICTED: bool>(
    x: Compressed<'_, I, F>,
    v: &[F],
    major: &[I],
    included: &[bool],
    out: &mut [F],
) -> Result<(), IndexError> {
    // Sliced to the same length, so that the one bounds check of `v.get`
    // covers `included` too.
    let n_minor = if RESTRICTED {
        v.len().min(included.len())
    } else {
        v.len()
    };
    let (v, included) = (
        &v[..n_minor],
        &included[..if RESTRICTED { n_minor } else { 0 }],
    );
    let minor = IndexError {
        what: "stored",
        len: n_minor,
    };
    out.par_iter_mut()
        .enumerate()
        .with_min_len(x.task_len())
        .try_for_each(move |(p, o)| {
            let p = if RESTRICTED { major[p].as_usize() } else { p };
            let (minor_idx, vals) = x.entries(p);
            let mut acc = *o;
            for (&q, &xv) in minor_idx.iter().zip(vals) {
                let q = q.as_usize();
                let Some(&vq) = v.get(q) else {
                    return Err(minor);
                };
                if !RESTRICTED || included[q] {
                    acc += xv * vq;
                }
            }
            *o = acc;
            Ok(())
        })
}

/// `out[j] = sum_k weights[indices[k]] * data[k]**2` over the entries `k`
/// of column `j` of the CSC matrix `x`; `out` has one entry per column. The
/// stored indices are checked against `weights` in the loop.
pub fn transpose_square_dot_weights<I: Index, F: Float>(
    x: Compressed<'_, I, F>,
    weights: &[F],
    out: &mut [F],
) -> Result<(), IndexError> {
    let row = IndexError {
        what: "stored",
        len: weights.len(),
    };
    out.par_iter_mut()
        .enumerate()
        .with_min_len(x.task_len())
        .try_for_each(move |(j, o)| {
            let (rows, vals) = x.entries(j);
            let mut acc = F::default();
            for (&i, &v) in rows.iter().zip(vals) {
                acc += *weights.get(i.as_usize()).ok_or(row)? * (v * v);
            }
            *o = acc;
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    //! The kernels against plain reference loops over a dense copy of the
    //! sparse matrix, for both index types, subsets, both layouts of the
    //! dense operand, with and without a center, and several thread counts.

    use ndarray::Array2;

    use super::super::check_indices;
    use super::super::testing::{assert_close, with_threads, Rng, THREADS};
    use super::*;

    /// A random sparse matrix as a dense array plus its CSR and CSC arrays
    /// (sorted indices, as scipy produces).
    struct Sparse {
        dense: Array2<f64>,
        csr: (Vec<f64>, Vec<i64>, Vec<i64>),
        csc: (Vec<f64>, Vec<i64>, Vec<i64>),
    }

    fn random_sparse(rng: &mut Rng, n: usize, m: usize, density: f64) -> Sparse {
        let mut dense = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                if rng.uniform() < density {
                    dense[[i, j]] = rng.value(0.5);
                }
            }
        }
        let compress = |major: usize, minor: usize, at: &dyn Fn(usize, usize) -> f64| {
            let (mut data, mut indices, mut indptr) = (vec![], vec![], vec![0i64]);
            for p in 0..major {
                for q in 0..minor {
                    let v = at(p, q);
                    if v != 0.0 {
                        data.push(v);
                        indices.push(q as i64);
                    }
                }
                indptr.push(data.len() as i64);
            }
            (data, indices, indptr)
        };
        let csr = compress(n, m, &|i, j| dense[[i, j]]);
        let csc = compress(m, n, &|j, i| dense[[i, j]]);
        Sparse { dense, csr, csc }
    }

    fn as_i32(v: &[i64]) -> Vec<i32> {
        v.iter().map(|&x| x as i32).collect()
    }

    fn compressed<'a, I: Index>(
        arrays: &'a (Vec<f64>, Vec<I>, Vec<I>),
        shape: (usize, usize),
    ) -> Compressed<'a, I, f64> {
        let c = Compressed {
            data: &arrays.0,
            indices: &arrays.1,
            indptr: &arrays.2,
            shape,
        };
        c.check(arrays.2.len() - 1).unwrap();
        c
    }

    fn run_all<I: Index + From<i32>>(
        rng: &mut Rng,
        sp: &Sparse,
        n: usize,
        m: usize,
        csr: &(Vec<f64>, Vec<I>, Vec<I>),
        csc: &(Vec<f64>, Vec<I>, Vec<I>),
    ) {
        let a = compressed(csc, (n, m));
        let at = compressed(csr, (n, m));
        let d = rng.values(n, 1.0);
        let r = 7;
        let b_data = Array2::from_shape_vec((n, r), rng.values(n * r, 4.0)).unwrap();
        let b_f = b_data.t().as_standard_layout().t().to_owned();
        let center: Vec<f64> = (0..r).map(|j| b_data.column(j).mean().unwrap()).collect();
        let v_rows = rng.values(n, 0.0);
        let v_cols = rng.values(m, 0.0);
        let all_rows: Vec<I> = (0..n as i32).map(I::from).collect();
        let all_cols: Vec<I> = (0..m as i32).map(I::from).collect();
        let sub_rows: Vec<I> = rng.subset(n).into_iter().map(I::from).collect();
        let sub_cols: Vec<I> = rng.subset(m).into_iter().map(I::from).collect();
        let all_b: Vec<I> = (0..r as i32).map(I::from).collect();
        let sub_b: Vec<I> = rng.subset(r).into_iter().map(I::from).collect();
        let dense = &sp.dense;

        for (rows, cols, b_cols) in [
            (&all_rows, &all_cols, &all_b),
            (&sub_rows, &sub_cols, &sub_b),
        ] {
            check_indices(rows, n, "row").unwrap();
            check_indices(cols, m, "column").unwrap();
            let row_included = included_mask(rows, n);
            let col_map = position_map(cols, m);
            // reference sandwich
            let mc = cols.len();
            let mut expected = vec![0.0; mc * mc];
            for (ci, &i) in cols.iter().enumerate() {
                for (cj, &j) in cols.iter().enumerate() {
                    expected[ci * mc + cj] = rows
                        .iter()
                        .map(|&k| {
                            let k = k.as_usize();
                            d[k] * dense[[k, i.as_usize()]] * dense[[k, j.as_usize()]]
                        })
                        .sum();
                }
            }
            let acol_map = position_map(cols, m);
            for threads in THREADS {
                let mut first = None;
                for _ in 0..2 {
                    let mut out = vec![0.0; mc * mc];
                    with_threads(threads, || {
                        sandwich(a, at, &d, &row_included, cols, &col_map, &mut out)
                    });
                    assert_close(&out, &expected, 1e-12, "sparse_sandwich");
                    match &first {
                        None => first = Some(out),
                        Some(f) => assert!(*f == out, "run-to-run difference"),
                    }
                }
            }
            // csr x dense, both layouts, with and without a center
            for (name, b) in [("C", &b_data), ("F", &b_f)] {
                for center in [None, Some(&center[..])] {
                    let nbc = b_cols.len();
                    let mut expected = vec![0.0; mc * nbc];
                    for (ci, &i) in cols.iter().enumerate() {
                        for (cj, &j) in b_cols.iter().enumerate() {
                            let j = j.as_usize();
                            let mu = center.map_or(0.0, |c| c[j]);
                            expected[ci * nbc + cj] = rows
                                .iter()
                                .map(|&k| {
                                    let k = k.as_usize();
                                    d[k] * dense[[k, i.as_usize()]] * (b[[k, j]] - mu)
                                })
                                .sum();
                        }
                    }
                    let bop = Centered {
                        x: b.view(),
                        center,
                    };
                    for threads in THREADS {
                        let mut first = None;
                        for _ in 0..2 {
                            let mut out = vec![0.0; mc * nbc];
                            with_threads(threads, || {
                                csr_dense_sandwich(
                                    at, bop, &d, rows, &acol_map, b_cols, mc, &mut out,
                                )
                            });
                            assert_close(
                                &out,
                                &expected,
                                1e-12,
                                &format!("csr_dense_sandwich {name}"),
                            );
                            match &first {
                                None => first = Some(out),
                                Some(f) => assert!(*f == out, "run-to-run difference"),
                            }
                        }
                    }
                }
            }
            // matvecs
            let col_included = included_mask(cols, m);
            let exp_csr: Vec<f64> = rows
                .iter()
                .map(|&i| {
                    cols.iter()
                        .map(|&j| dense[[i.as_usize(), j.as_usize()]] * v_cols[j.as_usize()])
                        .sum()
                })
                .collect();
            let exp_csc: Vec<f64> = cols
                .iter()
                .map(|&j| {
                    rows.iter()
                        .map(|&i| dense[[i.as_usize(), j.as_usize()]] * v_rows[i.as_usize()])
                        .sum()
                })
                .collect();
            for threads in THREADS {
                let mut out_r = vec![0.0; rows.len()];
                let mut out_c = vec![0.0; cols.len()];
                with_threads(threads, || {
                    matvec::<I, f64, true>(at, &v_cols, rows, &col_included, &mut out_r).unwrap();
                    matvec::<I, f64, true>(a, &v_rows, cols, &row_included, &mut out_c).unwrap();
                });
                assert_close(&out_r, &exp_csr, 1e-12, "csr_matvec");
                assert_close(&out_c, &exp_csc, 1e-12, "csc_rmatvec");
            }
        }
        // unrestricted matvecs and tsdw
        let exp_csr: Vec<f64> = (0..n)
            .map(|i| (0..m).map(|j| dense[[i, j]] * v_cols[j]).sum())
            .collect();
        let exp_csc: Vec<f64> = (0..m)
            .map(|j| (0..n).map(|i| dense[[i, j]] * v_rows[i]).sum())
            .collect();
        let exp_tsdw: Vec<f64> = (0..m)
            .map(|j| (0..n).map(|i| d[i] * dense[[i, j]].powi(2)).sum())
            .collect();
        for threads in THREADS {
            let mut out_r = vec![1.0; n];
            let mut out_c = vec![0.0; m];
            let mut out_t = vec![0.0; m];
            with_threads(threads, || {
                matvec::<I, f64, false>(at, &v_cols, &[], &[], &mut out_r).unwrap();
                matvec::<I, f64, false>(a, &v_rows, &[], &[], &mut out_c).unwrap();
                transpose_square_dot_weights(a, &d, &mut out_t).unwrap();
            });
            let exp_r1: Vec<f64> = exp_csr.iter().map(|x| x + 1.0).collect();
            assert_close(&out_r, &exp_r1, 1e-12, "csr_matvec_unrestricted");
            assert_close(&out_c, &exp_csc, 1e-12, "csc_rmatvec_unrestricted");
            assert_close(&out_t, &exp_tsdw, 1e-12, "tsdw");
        }
    }

    #[test]
    fn kernels_match_reference() {
        let mut rng = Rng::new(21);
        for &(n, m, density) in &[(300usize, 40usize, 0.1), (1500, 150, 0.02), (60, 9, 0.5)] {
            let sp = random_sparse(&mut rng, n, m, density);
            let csr32 = (sp.csr.0.clone(), as_i32(&sp.csr.1), as_i32(&sp.csr.2));
            let csc32 = (sp.csc.0.clone(), as_i32(&sp.csc.1), as_i32(&sp.csc.2));
            run_all::<i64>(&mut rng, &sp, n, m, &sp.csr, &sp.csc);
            run_all::<i32>(&mut rng, &sp, n, m, &csr32, &csc32);
        }
    }

    #[test]
    fn malformed_structure_and_bad_indices_are_errors() {
        let data = [1.0, 2.0];
        let indices = [0i32, 5];
        let indptr = [0i32, 1, 2];
        let x = Compressed {
            data: &data,
            indices: &indices,
            indptr: &indptr,
            shape: (2, 2),
        };
        assert_eq!(x.check(3), Err(FormatError::IndptrLength { expected: 4 }));
        assert!(x.check(2).is_ok());
        let bad = [0i32, 2, 1];
        let y = Compressed {
            data: &data,
            indices: &indices,
            indptr: &bad,
            shape: (2, 2),
        };
        assert_eq!(y.check(2), Err(FormatError::IndptrEntries));
        let mut out = [0.0; 2];
        assert!(matvec::<i32, f64, false>(x, &[1.0, 1.0], &[], &[], &mut out).is_err());
        assert!(transpose_square_dot_weights(x, &[1.0, 1.0], &mut out).is_err());
    }
}
