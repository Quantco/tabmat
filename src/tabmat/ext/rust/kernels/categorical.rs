//! Kernels for `tabmat.CategoricalMatrix`: a categorical matrix is stored as
//! one `int32` index per row, giving the column of that row's single
//! nonzero entry (`-1` for a missing value).
//!
//! Each kernel exists in a plain and a "complex" flavour that differ only in
//! the handling of the index: the complex one subtracts `drop_first` and
//! skips rows whose adjusted column index is negative (dropped reference
//! category or missing value). The flavours share one kernel, specialised at
//! compile time through the `DROP` const generic, so that the plain path has
//! no check at all (a run-time check measured 28% slower on cache-resident
//! results).
//!
//! The categories are part of the data the kernels stream through, so they
//! are checked in the loop, as one comparison against the number of columns
//! that also lets the compiler drop the bounds check of the scatter (an
//! `Option`-returning lookup in its place measured up to 30% slower, and
//! validating gathered chunks of categories up front up to 2x slower, on
//! cache-resident results). The scatter kernels are parallelised with
//! [`scatter_add`], the gather of `matvec` over chunks of rows. The closures
//! capture by value: captured references made the compiler reload every
//! invariant from memory on each row.

use rayon::prelude::*;

use super::{
    category_column, check_indices, scatter_add, Float, IndexError, PARALLEL_MIN_ROWS,
    VALIDATE_CHUNK,
};

/// Indicator of the columns listed in `cols` (`get_col_included` in Cython).
pub fn col_mask(cols: &[i32], n_cols: usize) -> Result<Vec<bool>, IndexError> {
    check_indices(cols, n_cols, "column")?;
    let mut mask = vec![false; n_cols];
    for &c in cols {
        mask[c as usize] = true;
    }
    Ok(mask)
}

/// `out[indices[row] - drop_first] += other[row]` over `rows`, restricted to
/// the columns in `cols`.
///
/// `rows` of `None` (or of full length) means all rows and `cols` of `None`
/// (or of full length) means all columns, as in the Cython original.
pub fn transpose_matvec<F: Float, const DROP: bool>(
    indices: &[i32],
    other: &[F],
    out: &mut [F],
    rows: Option<&[i32]>,
    cols: Option<&[i32]>,
    drop_first: i32,
) -> Result<(), IndexError> {
    let n_cols = out.len();
    let rows = rows.filter(|r| r.len() != indices.len());
    let mask = cols
        .filter(|c| c.len() != n_cols)
        .map(|c| col_mask(c, n_cols))
        .transpose()?;
    let mask = mask.as_deref();
    let category = IndexError {
        what: "category",
        len: n_cols,
    };

    match rows {
        None => scatter_add(indices.len(), out, move |r, res| {
            let other = other.get(r.clone()).ok_or(IndexError {
                what: "row",
                len: other.len(),
            })?;
            let res = &mut res[..n_cols];
            for (&index, &val) in indices[r].iter().zip(other) {
                if let Some(col) = category_column::<DROP>(index, drop_first) {
                    if col >= n_cols {
                        return Err(category);
                    }
                    if mask.is_none_or(|m| m[col]) {
                        res[col] += val;
                    }
                }
            }
            Ok(())
        }),
        Some(rows) => {
            let n_rows = indices.len().min(other.len());
            let (indices, other) = (&indices[..n_rows], &other[..n_rows]);
            let row = IndexError {
                what: "row",
                len: n_rows,
            };
            scatter_add(rows.len(), out, move |r, res| {
                let res = &mut res[..n_cols];
                for &k in &rows[r] {
                    let k = k as usize;
                    if k >= n_rows {
                        return Err(row);
                    }
                    if let Some(col) = category_column::<DROP>(indices[k], drop_first) {
                        if col >= n_cols {
                            return Err(category);
                        }
                        if mask.is_none_or(|m| m[col]) {
                            res[col] += other[k];
                        }
                    }
                }
                Ok(())
            })
        }
    }
}

/// `out[i] += other[indices[i] - drop_first]` for every row of `indices`,
/// skipping negative indices and, if `cols` is given, columns not in it.
pub fn matvec<F: Float, const DROP: bool>(
    indices: &[i32],
    other: &[F],
    out: &mut [F],
    cols: Option<&[i32]>,
    drop_first: i32,
) -> Result<(), IndexError> {
    let n_cols = other.len();
    let mask = cols.map(|c| col_mask(c, n_cols)).transpose()?;
    let mask = mask.as_deref();
    let category = IndexError {
        what: "category",
        len: n_cols,
    };
    out.par_chunks_mut(VALIDATE_CHUNK)
        .zip(indices.par_chunks(VALIDATE_CHUNK))
        .with_min_len(PARALLEL_MIN_ROWS / VALIDATE_CHUNK)
        .try_for_each(move |(out, indices)| {
            for (o, &index) in out.iter_mut().zip(indices) {
                if let Some(col) = category_column::<DROP>(index, drop_first) {
                    if col >= n_cols {
                        return Err(category);
                    }
                    if mask.is_none_or(|m| m[col]) {
                        *o += other[col];
                    }
                }
            }
            Ok(())
        })
}

/// `res[indices[k] - drop_first] += d[k]` over the rows `k` in `rows`: the
/// diagonal of the sandwich product `cat.T @ diag(d) @ cat`.
pub fn sandwich<F: Float, const DROP: bool>(
    indices: &[i32],
    d: &[F],
    rows: &[i32],
    res: &mut [F],
    drop_first: i32,
) -> Result<(), IndexError> {
    let n_rows = indices.len().min(d.len());
    let (indices, d) = (&indices[..n_rows], &d[..n_rows]);
    let n_cols = res.len();
    let row = IndexError {
        what: "row",
        len: n_rows,
    };
    let category = IndexError {
        what: "category",
        len: n_cols,
    };
    scatter_add(rows.len(), res, move |r, res| {
        let res = &mut res[..n_cols];
        for &k in &rows[r] {
            let k = k as usize;
            if k >= n_rows {
                return Err(row);
            }
            if let Some(col) = category_column::<DROP>(indices[k], drop_first) {
                if col >= n_cols {
                    return Err(category);
                }
                res[col] += d[k];
            }
        }
        Ok(())
    })
}

/// CSR `indices` and `indptr` of the matrix with the first `drop_first`
/// categories dropped and missing values skipped: one entry per row whose
/// shifted column index is non-negative.
pub fn csr_structure(indices: &[i32], drop_first: i32) -> (Vec<i32>, Vec<i32>) {
    let mut new_indices = Vec::with_capacity(indices.len());
    let mut indptr = Vec::with_capacity(indices.len() + 1);
    for &index in indices {
        indptr.push(new_indices.len() as i32);
        if let Some(col) = category_column::<true>(index, drop_first) {
            new_indices.push(col as i32);
        }
    }
    indptr.push(new_indices.len() as i32);
    (new_indices, indptr)
}

/// The entries of `d` at the rows kept by [`csr_structure`]: the `data` of
/// the CSR matrix `diag(d) @ cat`.
pub fn kept_data<T: Copy>(indices: &[i32], d: &[T], drop_first: i32) -> Vec<T> {
    indices
        .iter()
        .zip(d)
        .filter(|(&index, _)| category_column::<true>(index, drop_first).is_some())
        .map(|(_, &v)| v)
        .collect()
}

#[cfg(test)]
mod tests {
    //! The kernels against plain reference loops, over random inputs,
    //! subsets, both flavours and several thread counts.

    use super::super::testing::{assert_close, with_threads, Rng, THREADS};
    use super::*;

    fn column(index: i32, drop_first: i32) -> Option<usize> {
        let col = index - drop_first;
        (col >= 0).then_some(col as usize)
    }

    fn reference_transpose_matvec(
        indices: &[i32],
        other: &[f64],
        n_cols: usize,
        rows: Option<&[i32]>,
        cols: Option<&[i32]>,
        drop_first: i32,
    ) -> Vec<f64> {
        let mut out = vec![0.0; n_cols];
        let all_rows: Vec<i32> = (0..indices.len() as i32).collect();
        for &k in rows.unwrap_or(&all_rows) {
            if let Some(col) = column(indices[k as usize], drop_first) {
                if cols.is_none_or(|c| c.contains(&(col as i32))) {
                    out[col] += other[k as usize];
                }
            }
        }
        out
    }

    fn reference_matvec(
        indices: &[i32],
        other: &[f64],
        cols: Option<&[i32]>,
        drop_first: i32,
    ) -> Vec<f64> {
        indices
            .iter()
            .map(|&index| match column(index, drop_first) {
                Some(col) if cols.is_none_or(|c| c.contains(&(col as i32))) => other[col],
                _ => 0.0,
            })
            .collect()
    }

    fn reference_sandwich(
        indices: &[i32],
        d: &[f64],
        rows: &[i32],
        n_cols: usize,
        drop_first: i32,
    ) -> Vec<f64> {
        let mut res = vec![0.0; n_cols];
        for &k in rows {
            if let Some(col) = column(indices[k as usize], drop_first) {
                res[col] += d[k as usize];
            }
        }
        res
    }

    /// Random categories, with some `-1` (missing) when `missing`.
    fn categories(rng: &mut Rng, n: usize, n_cats: usize, missing: bool) -> Vec<i32> {
        (0..n)
            .map(|_| {
                if missing && rng.uniform() < 0.1 {
                    -1
                } else {
                    rng.below(n_cats) as i32
                }
            })
            .collect()
    }

    #[test]
    fn kernels_match_reference() {
        let mut rng = Rng::new(7);
        for &(n, n_cats) in &[(50usize, 4usize), (40_000, 30), (70_000, 1000)] {
            for drop_first in [0, 1] {
                let indices = categories(&mut rng, n, n_cats, drop_first == 1);
                let n_cols = n_cats - drop_first as usize;
                let other = rng.values(n, 3.0);
                let v_cols = rng.values(n_cols, 1.0);
                let rows = rng.subset(n);
                let cols = rng.subset(n_cols);
                for threads in THREADS {
                    let mut first: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
                    for _ in 0..2 {
                        let run = with_threads(threads, || {
                            let mut t = vec![0.0; n_cols];
                            let mut m = vec![0.0; n];
                            let mut s = vec![0.0; n_cols];
                            if drop_first == 1 {
                                transpose_matvec::<f64, true>(
                                    &indices,
                                    &other,
                                    &mut t,
                                    Some(&rows),
                                    Some(&cols),
                                    1,
                                )
                                .unwrap();
                                matvec::<f64, true>(&indices, &v_cols, &mut m, Some(&cols), 1)
                                    .unwrap();
                                sandwich::<f64, true>(&indices, &other, &rows, &mut s, 1).unwrap();
                            } else {
                                transpose_matvec::<f64, false>(
                                    &indices, &other, &mut t, None, None, 0,
                                )
                                .unwrap();
                                matvec::<f64, false>(&indices, &v_cols, &mut m, None, 0).unwrap();
                                sandwich::<f64, false>(&indices, &other, &rows, &mut s, 0).unwrap();
                            }
                            (t, m, s)
                        });
                        match &first {
                            None => {
                                let (rows_ref, cols_ref) = if drop_first == 1 {
                                    (Some(&rows[..]), Some(&cols[..]))
                                } else {
                                    (None, None)
                                };
                                assert_close(
                                    &run.0,
                                    &reference_transpose_matvec(
                                        &indices, &other, n_cols, rows_ref, cols_ref, drop_first,
                                    ),
                                    1e-12,
                                    "transpose_matvec",
                                );
                                assert_close(
                                    &run.1,
                                    &reference_matvec(&indices, &v_cols, cols_ref, drop_first),
                                    0.0,
                                    "matvec",
                                );
                                assert_close(
                                    &run.2,
                                    &reference_sandwich(
                                        &indices, &other, &rows, n_cols, drop_first,
                                    ),
                                    1e-12,
                                    "sandwich",
                                );
                                first = Some(run);
                            }
                            Some(f) => {
                                assert!(*f == run, "run-to-run difference at {threads} threads")
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bad_indices_are_errors() {
        let indices = [0, 1, 5];
        let v = [1.0; 3];
        let mut out = [0.0; 3];
        assert!(transpose_matvec::<f64, false>(&indices, &v, &mut out, None, None, 0).is_err());
        assert!(
            transpose_matvec::<f64, true>(&indices, &v, &mut out, Some(&[0, 7]), None, 1).is_err()
        );
        assert!(matvec::<f64, false>(&indices, &v, &mut out, None, 0).is_err());
        assert!(sandwich::<f64, false>(&indices, &v, &[0, 1, 2], &mut out, 0).is_err());
        assert!(col_mask(&[0, 3], 3).is_err());
    }

    #[test]
    fn csr_structure_drops_and_skips() {
        let (indices, indptr) = csr_structure(&[2, 0, -1, 1], 1);
        assert_eq!(indices, [1, 0]);
        assert_eq!(indptr, [0, 1, 1, 1, 2]);
        assert_eq!(
            kept_data(&[2, 0, -1, 1], &[1.0, 2.0, 3.0, 4.0], 1),
            [1.0, 4.0]
        );
    }
}
