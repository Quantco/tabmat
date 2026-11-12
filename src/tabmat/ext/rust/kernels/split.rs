//! Kernels for the cross sandwich products of `tabmat.CategoricalMatrix`
//! with dense and categorical matrices, and the column bookkeeping of
//! `tabmat.SplitMatrix`.
//!
//! A categorical matrix is stored as one `int32` index per row, giving the
//! column of that row's single nonzero entry (`-1` for a missing value). As
//! in `categorical.rs`, each sandwich kernel comes in a plain and a
//! "complex" flavour: the complex one subtracts `drop_first` from the index
//! and skips rows whose adjusted column is negative (dropped reference
//! category or missing value), the plain one has no check at all. Both are
//! one kernel specialised through the `DROP` const generic.
//!
//! The sandwich products scatter `d[k]` (times a row of the dense matrix)
//! into the result row of each selected row's category. They are
//! parallelised with [`scatter_add`]: every thread scatters a contiguous
//! range of rows into its own copy of the result and the copies are summed
//! afterwards, in place of the atomic adds of the C++ version, so the result
//! does not depend on thread timing. The kernels validate `rows` in L1-sized
//! sub-chunks ([`checked_chunks`]) and then load the rows' data without
//! checks, since a check per load cost up to 50% in the kernels whose result
//! fits the cache, and a validation pass over the whole array cost 20% of
//! memory traffic in the parallel, bandwidth-bound case.

use super::{category_column, checked_chunks, scatter_add, Centered, Float, IndexError};

// ---------------------------------------------------------------------------
// sandwich_cat_dense: cat.T @ diag(d) @ (mat_j - center)[:, j_cols]
// ---------------------------------------------------------------------------

/// `res[k * nj + j] += d[k] * (mat_j[k, j_cols[j]] - mu[j_cols[j]])` for one
/// row `k`, `mu` the center vector or `None`. Two instantiations, so that
/// the uncentered product does not subtract zeros in its inner loop (which
/// measured 25% slower on a 5-column `mat_j`).
///
/// # Safety
/// `k` is a row of `mat_j`, every entry of `j_cols` a column, and `mu`
/// (if any) has one entry per column.
#[inline(always)]
unsafe fn add_row<F: Float, const CENTERED: bool>(
    mat_j: Centered<'_, F>,
    mu: &[F],
    k: usize,
    dk: F,
    j_cols: &[i32],
    res_row: &mut [F],
) {
    for (dst, &j) in res_row.iter_mut().zip(j_cols) {
        let j = j as usize;
        // SAFETY: the caller's contract.
        let v = unsafe { mat_j.get(k, j) };
        *dst += dk
            * if CENTERED {
                v - unsafe { *mu.get_unchecked(j) }
            } else {
                v
            };
    }
}

/// `res[cat[k], :] += d[k] * (mat_j - center)[k, j_cols]` over the rows `k`
/// in `rows`, for `mat_j` any strided view with an optional per-column
/// center. `res` is `i_ncol x len(j_cols)`, row-major and zeroed; `j_cols`
/// has been checked against the columns of `mat_j`, `rows` and the
/// categories are checked chunk by chunk.
pub fn sandwich_cat_dense<F: Float, const DROP: bool>(
    mat_j: Centered<'_, F>,
    indices: &[i32],
    d: &[F],
    rows: &[i32],
    j_cols: &[i32],
    drop_first: i32,
    res: &mut [F],
) -> Result<(), IndexError> {
    let nj = j_cols.len();
    let n_rows = indices.len().min(d.len()).min(mat_j.nrows());
    let n_cats = res.len() / nj.max(1);
    let centered = mat_j.center.is_some();
    let mu = mat_j.center.unwrap_or(&[]);
    let category = IndexError {
        what: "category",
        len: n_cats,
    };
    scatter_add(rows.len(), res, move |r, res| {
        for rows in checked_chunks(&rows[r], n_rows, "row") {
            for &k in rows? {
                let k = k as usize;
                // SAFETY (all unchecked loads in this loop): the chunk of
                // `rows` was just checked against the lengths of `indices`
                // and `d` and the rows of `mat_j`; `j_cols` was checked by
                // the caller, and `mu` has an entry per column of `mat_j`.
                let (index, dk) = unsafe { (*indices.get_unchecked(k), *d.get_unchecked(k)) };
                let Some(i) = category_column::<DROP>(index, drop_first) else {
                    continue;
                };
                if i >= n_cats {
                    return Err(category);
                }
                let res_row = &mut res[i * nj..(i + 1) * nj];
                if centered {
                    unsafe { add_row::<F, true>(mat_j, mu, k, dk, j_cols, res_row) };
                } else {
                    unsafe { add_row::<F, false>(mat_j, mu, k, dk, j_cols, res_row) };
                }
            }
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// sandwich_cat_cat: cat_i.T @ diag(d) @ cat_j
// ---------------------------------------------------------------------------

/// `res[cat_i[k], cat_j[k]] += d[k]` over the rows `k` in `rows`. `res` is
/// `i_ncol x j_ncol`, row-major and zeroed; the categories are checked in
/// the loop.
#[allow(clippy::too_many_arguments)]
pub fn sandwich_cat_cat<F: Float, const DROP: bool>(
    i_indices: &[i32],
    j_indices: &[i32],
    d: &[F],
    rows: &[i32],
    j_ncol: usize,
    i_drop_first: i32,
    j_drop_first: i32,
    res: &mut [F],
) -> Result<(), IndexError> {
    let n_rows = i_indices.len().min(j_indices.len()).min(d.len());
    let i_ncol = res.len() / j_ncol.max(1);
    let (i_cat, j_cat) = (
        IndexError {
            what: "category",
            len: i_ncol,
        },
        IndexError {
            what: "category",
            len: j_ncol,
        },
    );
    scatter_add(rows.len(), res, move |r, res| {
        for rows in checked_chunks(&rows[r], n_rows, "row") {
            for &k in rows? {
                let k = k as usize;
                // SAFETY: the chunk of `rows` was just checked against the
                // three lengths.
                let (ii, jj, dk) = unsafe {
                    (
                        *i_indices.get_unchecked(k),
                        *j_indices.get_unchecked(k),
                        *d.get_unchecked(k),
                    )
                };
                let Some(i) = category_column::<DROP>(ii, i_drop_first) else {
                    continue;
                };
                let Some(j) = category_column::<DROP>(jj, j_drop_first) else {
                    continue;
                };
                if i >= i_ncol {
                    return Err(i_cat);
                }
                if j >= j_ncol {
                    return Err(j_cat);
                }
                res[i * j_ncol + j] += dk;
            }
        }
        Ok(())
    })
}

/// `cat_i[rows].T @ diag(d[rows]) @ cat_j[rows]` restricted to the columns
/// `i_cols` and `j_cols`: a single-threaded reference implementation kept
/// for the tests. The restriction is applied by masking, so the result keeps
/// the full `i_ncol x j_ncol` shape; rows and categories outside the matrix
/// raise `IndexError`.
#[allow(clippy::too_many_arguments)]
pub fn sandwich_cat_cat_limited_rows_cols(
    i_indices: &[i32],
    j_indices: &[i32],
    d: &[f64],
    rows: &[i32],
    i_ncol: usize,
    j_ncol: usize,
    i_cols: &[i32],
    j_cols: &[i32],
) -> Result<Vec<f64>, IndexError> {
    let mut i_included = vec![false; i_ncol];
    for &c in i_cols {
        *i_included.get_mut(c as usize).ok_or(IndexError {
            what: "column",
            len: i_ncol,
        })? = true;
    }
    let mut j_included = vec![false; j_ncol];
    for &c in j_cols {
        *j_included.get_mut(c as usize).ok_or(IndexError {
            what: "column",
            len: j_ncol,
        })? = true;
    }
    let n_rows = i_indices.len().min(j_indices.len()).min(d.len());
    let mut res = vec![0.0f64; i_ncol * j_ncol];
    for &k in rows {
        let k = k as usize;
        if k >= n_rows {
            return Err(IndexError {
                what: "row",
                len: n_rows,
            });
        }
        let i = i_indices[k] as usize;
        if i_included.get(i).copied().unwrap_or(false) {
            let j = j_indices[k] as usize;
            if j_included.get(j).copied().unwrap_or(false) {
                res[i * j_ncol + j] += d[k];
            }
        }
    }
    Ok(res)
}

// ---------------------------------------------------------------------------
// SplitMatrix column bookkeeping
// ---------------------------------------------------------------------------

/// For a `SplitMatrix` whose sub-matrices own the (sorted) global columns
/// `indices[i]`, and a sorted column subset `cols`: the positions in `cols`
/// of the columns of each sub-matrix (`subset_cols_indices`) and the
/// positions of those columns within each sub-matrix (`subset_cols`), such
/// that `indices[i][subset_cols[i]] == cols[subset_cols_indices[i]]`.
pub fn split_col_subsets(indices: &[&[i64]], cols: &[i32]) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let n_matrices = indices.len();
    let mut next_subset_idx = vec![0usize; n_matrices];
    let mut subset_cols_indices: Vec<Vec<i32>> = vec![Vec::new(); n_matrices];
    let mut subset_cols: Vec<Vec<i32>> = vec![Vec::new(); n_matrices];
    for (i, &col) in cols.iter().enumerate() {
        let col = i64::from(col);
        for j in 0..n_matrices {
            let idx = indices[j];
            let next = &mut next_subset_idx[j];
            while *next < idx.len() && idx[*next] < col {
                *next += 1;
            }
            if *next < idx.len() && idx[*next] == col {
                subset_cols_indices[j].push(i as i32);
                subset_cols[j].push(*next as i32);
                *next += 1;
                break;
            }
        }
    }
    (subset_cols_indices, subset_cols)
}

/// Whether `a` is sorted in nondecreasing order.
pub fn is_sorted(a: &[i64]) -> bool {
    a.windows(2).all(|w| w[1] >= w[0])
}

#[cfg(test)]
mod tests {
    //! The kernels against plain reference loops, over random categories
    //! (with missing values), subsets, both layouts of the dense operand,
    //! with and without a center, and several thread counts.

    use ndarray::Array2;

    use super::super::check_indices;
    use super::super::testing::{assert_close, with_threads, Rng, THREADS};
    use super::*;

    fn column(index: i32, drop_first: i32) -> Option<usize> {
        let col = index - drop_first;
        (col >= 0).then_some(col as usize)
    }

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
    fn cross_products_match_reference() {
        let mut rng = Rng::new(31);
        for &(n, nci, ncj, nd) in &[
            (60_000usize, 20usize, 7usize, 5usize),
            (3000, 300, 400, 3),
            (500, 4, 4, 2),
        ] {
            for drop in [0, 1] {
                let i_idx = categories(&mut rng, n, nci, drop == 1);
                let j_idx = categories(&mut rng, n, ncj, drop == 1);
                let (i_ncol, j_ncol) = (nci - drop as usize, ncj - drop as usize);
                let d = rng.values(n, 1.0);
                let x = Array2::from_shape_vec((n, nd), rng.values(n * nd, 4.0)).unwrap();
                let x_f = x.t().as_standard_layout().t().to_owned();
                let center: Vec<f64> = (0..nd).map(|j| x.column(j).mean().unwrap()).collect();
                let all_rows: Vec<i32> = (0..n as i32).collect();
                let sub_rows = rng.subset(n);
                let all_j: Vec<i32> = (0..nd as i32).collect();
                let sub_j = rng.subset(nd);
                for (rows, j_cols) in [(&all_rows, &all_j), (&sub_rows, &sub_j)] {
                    check_indices(j_cols, nd, "column").unwrap();
                    // cat x cat reference
                    let mut exp_cc = vec![0.0; i_ncol * j_ncol];
                    for &k in rows {
                        let k = k as usize;
                        if let (Some(i), Some(j)) = (column(i_idx[k], drop), column(j_idx[k], drop))
                        {
                            exp_cc[i * j_ncol + j] += d[k];
                        }
                    }
                    for threads in THREADS {
                        let mut first = None;
                        for _ in 0..2 {
                            let mut res = vec![0.0; i_ncol * j_ncol];
                            with_threads(threads, || {
                                if drop == 1 {
                                    sandwich_cat_cat::<f64, true>(
                                        &i_idx, &j_idx, &d, rows, j_ncol, 1, 1, &mut res,
                                    )
                                    .unwrap()
                                } else {
                                    sandwich_cat_cat::<f64, false>(
                                        &i_idx, &j_idx, &d, rows, j_ncol, 0, 0, &mut res,
                                    )
                                    .unwrap()
                                }
                            });
                            assert_close(&res, &exp_cc, 1e-12, "cat_cat");
                            match &first {
                                None => first = Some(res),
                                Some(f) => assert!(*f == res, "run-to-run difference"),
                            }
                        }
                    }
                    // cat x dense reference, both layouts, with and without center
                    for (name, xa) in [("C", &x), ("F", &x_f)] {
                        for center in [None, Some(&center[..])] {
                            let nj = j_cols.len();
                            let mut exp = vec![0.0; i_ncol * nj];
                            for &k in rows {
                                let k = k as usize;
                                if let Some(i) = column(i_idx[k], drop) {
                                    for (cj, &j) in j_cols.iter().enumerate() {
                                        let j = j as usize;
                                        exp[i * nj + cj] +=
                                            d[k] * (xa[[k, j]] - center.map_or(0.0, |c| c[j]));
                                    }
                                }
                            }
                            let op = Centered {
                                x: xa.view(),
                                center,
                            };
                            for threads in THREADS {
                                let mut first = None;
                                for _ in 0..2 {
                                    let mut res = vec![0.0; i_ncol * nj];
                                    with_threads(threads, || {
                                        if drop == 1 {
                                            sandwich_cat_dense::<f64, true>(
                                                op, &i_idx, &d, rows, j_cols, 1, &mut res,
                                            )
                                            .unwrap()
                                        } else {
                                            sandwich_cat_dense::<f64, false>(
                                                op, &i_idx, &d, rows, j_cols, 0, &mut res,
                                            )
                                            .unwrap()
                                        }
                                    });
                                    assert_close(&res, &exp, 1e-12, &format!("cat_dense {name}"));
                                    match &first {
                                        None => first = Some(res),
                                        Some(f) => assert!(*f == res, "run-to-run difference"),
                                    }
                                }
                            }
                        }
                    }
                }
                if drop == 0 {
                    let i_cols = rng.subset(i_ncol);
                    let j_cols = rng.subset(j_ncol);
                    let res = sandwich_cat_cat_limited_rows_cols(
                        &i_idx, &j_idx, &d, &sub_rows, i_ncol, j_ncol, &i_cols, &j_cols,
                    )
                    .unwrap();
                    let mut exp = vec![0.0; i_ncol * j_ncol];
                    for &k in &sub_rows {
                        let (i, j) = (i_idx[k as usize] as usize, j_idx[k as usize] as usize);
                        if i_cols.contains(&(i as i32)) && j_cols.contains(&(j as i32)) {
                            exp[i * j_ncol + j] += d[k as usize];
                        }
                    }
                    assert_close(&res, &exp, 1e-12, "limited");
                }
            }
        }
    }

    #[test]
    fn bad_rows_and_categories_are_errors() {
        let x = Array2::<f64>::zeros((3, 2));
        let op = Centered {
            x: x.view(),
            center: None,
        };
        let d = [1.0; 3];
        let mut res = vec![0.0; 4];
        assert!(sandwich_cat_dense::<f64, false>(
            op,
            &[0, 1, 1],
            &d,
            &[0, 5],
            &[0, 1],
            0,
            &mut res
        )
        .is_err());
        assert!(sandwich_cat_dense::<f64, false>(
            op,
            &[0, 1, 7],
            &d,
            &[0, 1, 2],
            &[0, 1],
            0,
            &mut res
        )
        .is_err());
        assert!(sandwich_cat_cat::<f64, false>(
            &[0, 1, 1],
            &[0, 9, 1],
            &d,
            &[0, 1, 2],
            2,
            0,
            0,
            &mut res
        )
        .is_err());
        assert!(sandwich_cat_cat::<f64, false>(
            &[0, 1, 1],
            &[0, 1, 1],
            &d,
            &[3],
            2,
            0,
            0,
            &mut res
        )
        .is_err());
        assert!(sandwich_cat_cat_limited_rows_cols(
            &[0, 1],
            &[0, 1],
            &[1.0, 1.0],
            &[0, 1],
            2,
            2,
            &[0, 5],
            &[0]
        )
        .is_err());
    }

    #[test]
    fn column_bookkeeping() {
        let indices: [&[i64]; 2] = [&[0, 2, 5], &[1, 3, 4]];
        let (sci, sc) = split_col_subsets(&indices, &[0, 1, 3, 5]);
        assert_eq!(sci, [vec![0, 3], vec![1, 2]]);
        assert_eq!(sc, [vec![0, 2], vec![0, 1]]);
        assert!(is_sorted(&[1, 1, 2]));
        assert!(!is_sorted(&[2, 1]));
    }
}
