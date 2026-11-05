import numpy as np
import pytest
from scipy import sparse as sps

import tabmat as tm

# There's a lot more testing happening in the generic setting in
# test_matrices.py

np.random.seed(0)
n_rows = 8
n_cols = 5
sp_mat = tm.SparseMatrix(sps.random(n_rows, n_cols, density=0.8))
col_shift = np.random.uniform(0, 1, n_cols)
col_mult = np.random.uniform(0.5, 1.5, n_cols)
expected_mat = col_mult[None, :] * sp_mat.toarray() + col_shift[None, :]
standardized_mat = tm.StandardizedMatrix(sp_mat, col_shift, col_mult)


def test_setup_and_densify_col():
    assert standardized_mat.toarray().shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(standardized_mat.toarray(), expected_mat)


def test_standardized_matvec():
    v = np.random.rand(standardized_mat.shape[1])
    np.testing.assert_almost_equal(standardized_mat.matvec(v), expected_mat.dot(v))


def test_standardized_transpose_matvec():
    v = np.random.rand(standardized_mat.shape[0])
    np.testing.assert_almost_equal(
        standardized_mat.transpose_matvec(v), v @ expected_mat
    )


def test_standardized_sandwich():
    v = np.random.rand(standardized_mat.shape[0])
    expected = (expected_mat.T * v) @ expected_mat
    np.testing.assert_almost_equal(standardized_mat.sandwich(v), expected)


def test_zero_sd_cols():
    n_rows = 100
    weights = np.ones(n_rows) / n_rows
    X = tm.DenseMatrix(np.ones([n_rows, 1])).standardize(weights, True, True)[0]
    assert X.mult == 1


def _large_mean_dense(dtype, offset, n=20_000, p=6, seed=0):
    """Dense data whose column means are much larger than their std (GH #414)."""
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, p)) * rng.uniform(0.5, 2.0, p) + offset).astype(dtype)
    weights = np.full(n, 1 / n, dtype=dtype)
    mat, means, _ = tm.DenseMatrix(X).standardize(weights, True, True)
    # Exact standardized data: center first, in float64.
    Z = (X.astype(np.float64) - means.astype(np.float64)) * mat.mult.astype(np.float64)
    # Weights that depend on the data, like the IRLS weights in glum.
    d = (1 + 0.5 * np.tanh(Z[:, 0])).astype(dtype)
    return mat, Z, d


def _assert_sandwich_close(actual, Z, d, rtol):
    expected = Z.T @ (d.astype(np.float64)[:, None] * Z)
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * np.abs(expected).max()
    )


@pytest.mark.parametrize(
    "dtype, offset, rtol", [(np.float64, 1e6, 1e-12), (np.float32, 1e3, 1e-5)]
)
def test_dense_sandwich_large_mean_is_accurate(dtype, offset, rtol):
    mat, Z, d = _large_mean_dense(dtype, offset)
    _assert_sandwich_close(mat.sandwich(d), Z, d, rtol)

    rows = np.arange(0, Z.shape[0], 3, dtype=np.int32)
    cols = np.array([4, 0, 2], dtype=np.int32)
    _assert_sandwich_close(
        mat.sandwich(d, rows, cols), Z[np.ix_(rows, cols)], d[rows], rtol
    )


def test_dense_sandwich_large_mean_derived_matrices():
    mat, Z, d = _large_mean_dense(np.float64, 1e6)
    _assert_sandwich_close(mat.getcol(2).sandwich(d), Z[:, [2]], d, 1e-12)
    _assert_sandwich_close(mat[:, [3, 1]].sandwich(d), Z[:, [3, 1]], d, 1e-12)
    _assert_sandwich_close(mat[:5000, :].sandwich(d[:5000]), Z[:5000], d[:5000], 1e-12)


def test_dense_sandwich_large_mean_astype():
    mat, _, d = _large_mean_dense(np.float64, 1e3)
    cast = mat.astype(np.float32)
    # The cast data and centers are the reference: float32 cannot represent
    # the original standardized values more precisely than that.
    X32 = cast.mat.toarray().astype(np.float64)
    center32 = cast._center.astype(np.float64)
    Z32 = (X32 - center32) * cast.mult.astype(np.float64)
    d32 = d.astype(np.float32)
    _assert_sandwich_close(cast.sandwich(d32), Z32, d32, 1e-5)


def test_dense_sandwich_large_mean_without_known_centers():
    # Built directly, so the centers have to be recovered from shift and mult.
    rng = np.random.default_rng(1)
    n, p = 20_000, 4
    X = rng.standard_normal((n, p)) + 1e5
    center = X.mean(axis=0)
    mult = 1 / X.std(axis=0)
    mat = tm.StandardizedMatrix(tm.DenseMatrix(X), -center * mult, mult)
    Z = (X - center) * mult
    d = rng.uniform(0.5, 1.5, n)
    _assert_sandwich_close(mat.sandwich(d), Z, d, 1e-9)


def test_dense_sandwich_follows_shift_changes():
    rng = np.random.default_rng(2)
    X = tm.DenseMatrix(rng.standard_normal((500, 3)) + 10)
    mat = X.standardize(np.full(500, 1 / 500), True, True)[0]
    mat.shift += np.array([1.0, -2.0, 0.5])
    d = rng.uniform(0.5, 1.5, 500)
    _assert_sandwich_close(mat.sandwich(d), mat.toarray(), d, 1e-12)


def test_dense_sandwich_zero_mult():
    # A zero multiplier leaves no center to recover; the expansion still works.
    rng = np.random.default_rng(3)
    mat = tm.StandardizedMatrix(
        tm.DenseMatrix(rng.standard_normal((200, 3))), [1.0, 2.0, 3.0], [1.0, 0.0, 2.0]
    )
    d = rng.uniform(0.5, 1.5, 200)
    _assert_sandwich_close(mat.sandwich(d), mat.toarray(), d, 1e-12)


def test_astype_keeps_mult():
    rng = np.random.default_rng(4)
    X = tm.DenseMatrix(rng.standard_normal((100, 3)) * [1.0, 10.0, 100.0] + 5)
    mat = X.standardize(np.full(100, 0.01), True, True)[0]
    np.testing.assert_allclose(mat.astype(np.float64).toarray(), mat.toarray())
