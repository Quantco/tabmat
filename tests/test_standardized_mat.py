import numpy as np
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
expected_mat = col_mult[None, :] * sp_mat.A + col_shift[None, :]
standardized_mat = tm.StandardizedMatrix(sp_mat, col_shift, col_mult)


def test_setup_and_densify_col():
    assert standardized_mat.A.shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(standardized_mat.A, expected_mat)


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
