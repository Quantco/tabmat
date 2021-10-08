import numpy as np
import pandas as pd
import pytest

import tabmat as tm


@pytest.fixture()
def X():
    df = pd.read_pickle("tests/real_matrix.pkl")
    X_split = tm.from_pandas(df, np.float64)
    wts = np.ones(df.shape[0]) / df.shape[0]
    X_std = X_split.standardize(wts, True, True)[0]
    return X_std


def test_full_sandwich(X):
    X_dense = tm.DenseMatrix(X.toarray())
    r = np.random.rand(X.shape[0])
    simple = X_dense.sandwich(r)
    fancy = X.sandwich(r)
    np.testing.assert_almost_equal(simple, fancy, 12)


def test_split_sandwich_rows_cols(X):
    X_split = X.mat
    X_split_dense = tm.DenseMatrix(X_split.toarray())
    r = np.random.rand(X.shape[0])
    rows = np.arange(X.shape[0])
    cols = np.arange(X.shape[1])
    simple = X_split_dense.sandwich(r, rows, cols)
    fancy = X_split.sandwich(r, rows, cols)
    np.testing.assert_almost_equal(simple, fancy, 12)
