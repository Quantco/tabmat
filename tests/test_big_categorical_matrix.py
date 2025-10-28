import narwhals as nw
import numpy as np
import pandas as pd

import tabmat as tm


def test_transpose_matvec_does_not_crash():
    # With high n_categories CategoricalMatrix.indices is read-only.
    # As a result, transpose_matvec method crashed with "buffer
    # source array is read-only".

    n = 797_586
    n_categories = 58_059
    categories = [f"cat[{i}]" for i in range(n_categories)]
    indices = np.linspace(0, n_categories - 1, n).round().astype(int)
    cat_vec = pd.Series(pd.Categorical.from_codes(indices, categories=categories))
    cat_vec_nw = nw.from_native(cat_vec, allow_series=True)
    weights = np.ones(n)
    cat_matrix_tm = tm.CategoricalMatrix(cat_vec_nw)

    result = cat_matrix_tm.transpose_matvec(weights)

    assert result is not None


def make_categorical_matrix(n, n_categories):
    categories = [f"cat[{i}]" for i in range(n_categories)]
    indices = np.linspace(0, n_categories - 1, n).round().astype(int)
    cat_vec = pd.Series(pd.Categorical.from_codes(indices, categories=categories))
    cat_vec_nw = nw.from_native(cat_vec, allow_series=True)
    cat_matrix_tm = tm.CategoricalMatrix(cat_vec_nw)
    return cat_matrix_tm


def test_sandwich_cat_cat_does_not_crash():
    n = 797_586
    n_categories_A = 58_059
    n_categories_B = 2725

    weights = np.ones(n) / n
    cat_matrix_A = make_categorical_matrix(n, n_categories_A)
    cat_matrix_B = make_categorical_matrix(n, n_categories_B)
    rows = np.arange(n)
    cols_A = np.arange(n_categories_A)
    cols_B = np.arange(n_categories_B)

    res = cat_matrix_A._cross_categorical(cat_matrix_B, weights, rows, cols_A, cols_B)

    assert res is not None
