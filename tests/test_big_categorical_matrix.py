import narwhals as nw
import numpy as np
import pandas as pd
import pytest

import tabmat as tm
from tabmat.ext.split import _sandwich_cat_cat_limited_rows_cols


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


def make_categorical_matrix(n, n_categories, **categorical_kwargs):
    categories = [f"cat[{i}]" for i in range(n_categories)]
    indices = np.linspace(0, n_categories - 1, n).round().astype(int)
    cat_vec = pd.Series(pd.Categorical.from_codes(indices, categories=categories))
    cat_vec_nw = nw.from_native(cat_vec, allow_series=True)
    cat_matrix_tm = tm.CategoricalMatrix(cat_vec_nw, **categorical_kwargs)
    return cat_matrix_tm


def test_sandwich_cat_cat_does_not_crash():
    n = 797_586

    for n_cat_A, n_cat_B in [(58_059, 2725), (2725, 58_059)]:
        weights = np.ones(n) / n
        cat_matrix_A = make_categorical_matrix(n, n_cat_A)
        cat_matrix_B = make_categorical_matrix(n, n_cat_B)
        rows = np.arange(n)
        cols_A = np.arange(n_cat_A)
        cols_B = np.arange(n_cat_B)

        res = cat_matrix_A._cross_categorical(
            cat_matrix_B, weights, rows, cols_A, cols_B
        )

    assert res is not None


@pytest.fixture(scope="module")
def big_categorical_matrix():
    n = 797_586
    n_categories = 58_059
    mat = make_categorical_matrix(n, n_categories)
    assert not mat.indices.flags.writeable
    return mat


@pytest.fixture(scope="module")
def big_categorical_matrix_drop_first():
    n = 797_586
    n_categories = 58_059
    mat = make_categorical_matrix(n, n_categories, drop_first=True)
    assert not mat.indices.flags.writeable
    return mat


def test_cross_dense_does_not_crash(big_categorical_matrix):
    # With high n_categories CategoricalMatrix.indices is read-only.
    # This should not prevent cross-sandwich with a dense matrix.
    cat_matrix = big_categorical_matrix
    n_rows = cat_matrix.shape[0]
    n_dense_cols = 10

    dense_array = np.ones((n_rows, n_dense_cols))
    dense_matrix = tm.DenseMatrix(dense_array)
    weights = np.ones(n_rows) / n_rows
    rows = np.arange(n_rows)
    L_cols = np.arange(cat_matrix.shape[1])
    R_cols = np.arange(n_dense_cols)

    res = cat_matrix._cross_sandwich(dense_matrix, weights, rows, L_cols, R_cols)

    assert res is not None


def test_multiply_does_not_crash_with_drop_first(big_categorical_matrix_drop_first):
    # When drop_first is True and indices are read-only, multiply()
    # currently goes through ext.multiply_complex, which must accept
    # read-only index buffers.
    cat_matrix = big_categorical_matrix_drop_first
    n_rows = cat_matrix.shape[0]
    weights = np.ones(n_rows)

    res = cat_matrix.multiply(weights)

    assert res is not None


def test_tocsr_does_not_crash_with_drop_first(big_categorical_matrix_drop_first):
    # When drop_first is True and indices are read-only, tocsr()
    # currently goes through ext.subset_categorical_complex, which must
    # accept read-only index buffers.
    cat_matrix = big_categorical_matrix_drop_first

    csr = cat_matrix.tocsr()

    assert csr.shape == cat_matrix.shape


def test_sandwich_cat_cat_limited_rows_cols_does_not_crash(
    big_categorical_matrix,
):
    # Directly exercise the slower _sandwich_cat_cat_limited_rows_cols helper
    # with read-only indices.
    n = 797_586

    for n_cat_A, n_cat_B in [(58_059, 2725), (2725, 58_059)]:
        cat_matrix_A = make_categorical_matrix(n, n_cat_A)
        cat_matrix_B = make_categorical_matrix(n, n_cat_B)
        i_indices = cat_matrix_A.indices
        j_indices = cat_matrix_B.indices
        n_rows = n
        i_ncol = cat_matrix_A.shape[1]
        j_ncol = cat_matrix_B.shape[1]

        d = np.ones(n_rows)
        rows = np.arange(n_rows, dtype=np.int32)
        i_cols = np.arange(i_ncol, dtype=np.int32)
        j_cols = np.arange(j_ncol, dtype=np.int32)

        res = _sandwich_cat_cat_limited_rows_cols(
            i_indices,
            j_indices,
            i_ncol,
            j_ncol,
            d,
            rows,
            i_cols,
            j_cols,
        )

    assert res is not None
