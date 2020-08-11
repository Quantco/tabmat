import numpy as np
import pandas as pd
import pytest

from quantcore.matrix.categorical_matrix import CategoricalMatrix


@pytest.fixture
def cat_vec():
    m = 10
    seed = 0
    np.random.seed(seed)
    return np.random.choice(np.arange(4, dtype=int), m)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
def test_recover_orig(cat_vec, vec_dtype):
    orig_recovered = CategoricalMatrix(cat_vec).recover_orig()
    np.testing.assert_equal(orig_recovered, cat_vec)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
def test_csr_matvec_categorical(cat_vec, vec_dtype):
    mat = pd.get_dummies(cat_vec)
    cat_mat = CategoricalMatrix(cat_vec)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = cat_mat.matvec(vec)
    np.testing.assert_allclose(res, cat_mat.A.dot(vec))


def test_tocsr(cat_vec):
    cat_mat = CategoricalMatrix(cat_vec)
    res = cat_mat.tocsr().A
    expected = pd.get_dummies(cat_vec)
    np.testing.assert_allclose(res, expected)


def test_transpose_matvec(cat_vec):
    cat_mat = CategoricalMatrix(cat_vec)
    other = np.random.random(cat_mat.shape[0])
    res = cat_mat.transpose_matvec(other)
    expected = cat_mat.A.T.dot(other)
    np.testing.assert_allclose(res, expected)
