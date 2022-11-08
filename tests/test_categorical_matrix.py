import numpy as np
import pandas as pd
import pytest

from tabmat.categorical_matrix import CategoricalMatrix


@pytest.fixture
def cat_vec():
    m = 10
    seed = 0
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1, 2, np.inf, -np.inf], size=m)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize("drop_first", [True, False])
def test_recover_orig(cat_vec, vec_dtype, drop_first):
    orig_recovered = CategoricalMatrix(cat_vec, drop_first=drop_first).recover_orig()
    np.testing.assert_equal(orig_recovered, cat_vec)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize("drop_first", [True, False])
def test_csr_matvec_categorical(cat_vec, vec_dtype, drop_first):
    mat = pd.get_dummies(cat_vec, drop_first=drop_first, dtype="uint8")
    cat_mat = CategoricalMatrix(cat_vec, drop_first=drop_first)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = cat_mat.matvec(vec)
    np.testing.assert_allclose(res, cat_mat.A.dot(vec))


@pytest.mark.parametrize("drop_first", [True, False])
def test_tocsr(cat_vec, drop_first):
    cat_mat = CategoricalMatrix(cat_vec, drop_first=drop_first)
    res = cat_mat.tocsr().A
    expected = pd.get_dummies(cat_vec, drop_first=drop_first, dtype="uint8")
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("drop_first", [True, False])
def test_transpose_matvec(cat_vec, drop_first):
    cat_mat = CategoricalMatrix(cat_vec, drop_first=drop_first)
    other = np.random.random(cat_mat.shape[0])
    res = cat_mat.transpose_matvec(other)
    expected = pd.get_dummies(cat_vec, drop_first=drop_first, dtype="uint8").T.dot(
        other
    )
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("drop_first", [True, False])
def test_multiply(cat_vec, drop_first):
    cat_mat = CategoricalMatrix(cat_vec, drop_first=drop_first)
    other = np.arange(len(cat_vec))[:, None]
    actual = cat_mat.multiply(other)
    expected = pd.get_dummies(cat_vec, drop_first=drop_first, dtype="uint8") * other
    np.testing.assert_allclose(actual.A, expected)


@pytest.mark.parametrize("mi_element", [np.nan, None])
def test_nulls(mi_element):
    vec = [0, mi_element, 1]
    with pytest.raises(ValueError, match="Categorical data can't have missing values"):
        CategoricalMatrix(vec)


@pytest.mark.parametrize("drop_first", [True, False])
def test_categorical_indexing(drop_first):
    catvec = [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3]
    mat = CategoricalMatrix(catvec, drop_first=drop_first)
    expected = pd.get_dummies(catvec, drop_first=drop_first).to_numpy()[:, [0, 1]]
    np.testing.assert_allclose(mat[:, [0, 1]].A, expected)
