from typing import List, Optional, Union

import numpy as np
import pytest
import scipy.sparse as sps

import tabmat as tm
from tabmat.constructor import _split_sparse_and_dense_parts
from tabmat.dense_matrix import DenseMatrix
from tabmat.ext.sparse import csr_dense_sandwich
from tabmat.split_matrix import SplitMatrix

N = 100


def make_X() -> np.ndarray:
    X = np.zeros((N, 4))
    X[:, 0] = 1.0
    X[:10, 1] = 0.5
    X[-20:, 2] = 0.25
    X[:, 3] = 2.0
    return X


@pytest.fixture
def X() -> np.ndarray:
    return make_X()


def test_csc_to_split(X: np.ndarray):
    for T, D, S in [(0.05, 4, 0), (0.1, 3, 1), (0.2, 2, 2), (0.3, 2, 2), (1.0, 0, 4)]:
        dense, sparse, dense_ix, sparse_ix = _split_sparse_and_dense_parts(
            sps.csc_matrix(X), T
        )
        fully_dense = SplitMatrix([dense, sparse], [dense_ix, sparse_ix])
        if S == 0:
            assert fully_dense.indices[0].shape[0] == D
            assert len(fully_dense.indices) == 1
        elif D == 0:
            assert fully_dense.indices[0].shape[0] == S
            assert len(fully_dense.indices) == 1
        else:
            assert fully_dense.indices[0].shape[0] == D
            assert fully_dense.indices[1].shape[0] == S


def split_mat() -> SplitMatrix:
    X = make_X()
    threshold = 0.1
    cat_mat = tm.CategoricalMatrix(np.random.choice(range(4), X.shape[0]))
    dense, sparse, dense_ix, sparse_ix = _split_sparse_and_dense_parts(
        sps.csc_matrix(X), threshold
    )
    cat_start = 1 + max(dense_ix.max(), sparse_ix.max())
    mat = SplitMatrix(
        [dense, sparse, cat_mat],
        [dense_ix, sparse_ix, range(cat_start, cat_start + cat_mat.shape[1])],
    )
    return mat


def get_split_with_cat_components() -> List[
    Union[tm.SparseMatrix, tm.DenseMatrix, tm.CategoricalMatrix]
]:
    n_rows = 10
    np.random.seed(0)
    dense_1 = tm.DenseMatrix(np.random.random((n_rows, 3)))
    sparse_1 = tm.SparseMatrix(sps.random(n_rows, 3).tocsc())
    cat = tm.CategoricalMatrix(np.random.choice(range(3), n_rows))
    dense_2 = tm.DenseMatrix(np.random.random((n_rows, 3)))
    sparse_2 = tm.SparseMatrix(sps.random(n_rows, 3, density=0.5).tocsc())
    cat_2 = tm.CategoricalMatrix(np.random.choice(range(3), n_rows))
    return [dense_1, sparse_1, cat, dense_2, sparse_2, cat_2]


def split_with_cat() -> SplitMatrix:
    """Initialized with multiple sparse and dense parts and no indices."""
    return tm.SplitMatrix(get_split_with_cat_components())


def split_with_cat_64() -> SplitMatrix:
    mat = tm.SplitMatrix(get_split_with_cat_components())
    matrices = mat.matrices

    for i, mat_ in enumerate(mat.matrices):
        if isinstance(mat_, tm.SparseMatrix):
            matrices[i] = tm.SparseMatrix(
                (
                    mat_.data,
                    mat_.indices.astype(np.int64),
                    mat_.indptr.astype(np.int64),
                ),
                shape=mat_.shape,
            )
        elif isinstance(mat_, tm.DenseMatrix):
            matrices[i] = mat_.astype(np.float64)
    return tm.SplitMatrix(matrices, mat.indices)


@pytest.mark.parametrize("mat", [split_with_cat(), split_with_cat_64()])
def test_init(mat: SplitMatrix):
    assert len(mat.indices) == 4
    assert len(mat.matrices) == 4
    assert (mat.indices[0] == np.concatenate([np.arange(3), np.arange(9, 12)])).all()
    assert mat.matrices[0].shape == (10, 6)
    assert mat.matrices[1].shape == (10, 6)
    assert mat.matrices[2].shape == (10, 3)


def test_init_unsorted_indices():
    dense = tm.DenseMatrix(np.random.random((10, 3)))
    with pytest.raises(ValueError):
        tm.SplitMatrix([dense], [[1, 0, 2]])


@pytest.mark.parametrize(
    "Acols", [np.arange(2, dtype=np.int32), np.array([1], dtype=np.int32)]
)
@pytest.mark.parametrize(
    "Bcols",
    [
        np.arange(4, dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
    ],
)
def test_sandwich_sparse_dense(X: np.ndarray, Acols, Bcols):
    np.random.seed(0)
    n, k = X.shape
    d = np.random.random((n,))
    A = sps.random(n, 2).tocsr()
    rows = np.arange(d.shape[0], dtype=np.int32)
    result = csr_dense_sandwich(A, X, d, rows, Acols, Bcols)
    expected = A.T.A[Acols, :] @ np.diag(d) @ X[:, Bcols]
    np.testing.assert_allclose(result, expected)


# TODO: ensure cols are in order
@pytest.mark.parametrize("mat", [split_mat(), split_with_cat(), split_with_cat_64()])
@pytest.mark.parametrize(
    "cols",
    [None, [0], [1, 2, 3], [1, 5]],
)
def test_sandwich(mat: tm.SplitMatrix, cols):
    for _ in range(10):
        v = np.random.rand(mat.shape[0])
        y1 = mat.sandwich(v, cols=cols)
        mat_limited = mat.A if cols is None else mat.A[:, cols]
        y2 = (mat_limited.T * v[None, :]) @ mat_limited
        np.testing.assert_allclose(y1, y2, atol=1e-12)


@pytest.mark.parametrize("mat", [split_mat(), split_with_cat(), split_with_cat_64()])
@pytest.mark.parametrize("cols", [None, [0], [1, 2, 3], [1, 5]])
def test_split_col_subsets(mat: tm.SplitMatrix, cols):
    subset_cols_indices, subset_cols, n_cols = mat._split_col_subsets(cols)
    n_cols_correct = mat.shape[1] if cols is None else len(cols)

    def _get_lengths(vec_list: List[Optional[np.ndarray]]):
        return (
            mat_.shape[1] if v is None else len(v)
            for v, mat_ in zip(vec_list, mat.matrices)
        )

    assert n_cols == n_cols_correct
    assert sum(_get_lengths(subset_cols_indices)) == n_cols
    assert sum(_get_lengths(subset_cols)) == n_cols

    if cols is not None:
        cols = np.asarray(cols)

    for i in range(len(mat.indices)):
        if cols is not None:
            assert (
                mat.indices[i][subset_cols[i]] == cols[subset_cols_indices[i]]
            ).all()
        else:
            assert subset_cols[i] is None
            assert (mat.indices[i] == subset_cols_indices[i]).all()


def random_split_matrix(seed=0, n_rows=10, n_cols_per=3):
    if seed is not None:
        np.random.seed(seed)
    dense_1 = tm.DenseMatrix(np.random.random((n_rows, n_cols_per)))
    sparse = tm.SparseMatrix(sps.random(n_rows, n_cols_per).tocsc())
    cat = tm.CategoricalMatrix(np.random.choice(range(n_cols_per), n_rows))
    dense_2 = tm.DenseMatrix(np.random.random((n_rows, n_cols_per)))
    cat_2 = tm.CategoricalMatrix(np.random.choice(range(n_cols_per), n_rows))
    mat = tm.SplitMatrix([dense_1, sparse, cat, dense_2, cat_2])
    return mat


def many_random_tests(checker):
    for i in range(10):
        mat = random_split_matrix(
            seed=(1 if i == 0 else None),
            n_rows=np.random.randint(130),
            n_cols_per=1 + np.random.randint(10),
        )
        checker(mat)


def test_sandwich_many_types():
    def check(mat):
        d = np.random.random(mat.shape[0])
        res = mat.sandwich(d)
        expected = (mat.A.T * d[None, :]) @ mat.A
        np.testing.assert_allclose(res, expected)

    many_random_tests(check)


def test_transpose_matvec_many_types():
    def check(mat):
        d = np.random.random(mat.shape[0])
        res = mat.transpose_matvec(d)
        expected = mat.A.T.dot(d)
        np.testing.assert_almost_equal(res, expected)

    many_random_tests(check)


def test_matvec_many_types():
    def check(mat):
        d = np.random.random(mat.shape[1])
        res = mat.matvec(d)
        expected = mat.A.dot(d)
        np.testing.assert_almost_equal(res, expected)

    many_random_tests(check)


def test_init_from_1d():
    m1 = DenseMatrix(np.arange(10, dtype=float))
    m2 = DenseMatrix(np.ones(shape=(10, 2), dtype=float))

    res = SplitMatrix([m1, m2])
    assert res.shape == (10, 3)
