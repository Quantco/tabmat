import numpy as np
import pytest
from scipy import sparse as sps

from tabmat import SparseMatrix


@pytest.fixture
def x() -> sps.csc_array:
    np.random.seed(0)
    return sps.random_array((10, 3), density=0.1).tocsc()


def test_mkl_sparse_init(x: sps.csc_array):
    one = SparseMatrix(x)
    two = SparseMatrix((x.data, x.indices, x.indptr), shape=x.shape)
    three = SparseMatrix(x.toarray())
    np.testing.assert_allclose(one.toarray(), two.toarray())
    np.testing.assert_allclose(one.toarray(), three.toarray())


def test_to_csc(x: sps.csc_array):
    one = SparseMatrix(x).tocsc()
    two = SparseMatrix((x.data, x.indices, x.indptr), shape=x.shape).tocsc()
    three = SparseMatrix(x.toarray()).tocsc()

    assert isinstance(one, sps.csc_array)
    assert isinstance(two, sps.csc_array)
    assert isinstance(three, sps.csc_array)

    def _test_same(mat1, mat2):
        assert mat1.nnz == mat2.nnz
        np.testing.assert_allclose(mat1.data, mat2.data)
        assert (mat1.indices == mat2.indices).all()
        assert (mat1.indptr == mat2.indptr).all()

    _test_same(one, two)
    _test_same(one, three)
