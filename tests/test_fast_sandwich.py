import numpy as np
import pytest
import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix

from tabmat import DenseMatrix, SparseMatrix, SplitMatrix
from tabmat.ext.dense import dense_sandwich
from tabmat.ext.sparse import sparse_sandwich


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fast_sandwich_sparse(dtype):
    np.random.seed(123)
    for _ in range(10):
        nrows, ncols = np.random.randint(200, size=2)

        A = simulate_matrix(shape=(nrows, ncols), seed=None, dtype=dtype).tocsc()

        d = np.random.rand(A.shape[0]).astype(dtype)
        true = (A.T.multiply(d)).dot(A).toarray()

        out = sparse_sandwich(
            A,
            A.tocsr(),
            d,
            np.arange(A.shape[0], dtype=np.int32),
            np.arange(A.shape[1], dtype=np.int32),
        )
        np.testing.assert_allclose(true, out, atol=np.sqrt(np.finfo(dtype).eps))


@pytest.mark.high_memory
def test_fast_sandwich_sparse_large():
    # note that 50000 * 50000 > 2^31 - 1, so this will segfault when we index
    # with 32 bit integers (see GH #160)
    A = simulate_matrix(
        nonzero_frac=1e-8, shape=(50000, 50000), seed=None, dtype=np.float32
    ).tocsc()
    d = np.random.rand(A.shape[0]).astype(np.float32)

    sparse_sandwich(
        A,
        A.tocsr(),
        d,
        np.arange(A.shape[0], dtype=np.int32),
        np.arange(A.shape[1], dtype=np.int32),
    )


def test_fast_sandwich_dense():
    for _ in range(5):
        A = simulate_matrix(shape=np.random.randint(1000, size=2))
        d = np.random.rand(A.shape[0])

        d[np.random.choice(np.arange(A.shape[0]), size=10, replace=False)] = 0.0

        check(A, d, np.arange(A.shape[1], dtype=np.int32))

        cols = np.random.choice(
            np.arange(A.shape[1]), size=np.random.randint(A.shape[1]), replace=False
        ).astype(np.int32)
        check(A, d, cols)


def test_dense_sandwich_on_non_contiguous():
    """Non-regression test for #208"""
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(100, 20))

    # Xd wraps a not-contiguous array.
    Xd = DenseMatrix(X[:, :10])
    Xs = SparseMatrix(csc_matrix(X[:, 10:]))
    Xm = SplitMatrix([Xd, Xs])

    # Making the sandwich product fail.
    with pytest.raises(Exception, match="The matrix X is not contiguous"):
        Xm.sandwich(np.ones(X.shape[0]))

    # Xd wraps a copy, which makes the data contiguous.
    Xd = DenseMatrix(X[:, :10].copy())
    Xm = SplitMatrix([Xd, Xs])

    # The sandwich product works without problem here.
    Xm.sandwich(np.ones(X.shape[0]))


def check(A, d, cols):
    Asub = A[:, cols]
    true = (Asub.T.multiply(d)).dot(Asub).toarray()
    nonzero = np.where(np.abs(d) > 1e-14)[0].astype(np.int32)
    out = dense_sandwich(np.asfortranarray(A.toarray()), d, nonzero, cols)
    np.testing.assert_allclose(true, out, atol=np.sqrt(np.finfo(np.float64).eps))


def simulate_matrix(nonzero_frac=0.05, shape=(100, 50), seed=0, dtype=np.float64):
    if seed is not None:
        np.random.seed(seed)
    nnz = int(np.prod(shape) * nonzero_frac)
    row_index = np.random.randint(shape[0], size=nnz)
    col_index = np.random.randint(shape[1], size=nnz)
    A = sp.sparse.csr_matrix(
        (np.random.randn(nnz).astype(dtype), (row_index, col_index)), shape
    )
    return A


@pytest.mark.high_memory
@pytest.mark.parametrize("order", ["C", "F"])
def test_fast_sandwich_dense_large(order):
    # this will segfault when we index with 32 bit integers (see GH #270)
    ii32 = np.iinfo(np.int32)
    K = 1000
    N = ii32.max // (K - 1) + 1  # to make sure we overflow
    rng = np.random.default_rng(seed=12345)
    A = DenseMatrix(
        np.ndarray(buffer=rng.random(N * K), dtype=float, shape=[N, K], order=order)
    )
    d = np.random.rand(A.shape[0])
    A.sandwich(d)
