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


@pytest.mark.skip(reason="too heavy")
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


# (3000, 40) takes the k-parallel path, (700, 600) the other one with several
# column blocks, and (7, 3) only the remainder loops.
@pytest.mark.parametrize("shape", [(3000, 40), (700, 600), (7, 3)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("restrict", [False, True], ids=["all", "restricted"])
def test_dense_sandwich_center(shape, order, dtype, restrict):
    rng = np.random.default_rng(0)
    n, m = shape
    X = np.array(
        rng.standard_normal(shape) + rng.uniform(-50, 50, m), dtype=dtype, order=order
    )
    d = rng.uniform(0.5, 1.5, n).astype(dtype)
    center = X.astype(np.float64).mean(axis=0)
    rows = np.arange(n, dtype=np.int32)
    cols = np.arange(m, dtype=np.int32)
    if restrict:
        rows = np.sort(rng.choice(n, max(1, n // 2), replace=False)).astype(np.int32)
        cols = rng.permutation(m)[: max(1, m // 3)].astype(np.int32)

    out = dense_sandwich(X, d, rows, cols, center=center)

    Xc = X.astype(np.float64)[np.ix_(rows, cols)] - center[cols]
    expected = Xc.T @ (d.astype(np.float64)[rows, None] * Xc)
    rtol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(
        out, expected, rtol=rtol, atol=rtol * np.abs(expected).max()
    )


def test_dense_sandwich_zero_center_matches_uncentered():
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.standard_normal((700, 50)))
    d = rng.uniform(0.5, 1.5, 700)
    rows = np.arange(700, dtype=np.int32)
    cols = np.arange(50, dtype=np.int32)
    np.testing.assert_array_equal(
        dense_sandwich(X, d, rows, cols, center=np.zeros(50)),
        dense_sandwich(X, d, rows, cols),
    )


def test_dense_sandwich_center_wrong_length():
    X = np.ones((10, 3))
    with pytest.raises(ValueError, match="center must have length 3"):
        dense_sandwich(
            X,
            np.ones(10),
            np.arange(10, dtype=np.int32),
            np.arange(3, dtype=np.int32),
            center=np.zeros(2),
        )


def test_dense_sandwich_on_non_contiguous():
    """Non-regression test for #208

    DenseMatrix now automatically ensures arrays are contiguous,
    so non-contiguous inputs are automatically copied and made contiguous.
    """
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(100, 20))

    # Column slicing creates a non-contiguous array, but DenseMatrix
    # automatically makes it contiguous (copying only if necessary).
    non_contiguous_array = X[:, :10]
    assert not non_contiguous_array.flags["C_CONTIGUOUS"]
    assert not non_contiguous_array.flags["F_CONTIGUOUS"]

    with pytest.warns(UserWarning, match="not contiguous"):
        Xd = DenseMatrix(non_contiguous_array)
    # The internal array should now be contiguous
    assert Xd.A.flags["C_CONTIGUOUS"] or Xd.A.flags["F_CONTIGUOUS"]

    Xs = SparseMatrix(csc_matrix(X[:, 10:]))
    Xm = SplitMatrix([Xd, Xs])

    # The sandwich product should work without problem
    result = Xm.sandwich(np.ones(X.shape[0]))
    assert result is not None


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


@pytest.mark.skip(reason="too heavy")
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
