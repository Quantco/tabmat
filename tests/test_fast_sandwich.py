import numpy as np
import pytest
import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix

from tabmat import CategoricalMatrix, DenseMatrix, SparseMatrix, SplitMatrix
from tabmat.crabmat.dense import dense_matvec, dense_rmatvec, dense_sandwich
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


def _assert_close_to(actual, expected, dtype):
    rtol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * np.abs(expected).max()
    )


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
    _assert_close_to(out, Xc.T @ (d.astype(np.float64)[rows, None] * Xc), dtype)


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


def _centered_kernel_data(order, dtype, n=1200, m=9):
    rng = np.random.default_rng(1)
    B = np.array(
        rng.standard_normal((n, m)) + rng.uniform(-50, 50, m), dtype=dtype, order=order
    )
    d = rng.uniform(0.5, 1.5, n).astype(dtype)
    # In the data's dtype, like the means from standardize. The kernels cast the
    # center to that dtype, so a float64 center would not be the one they use.
    center = B.mean(axis=0, dtype=np.float64).astype(dtype).astype(np.float64)
    rows = np.sort(rng.choice(n, n // 2, replace=False)).astype(np.int32)
    return rng, B, d, center, rows


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sparse_dense_sandwich_center(order, dtype):
    rng, B, d, center, rows = _centered_kernel_data(order, dtype)
    A = SparseMatrix(
        sp.sparse.random(
            B.shape[0], 7, density=0.3, format="csc", random_state=2
        ).astype(dtype)
    )
    A_cols = np.array([0, 2, 3, 6], dtype=np.int32)
    B_cols = np.array([1, 4, 5, 8], dtype=np.int32)

    out = A.sandwich_dense(B, d, rows, A_cols, B_cols, B_center=center)

    A_dense = A.toarray().astype(np.float64)[np.ix_(rows, A_cols)]
    Bc = B.astype(np.float64)[np.ix_(rows, B_cols)] - center[B_cols]
    _assert_close_to(out, A_dense.T @ (d.astype(np.float64)[rows, None] * Bc), dtype)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "kind", ["plain", "drop_first", "missing"], ids=["fast", "drop_first", "missing"]
)
def test_categorical_dense_sandwich_center(order, dtype, kind):
    rng, B, d, center, rows = _centered_kernel_data(order, dtype)
    levels = rng.choice(list("abcd"), B.shape[0]).astype(object)
    if kind == "missing":
        levels[rng.random(B.shape[0]) < 0.1] = None
    cat = CategoricalMatrix(
        levels,
        drop_first=kind == "drop_first",
        cat_missing_method="zero" if kind == "missing" else "fail",
        dtype=dtype,
    )
    B_cols = np.array([0, 3, 4, 7], dtype=np.int32)

    out = cat._cross_dense(B, d, rows, None, B_cols, dense_center=center)

    A_dense = cat.toarray().astype(np.float64)[rows]
    Bc = B.astype(np.float64)[np.ix_(rows, B_cols)] - center[B_cols]
    _assert_close_to(out, A_dense.T @ (d.astype(np.float64)[rows, None] * Bc), dtype)


# (5000, 30) runs in parallel and the others serially; widths 9 and 3 also
# exercise the scalar remainders of the vectorized loops. None selects all rows
# or columns, which the kernels vectorize along.
@pytest.mark.parametrize("shape", [(5000, 30), (1200, 9), (7, 3)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("restrict", ["none", "rows", "cols", "both"])
@pytest.mark.parametrize("transpose", [False, True], ids=["matvec", "rmatvec"])
def test_dense_matvec_center(shape, order, dtype, restrict, transpose):
    rng = np.random.default_rng(1)
    n, m = shape
    X = np.array(
        rng.standard_normal(shape) + rng.uniform(-50, 50, m), dtype=dtype, order=order
    )
    center = X.mean(axis=0, dtype=np.float64).astype(dtype).astype(np.float64)
    rows = cols = None
    if restrict in ("rows", "both"):
        rows = np.sort(rng.choice(n, max(1, n // 2), replace=False)).astype(np.int32)
    if restrict in ("cols", "both"):
        cols = rng.permutation(m)[: max(1, m // 2)].astype(np.int32)
    v = rng.standard_normal(n if transpose else m).astype(dtype)

    out = (dense_rmatvec if transpose else dense_matvec)(X, v, rows, cols, center)

    r = np.arange(n) if rows is None else rows
    c = np.arange(m) if cols is None else cols
    Xc = X.astype(np.float64)[np.ix_(r, c)] - center[c]
    v64 = v.astype(np.float64)
    _assert_close_to(out, Xc.T @ v64[r] if transpose else Xc @ v64[c], dtype)


def test_centered_kernels_zero_center_match_uncentered():
    rng, B, d, _, rows = _centered_kernel_data("F", np.float64)
    zeros = np.zeros(B.shape[1])
    all_cols = np.arange(B.shape[1], dtype=np.int32)
    A = SparseMatrix(sp.sparse.random(B.shape[0], 4, density=0.3, format="csc"))
    cat = CategoricalMatrix(rng.choice(list("abc"), B.shape[0]))
    # These kernels add up per-thread results atomically, in varying order, so
    # even two identical calls can differ in the last bit.
    np.testing.assert_allclose(
        A.sandwich_dense(B, d, rows, None, None, B_center=zeros),
        A.sandwich_dense(B, d, rows, None, None),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        cat._cross_dense(B, d, rows, None, None, dense_center=zeros),
        cat._cross_dense(B, d, rows, None, None),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        dense_rmatvec(B, d, rows, all_cols, center=zeros),
        dense_rmatvec(B, d, rows, all_cols),
        rtol=1e-13,
    )
    np.testing.assert_array_equal(
        dense_matvec(B, d[: B.shape[1]], center=zeros),
        dense_matvec(B, d[: B.shape[1]]),
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
