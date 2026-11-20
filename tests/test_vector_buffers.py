"""Buffer properties of the inputs to the matrix products.

Every matrix type must give the same result for a vector (or `d`) that is
read-only, in non-native byte order or non-contiguous as for a plain array, or
raise an error. It must never return a wrong result. The same holds for matrices
built from read-only arrays, e.g. from a pandas DataFrame with copy-on-write,
and for an `out` array that is not contiguous.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import tabmat as tm

KINDS = [
    "dense_c",
    "dense_f",
    "sparse",
    "categorical",
    "split",
    "std_dense",
    "std_sparse",
    "std_split",
]
N = 60


def _read_only(a):
    a = a.copy(order="K")  # keep the Fortran order of `dense_f`
    a.setflags(write=False)
    return a


def _big_endian_strided(v):
    # Change the byte order before slicing: `astype` of a strided array is contiguous.
    return np.repeat(v, 2).astype(v.dtype.newbyteorder(">"))[::2]


VECTOR_TRANSFORMS = {
    "read_only": _read_only,
    "big_endian": lambda v: v.astype(v.dtype.newbyteorder(">")),
    "strided": lambda v: np.repeat(v, 2)[::2],
    "big_endian_strided": _big_endian_strided,
}


def _build(kind, dtype, read_only_data=False):
    rng = np.random.default_rng(0)
    dense = rng.uniform(size=(N, 3)).astype(dtype)
    sp = sps.random(N, 4, density=0.3, random_state=1, format="csc", dtype=dtype)
    codes = rng.integers(0, 5, N)
    if kind == "dense_f":
        dense = np.asfortranarray(dense)
    if read_only_data:
        dense = _read_only(dense)
        sp = sps.csc_matrix(
            (_read_only(sp.data), _read_only(sp.indices), _read_only(sp.indptr)),
            shape=sp.shape,
        )
        codes = _read_only(codes)

    def split():
        return tm.SplitMatrix(
            [
                tm.DenseMatrix(dense),
                tm.SparseMatrix(sp),
                tm.CategoricalMatrix(codes, dtype=dtype),
            ]
        )

    def standardized(mat):
        shift = rng.uniform(size=mat.shape[1]).astype(dtype)
        return tm.StandardizedMatrix(mat, shift)

    return {
        "dense_c": lambda: tm.DenseMatrix(dense),
        "dense_f": lambda: tm.DenseMatrix(dense),
        "sparse": lambda: tm.SparseMatrix(sp),
        "categorical": lambda: tm.CategoricalMatrix(codes, dtype=dtype),
        "split": split,
        "std_dense": lambda: standardized(tm.DenseMatrix(dense)),
        "std_sparse": lambda: standardized(tm.SparseMatrix(sp)),
        "std_split": lambda: standardized(split()),
    }[kind]()


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# name: (length axis of the vector, call)
OPERATIONS = {
    "matvec": (1, lambda m, v, **kw: m.matvec(v, **kw)),
    "transpose_matvec": (0, lambda m, v, **kw: m.transpose_matvec(v, **kw)),
    "sandwich": (0, lambda m, v, **kw: m.sandwich(v, **kw)),
}
DTYPES = [np.float64, np.float32]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("op", OPERATIONS)
@pytest.mark.parametrize("transform", VECTOR_TRANSFORMS)
def test_vector_buffer_properties(kind, dtype, op, transform):
    mat = _build(kind, dtype)
    axis, call = OPERATIONS[op]
    vec = np.random.default_rng(1).uniform(size=mat.shape[axis]).astype(dtype)
    expected = _dense(call(mat, vec))
    result = _dense(call(mat, VECTOR_TRANSFORMS[transform](vec)))
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("op", OPERATIONS)
def test_read_only_matrix_data(kind, dtype, op):
    expected_mat = _build(kind, dtype)
    mat = _build(kind, dtype, read_only_data=True)
    axis, call = OPERATIONS[op]
    vec = np.random.default_rng(1).uniform(size=mat.shape[axis]).astype(dtype)
    np.testing.assert_allclose(
        _dense(call(mat, vec)), _dense(call(expected_mat, vec)), rtol=1e-5
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("op", ["matvec", "transpose_matvec"])
def test_strided_out_is_never_silently_wrong(kind, op):
    # `out` is written in place, so it cannot be copied. Raising is fine, a
    # wrong result is not.
    mat = _build(kind, np.float64)
    axis, call = OPERATIONS[op]
    vec = np.random.default_rng(1).uniform(size=mat.shape[axis])
    expected = _dense(call(mat, vec))
    out = np.zeros(2 * len(expected))[::2]
    try:
        call(mat, vec, out=out)
    except (ValueError, TypeError):
        # No part may have written to `out` before the error.
        assert not out.any()
        return
    np.testing.assert_allclose(out, expected, rtol=1e-10)


@pytest.mark.parametrize("kind", ["dense_c", "sparse", "categorical", "split"])
def test_standardize_big_endian_weights(kind):
    mat = _build(kind, np.float64)
    weights = np.random.default_rng(2).uniform(size=N)
    weights /= weights.sum()
    expected = mat.standardize(weights, True, True)
    result = mat.standardize(weights.astype(">f8"), True, True)
    np.testing.assert_allclose(result[1], expected[1])
    np.testing.assert_allclose(result[2], expected[2])
    np.testing.assert_allclose(result[0].toarray(), expected[0].toarray())


def _sparse_variants():
    sp = sps.random(N, 4, density=0.3, random_state=1, format="csc")

    def with_int64_indices():
        sp64 = sp.copy()
        sp64.indices = sp64.indices.astype(np.int64)
        sp64.indptr = sp64.indptr.astype(np.int64)
        return sp64

    def with_strided_components():
        # The same matrix, with `data` and `indices` as views with stride 2.
        data = np.zeros(2 * sp.nnz)
        data[::2] = sp.data
        indices = np.zeros(2 * sp.nnz, dtype=np.int32)
        indices[::2] = sp.indices
        return sps.csc_matrix(
            (data[::2], indices[::2], sp.indptr), shape=sp.shape, copy=False
        )

    return sp, {
        "int64_indices": with_int64_indices,
        "strided_components": with_strided_components,
    }


@pytest.mark.parametrize("variant", ["int64_indices", "strided_components"])
@pytest.mark.parametrize("split", [False, True], ids=["sparse", "split"])
@pytest.mark.parametrize("op", OPERATIONS)
def test_unusual_sparse_components(variant, split, op):
    sp, variants = _sparse_variants()
    dense = tm.DenseMatrix(np.random.default_rng(0).uniform(size=(N, 3)))

    def wrap(csc):
        sparse = tm.SparseMatrix(csc)
        return tm.SplitMatrix([dense, sparse]) if split else sparse

    mat, expected_mat = wrap(variants[variant]()), wrap(sp)
    axis, call = OPERATIONS[op]
    vec = np.random.default_rng(1).uniform(size=mat.shape[axis])
    np.testing.assert_allclose(
        _dense(call(mat, vec)), _dense(call(expected_mat, vec)), rtol=1e-10
    )


def test_read_only_unsorted_sparse_indices():
    data = np.array([1.0, 2.0, 3.0])
    indices = np.array([2, 0, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3], dtype=np.int32)
    for a in (data, indices, indptr):
        a.flags.writeable = False
    mat = tm.SparseMatrix(sps.csc_matrix((data, indices, indptr), shape=(3, 2)))
    np.testing.assert_allclose(mat.toarray(), [[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    np.testing.assert_allclose(mat.matvec(np.array([1.0, 10.0])), [2.0, 30.0, 1.0])
    np.testing.assert_allclose(mat.transpose_matvec(np.ones(3)), [3.0, 3.0])
