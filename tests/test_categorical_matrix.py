import re

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pyarrow
import pytest
import scipy.sparse as sps

from tabmat import SparseMatrix
from tabmat.categorical_matrix import CategoricalMatrix, _extract_codes_and_categories


@pytest.fixture
def cat_vec(missing):
    m = 10
    seed = 0
    rng = np.random.default_rng(seed)
    vec = rng.choice([0, 1, 2, np.inf, -np.inf], size=m)
    if missing:
        vec[vec == 1] = np.nan
    return vec


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "no_missing"])
@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_recover_orig(cat_vec, drop_first, missing, cat_missing_method):
    if missing and cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            CategoricalMatrix(
                cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
            )
        return

    orig_recovered = CategoricalMatrix(
        cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
    ).recover_orig()
    np.testing.assert_equal(orig_recovered, cat_vec)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "no_missing"])
@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_csr_matvec_categorical(
    cat_vec, vec_dtype, drop_first, missing, cat_missing_method
):
    if missing and cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            CategoricalMatrix(
                cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
            )
        return

    mat = pd.get_dummies(
        cat_vec,
        drop_first=drop_first,
        dtype="uint8",
        dummy_na=(cat_missing_method == "convert" and missing),
    )
    cat_mat = CategoricalMatrix(
        cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
    )
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = cat_mat.matvec(vec)
    np.testing.assert_allclose(res, cat_mat.toarray().dot(vec))


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "no_missing"])
@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_tocsr(cat_vec, drop_first, missing, cat_missing_method):
    if missing and cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            CategoricalMatrix(
                cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
            )
        return

    cat_mat = CategoricalMatrix(
        cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
    )
    res = cat_mat.tocsr().toarray()
    expected = pd.get_dummies(
        cat_vec,
        drop_first=drop_first,
        dtype="uint8",
        dummy_na=cat_missing_method == "convert" and missing,
    )
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "no_missing"])
@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_transpose_matvec(cat_vec, drop_first, missing, cat_missing_method):
    if missing and cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            CategoricalMatrix(
                cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
            )
        return

    cat_mat = CategoricalMatrix(
        cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
    )
    other = np.random.random(cat_mat.shape[0])
    res = cat_mat.transpose_matvec(other)
    expected = pd.get_dummies(
        cat_vec,
        drop_first=drop_first,
        dtype="uint8",
        dummy_na=cat_missing_method == "convert" and missing,
    ).T.dot(other)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "no_missing"])
@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_multiply(cat_vec, drop_first, missing, cat_missing_method):
    if missing and cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            CategoricalMatrix(
                cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
            )
        return

    cat_mat = CategoricalMatrix(
        cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
    )
    other = np.arange(len(cat_vec))[:, None]
    actual = cat_mat.multiply(other)
    expected = (
        pd.get_dummies(
            cat_vec,
            drop_first=drop_first,
            dtype="uint8",
            dummy_na=cat_missing_method == "convert" and missing,
        )
        * other
    )
    np.testing.assert_allclose(actual.toarray(), expected)


@pytest.mark.parametrize("column_vector", [False, True])
@pytest.mark.parametrize("drop_first", [False, True])
def test_multiply_single_row(column_vector, drop_first):
    mat = CategoricalMatrix(
        pd.Categorical(["b"], categories=["a", "b"]), drop_first=drop_first
    )
    weights = np.array([[2.0]]) if column_vector else np.array([2.0])
    np.testing.assert_allclose(mat.multiply(weights).toarray(), mat.toarray() * 2)


@pytest.mark.parametrize("mi_element", [np.nan, None])
def test_nulls(mi_element):
    vec = [0, mi_element, 1]
    with pytest.raises(ValueError, match="Categorical data can't have missing values"):
        CategoricalMatrix(vec)


@pytest.mark.parametrize("cat_missing_name", ["(MISSING)", "__None__", "[NULL]"])
def test_cat_missing_name(cat_missing_name):
    vec = [None, "(MISSING)", "__None__", "a", "b"]
    if cat_missing_name in vec:
        with pytest.raises(
            ValueError,
            match=re.escape(f"Missing category {cat_missing_name} already exists."),
        ):
            CategoricalMatrix(
                vec, cat_missing_method="convert", cat_missing_name=cat_missing_name
            )
    else:
        cat = CategoricalMatrix(
            vec, cat_missing_method="convert", cat_missing_name=cat_missing_name
        )
        assert set(cat.categories) == set(vec) - {None} | {cat_missing_name}


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "no_missing"])
@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_categorical_indexing(drop_first, missing, cat_missing_method):
    if not missing:
        cat_vec = [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3]
    else:
        cat_vec = [0, None, 2, 0, None, 2, 0, None, 2, 3, 3]

    if missing and cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            CategoricalMatrix(
                cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
            )
        return

    mat = CategoricalMatrix(
        cat_vec, drop_first=drop_first, cat_missing_method=cat_missing_method
    )
    expected = pd.get_dummies(
        cat_vec,
        drop_first=drop_first,
        dummy_na=cat_missing_method == "convert" and missing,
    ).to_numpy()[:, [0, 1]]
    np.testing.assert_allclose(mat[:, [0, 1]].toarray(), expected)


@pytest.mark.parametrize(
    "input_type", ["pandas.Categorical", "pandas", "polars", "pyarrow", "list"]
)
@pytest.mark.parametrize("narwhals_input", [True, False])
def test_extract_codes_and_categories(input_type, narwhals_input):
    cat_vec = pd.Series(["a", "b", "c", pd.NA, "b", "a", "d"], dtype="category")
    if input_type == "pandas.Categorical":
        cat_vec = pd.Categorical(cat_vec)
    elif input_type == "polars":
        cat_vec = pl.Series(cat_vec)
    elif input_type == "pyarrow":
        cat_vec = pyarrow.chunked_array([cat_vec])
    elif input_type == "list":
        cat_vec = cat_vec.astype("object")

    if narwhals_input:
        if input_type in ["list", "pandas.Categorical"]:
            pytest.skip("Narwhals doesn't support list or pandas.Categorical inputs")
        cat_vec = nw.from_native(cat_vec, series_only=True)

    indices, categories = _extract_codes_and_categories(cat_vec)
    np.testing.assert_array_equal(indices, np.array([0, 1, 2, -1, 1, 0, 3]))
    np.testing.assert_array_equal(categories, np.array(["a", "b", "c", "d"]))


def test_polars_non_contiguous_codes():
    str_series = ["labrador", "boxer", "beagle"]
    # Share one set of categories so that the physical codes are non-contiguous.
    dtype = pl.Categorical(pl.Categories.random())
    _ = pl.Series(["beagle", "poodle", "labrador"], dtype=dtype)
    cat_series = pl.Series(str_series, dtype=dtype)

    indices, categories = _extract_codes_and_categories(cat_series)
    np.testing.assert_array_equal(str_series, categories[indices].tolist())


def test_shape_of_empty():
    cat_mat = CategoricalMatrix([], drop_first=True)
    assert cat_mat.shape == (0, 0)


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("cat_missing_method", [None, "zero", "convert"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("restrict", [False, True], ids=["all", "restricted"])
def test_cross_sandwich_sparse(
    drop_first, cat_missing_method, dtype, index_dtype, restrict
):
    rng = np.random.default_rng(0)
    n = 500
    vals = np.array([f"c{code}" for code in rng.integers(0, 7, n)], dtype=object)
    if cat_missing_method is not None:
        vals[rng.random(n) < 0.1] = None
    cat = CategoricalMatrix(
        pd.Categorical(vals),
        drop_first=drop_first,
        dtype=dtype,
        cat_missing_method=cat_missing_method or "fail",
    )
    values = rng.uniform(size=(n, 9)) * (rng.uniform(size=(n, 9)) < 0.3)
    sparse = sps.csc_matrix(values.astype(dtype))
    sparse.indices = sparse.indices.astype(index_dtype)
    sparse.indptr = sparse.indptr.astype(index_dtype)
    d = rng.uniform(size=n).astype(dtype)
    if restrict:
        rows = np.sort(rng.choice(n, n // 2, replace=False)).astype(np.int32)
        L_cols = np.array([0, 2, 3], dtype=np.int32)
        R_cols = np.array([1, 4, 5, 8], dtype=np.int32)
    else:
        rows = L_cols = R_cols = None

    res = cat._cross_sandwich(SparseMatrix(sparse), d, rows, L_cols, R_cols)
    res_t = SparseMatrix(sparse)._cross_sandwich(cat, d, rows, R_cols, L_cols)

    cat_dense, sparse_dense = cat.toarray(), sparse.toarray()
    r = slice(None) if rows is None else rows
    expected = cat_dense[r].T @ (d[r, None] * sparse_dense[r])
    if restrict:
        expected = expected[np.ix_(L_cols, R_cols)]
    tol = 1e-5 if dtype == np.float32 else 1e-12
    assert res.dtype == dtype
    np.testing.assert_allclose(res, expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(res_t, expected.T, rtol=tol, atol=tol)


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize(
    "L_cols",
    [[0], [5, 1, 3], [6, 0, 6, 2], [-1, 0], []],
    ids=["single", "unsorted", "duplicates", "negative", "empty"],
)
@pytest.mark.parametrize("restrict", [False, True], ids=["all", "restricted"])
def test_cross_sandwich_sparse_L_cols(drop_first, L_cols, restrict):
    rng = np.random.default_rng(0)
    n = 500
    cat = CategoricalMatrix(rng.integers(0, 8, n), drop_first=drop_first)
    sparse = SparseMatrix(sps.random(n, 9, density=0.3, format="csc", random_state=0))
    d = rng.uniform(size=n)
    L_cols = np.array(L_cols, dtype=np.int32)
    if restrict:
        rows = np.sort(rng.choice(n, n // 2, replace=False)).astype(np.int32)
        R_cols = np.array([1, 4, 5, 8], dtype=np.int32)
    else:
        rows = R_cols = None

    res = cat._cross_sandwich(sparse, d, rows, L_cols, R_cols)
    full = cat._cross_sandwich(sparse, d, rows, None, R_cols)
    np.testing.assert_allclose(res, full[L_cols], rtol=1e-12, atol=1e-12)


def _read_only(vec):
    vec = vec.copy()
    vec.setflags(write=False)
    return vec


@pytest.mark.parametrize("drop_first", [True, False], ids=["drop_first", "no_drop"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "transform",
    [
        _read_only,
        lambda v: v.astype(v.dtype.newbyteorder(">")),
        lambda v: np.repeat(v, 2)[::2],
    ],
    ids=["read_only", "big_endian", "strided"],
)
@pytest.mark.parametrize("restrict", [False, True], ids=["all", "restricted"])
def test_vector_buffer_properties(drop_first, dtype, transform, restrict):
    # Regression test for GH-565: vectors that are read-only, non-native byte
    # order or non-contiguous must give the same result as a plain array.
    rng = np.random.default_rng(0)
    n = 60
    cat = CategoricalMatrix(rng.integers(0, 5, n), drop_first=drop_first, dtype=dtype)
    rows = np.arange(0, n, 3, dtype=np.int32) if restrict else None
    cols = np.array([0, 2], dtype=np.int32) if restrict else None
    vec_k = rng.uniform(size=cat.shape[1]).astype(dtype)
    vec_n = rng.uniform(size=n).astype(dtype)

    np.testing.assert_allclose(
        cat.matvec(transform(vec_k), cols=cols), cat.matvec(vec_k, cols=cols)
    )
    np.testing.assert_allclose(
        cat.transpose_matvec(transform(vec_n), rows=rows, cols=cols),
        cat.transpose_matvec(vec_n, rows=rows, cols=cols),
    )
    np.testing.assert_allclose(
        cat.sandwich(transform(vec_n), rows=rows, cols=cols).toarray(),
        cat.sandwich(vec_n, rows=rows, cols=cols).toarray(),
    )
