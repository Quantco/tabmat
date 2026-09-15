import numpy as np
import pandas as pd
import polars as pl
import pytest

import tabmat as tm

N_ROWS = 50


def construct_data(backend):
    dense_column = np.linspace(-10, 10, num=N_ROWS, dtype=np.float64)
    dense_column_with_lots_of_zeros = dense_column.copy()
    dense_column_with_lots_of_zeros[:44] = 0.0
    sparse_column = np.zeros(N_ROWS, dtype=np.float64)
    sparse_column[0] = 1.0
    cat_column_lowdim = np.tile(["a", "b"], N_ROWS // 2)
    cat_column_highdim = np.arange(N_ROWS)

    data = {
        "d": dense_column,
        "ds": dense_column_with_lots_of_zeros,
        "s": sparse_column,
        "cl": cat_column_lowdim,
        "ch": cat_column_highdim,
    }

    if backend == "pandas":
        data["s"] = pd.Series(data["s"], dtype=pd.SparseDtype("float", 0.0))
        data["cl"] = cat_column_lowdim.astype("object")
        data["ch"] = pd.Categorical(cat_column_highdim)

        return pd.DataFrame(data)

    if backend == "polars":
        data["cl"] = pl.Series(cat_column_lowdim, dtype=pl.Categorical)
        data["ch"] = pl.Series(cat_column_highdim.astype("str"), dtype=pl.Categorical)

        return pl.DataFrame(data)

    raise ValueError


def test_pandas_to_matrix():
    df = construct_data("pandas")
    original_dtypes = df.dtypes.copy()

    mat = tm.from_df(
        df, dtype=np.float64, sparse_threshold=0.3, cat_threshold=4, object_as_cat=True
    )

    assert mat.shape == (N_ROWS, N_ROWS + 5)
    assert len(mat.matrices) == 3
    assert isinstance(mat, tm.SplitMatrix)

    nb_col_by_type = {
        tm.DenseMatrix: 3,  # includes low-dimension categorical
        tm.SparseMatrix: 2,  # sparse column
        tm.CategoricalMatrix: N_ROWS,
    }

    for submat in mat.matrices:
        assert submat.shape[1] == nb_col_by_type[type(submat)]

    # Prevent a regression where the column type of sparsified dense columns
    # was being changed in place.
    assert df["cl"].dtype == original_dtypes["cl"]
    assert df["ds"].dtype == original_dtypes["ds"]


@pytest.mark.parametrize("categorical_dtype", [pl.Categorical, pl.Enum(["a", "b"])])
def test_polars_to_matrix(categorical_dtype):
    df = construct_data("polars").with_columns(cl=pl.col("cl").cast(categorical_dtype))
    mat = tm.from_df(df, dtype=np.float64, sparse_threshold=0.3, cat_threshold=4)

    assert mat.shape == (N_ROWS, N_ROWS + 5)
    assert len(mat.matrices) == 3
    assert isinstance(mat, tm.SplitMatrix)

    nb_col_by_type = {
        tm.DenseMatrix: 3,  # includes low-dimension categorical
        tm.SparseMatrix: 2,  # sparse column
        tm.CategoricalMatrix: N_ROWS,
    }

    for submat in mat.matrices:
        assert submat.shape[1] == nb_col_by_type[type(submat)]

    # Prevent a regression where the column type of sparsified dense columns
    # was being changed in place.
    assert df["cl"].dtype == categorical_dtype
    assert df["ds"].dtype == pl.Float64


@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_from_pandas_missing(cat_missing_method):
    df = pd.DataFrame({"cat": pd.Categorical([1, 2, pd.NA, 1, 2, pd.NA])})

    if cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            tm.from_df(df, cat_missing_method=cat_missing_method)
    elif cat_missing_method == "zero":
        assert tm.from_df(df, cat_missing_method=cat_missing_method).shape == (6, 2)
    elif cat_missing_method == "convert":
        assert tm.from_df(df, cat_missing_method=cat_missing_method).shape == (6, 3)


@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_from_polars_missing(cat_missing_method):
    df = pl.DataFrame(
        {"cat": pl.Series(["1", "2", None, "1", "2", None], dtype=pl.Categorical)}
    )

    if cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            tm.from_df(df, cat_missing_method=cat_missing_method)
    elif cat_missing_method == "zero":
        assert tm.from_df(df, cat_missing_method=cat_missing_method).shape == (6, 2)
    elif cat_missing_method == "convert":
        assert tm.from_df(df, cat_missing_method=cat_missing_method).shape == (6, 3)


@pytest.mark.parametrize("prefix_sep", ["_", ": "])
@pytest.mark.parametrize("drop_first", [True, False])
def test_names_pandas(prefix_sep, drop_first):
    df = construct_data("pandas")
    categorical_format = "{name}" + prefix_sep + "{category}"

    mat_end = tm.from_df(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        object_as_cat=True,
        cat_position="end",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    expanded_df = pd.get_dummies(df, prefix_sep=prefix_sep, drop_first=drop_first)
    assert mat_end.column_names == expanded_df.columns.tolist()

    mat_expand = tm.from_df(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        object_as_cat=True,
        cat_position="expand",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    unique_terms = list(dict.fromkeys(mat_expand.term_names))
    assert unique_terms == df.columns.tolist()


@pytest.mark.parametrize("prefix_sep", ["_", ": "])
@pytest.mark.parametrize("drop_first", [True, False])
def test_names_polars(prefix_sep, drop_first):
    df = construct_data("polars")
    categorical_format = "{name}" + prefix_sep + "{category}"

    mat_end = tm.from_df(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        cat_position="end",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    # workround for https://github.com/pola-rs/polars/issues/24273#issuecomment-3255212324
    _df = df.to_pandas()
    categoricals = _df.select_dtypes("category").columns
    expanded_df = pd.get_dummies(
        _df.astype({x: "object" for x in categoricals}),
        prefix_sep=prefix_sep,
        drop_first=drop_first,
    )
    assert mat_end.column_names == list(expanded_df.columns)

    mat_expand = tm.from_df(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        cat_position="expand",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    unique_terms = list(dict.fromkeys(mat_expand.term_names))
    assert unique_terms == df.columns


def _members(mat):
    return mat.matrices if isinstance(mat, tm.SplitMatrix) else [mat]


def _sparse_series(values, fill_value=0.0):
    return pd.Series(
        pd.arrays.SparseArray(np.asarray(values, dtype=float), fill_value=fill_value)
    )


def test_from_pandas_sparse_columns_match_dense():
    rng = np.random.default_rng(0)
    n = 200

    def sparse_col(density):
        v = np.zeros(n)
        idx = rng.choice(n, int(n * density), replace=False)
        v[idx] = rng.standard_normal(idx.size)
        return v

    s0, s1, d = sparse_col(0.05), sparse_col(0.1), rng.standard_normal(n)
    df = pd.DataFrame({"s0": _sparse_series(s0), "s1": _sparse_series(s1), "d": d})
    mat = tm.from_pandas(df, dtype=np.float64)
    np.testing.assert_array_equal(mat.toarray(), np.column_stack([s0, s1, d]))
    sparse_members = [m for m in _members(mat) if isinstance(m, tm.SparseMatrix)]
    assert len(sparse_members) == 1
    assert sparse_members[0].shape[1] == 2


def test_from_pandas_sparse_explicit_zeros_are_dropped():
    # The product of two sparse arrays keeps the union of their indices, so it
    # stores zeros explicitly. Those must not end up as stored entries.
    a = pd.arrays.SparseArray([1.0, 0.0] * 10, fill_value=0.0)
    b = pd.arrays.SparseArray([0.0, 1.0] * 10, fill_value=0.0)
    prod = a * b
    assert prod.sp_values.size > 0 and (prod.sp_values == 0).all()
    mat = tm.from_pandas(pd.DataFrame({"s": pd.Series(prod)}))
    assert isinstance(mat, tm.SparseMatrix)
    assert mat.tocsc().nnz == 0
    np.testing.assert_array_equal(mat.toarray(), np.zeros((20, 1)))


@pytest.mark.parametrize("fill_value", [np.nan, 1.0])
def test_from_pandas_sparse_nonzero_fill_is_stored_dense(fill_value):
    values = np.full(30, fill_value)
    values[[3, 7]] = 2.0
    df = pd.DataFrame(
        {
            "s": pd.Series(pd.arrays.SparseArray(values, fill_value=fill_value)),
            "d": np.arange(30, dtype=float),
        }
    )
    with pytest.warns(UserWarning, match="fill_value"):
        mat = tm.from_pandas(df, dtype=np.float64)
    assert all(isinstance(m, tm.DenseMatrix) for m in _members(mat))
    expected = np.column_stack([values, np.arange(30, dtype=float)])
    np.testing.assert_array_equal(mat.toarray(), expected)


def test_from_pandas_sparse_all_fill_column():
    df = pd.DataFrame({"s": _sparse_series(np.zeros(25)), "d": np.ones(25)})
    mat = tm.from_pandas(df, dtype=np.float64)
    np.testing.assert_array_equal(
        mat.toarray(), np.column_stack([np.zeros(25), np.ones(25)])
    )
    sparse_members = [m for m in _members(mat) if isinstance(m, tm.SparseMatrix)]
    assert len(sparse_members) == 1
    assert sparse_members[0].tocsc().nnz == 0


def test_from_pandas_sparse_block_mixes_sources():
    # pandas sparse columns and numeric or boolean columns below the sparse
    # threshold share one SparseMatrix and must agree with the dense reference.
    n = 100
    s = np.zeros(n)
    s[[1, 50]] = 3.0
    ds = np.zeros(n)
    ds[[2, 60]] = -1.5
    b = np.zeros(n, dtype=bool)
    b[[4, 70]] = True
    d = np.linspace(0, 1, n)
    df = pd.DataFrame({"s": _sparse_series(s), "ds": ds, "b": b, "d": d})
    mat = tm.from_pandas(df, dtype=np.float64, sparse_threshold=0.1)
    np.testing.assert_array_equal(
        mat.toarray(), np.column_stack([s, ds, b.astype(float), d])
    )
    sparse_members = [m for m in _members(mat) if isinstance(m, tm.SparseMatrix)]
    assert len(sparse_members) == 1
    assert sparse_members[0].shape[1] == 3
