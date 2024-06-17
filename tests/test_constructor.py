import numpy as np
import pandas as pd
import polars as pl
import pytest

import tabmat as tm


def test_pandas_to_matrix():
    n_rows = 50
    dense_column = np.linspace(-10, 10, num=n_rows, dtype=np.float64)
    dense_column_with_lots_of_zeros = dense_column.copy()
    dense_column_with_lots_of_zeros[:44] = 0.0
    sparse_column = np.zeros(n_rows, dtype=np.float64)
    sparse_column[0] = 1.0
    cat_column_lowdim = np.tile(["a", "b"], n_rows // 2)
    cat_column_highdim = np.arange(n_rows)

    dense_ser = pd.Series(dense_column)
    lowdense_ser = pd.Series(dense_column_with_lots_of_zeros)
    sparse_ser = pd.Series(sparse_column, dtype=pd.SparseDtype("float", 0.0))
    cat_ser_lowdim = pd.Categorical(cat_column_lowdim)
    cat_ser_highdim = pd.Categorical(cat_column_highdim)

    df = pd.DataFrame(
        data={
            "d": dense_ser,
            "ds": lowdense_ser,
            "s": sparse_ser,
            "cl_obj": cat_ser_lowdim.astype(object),
            "ch": cat_ser_highdim,
        }
    )

    mat = tm.from_pandas(
        df, dtype=np.float64, sparse_threshold=0.3, cat_threshold=4, object_as_cat=True
    )

    assert mat.shape == (n_rows, n_rows + 5)
    assert len(mat.matrices) == 3
    assert isinstance(mat, tm.SplitMatrix)

    nb_col_by_type = {
        tm.DenseMatrix: 3,  # includes low-dimension categorical
        tm.SparseMatrix: 2,  # sparse column
        tm.CategoricalMatrix: n_rows,
    }
    for submat in mat.matrices:
        assert submat.shape[1] == nb_col_by_type[type(submat)]

    # Prevent a regression where the column type of sparsified dense columns
    # was being changed in place.
    assert df["cl_obj"].dtype == object
    assert df["ds"].dtype == np.float64


@pytest.mark.parametrize("categorical_dtype", [pl.Categorical, pl.Enum(["a", "b"])])
def test_polars_to_matrix(categorical_dtype):
    n_rows = 50
    dense_column = np.linspace(-10, 10, num=n_rows, dtype=np.float64)
    dense_column_with_lots_of_zeros = dense_column.copy()
    dense_column_with_lots_of_zeros[:44] = 0.0
    sparse_column = np.zeros(n_rows, dtype=np.float64)
    sparse_column[0] = 1.0
    cat_column_lowdim = np.tile(["a", "b"], n_rows // 2)
    cat_column_highdim = np.arange(n_rows).astype("str")

    dense_ser = pl.Series(dense_column)
    lowdense_ser = pl.Series(dense_column_with_lots_of_zeros)
    sparse_ser = pl.Series(sparse_column)
    cat_ser_lowdim = pl.Series(cat_column_lowdim, dtype=categorical_dtype)
    cat_ser_highdim = pl.Series(cat_column_highdim, dtype=pl.Categorical)

    df = pl.DataFrame(
        data={
            "d": dense_ser,
            "ds": lowdense_ser,
            "s": sparse_ser,
            "cl": cat_ser_lowdim,
            "ch": cat_ser_highdim,
        }
    )

    mat = tm.from_polars(df, dtype=np.float64, sparse_threshold=0.3, cat_threshold=4)

    assert mat.shape == (n_rows, n_rows + 5)
    assert len(mat.matrices) == 3
    assert isinstance(mat, tm.SplitMatrix)

    nb_col_by_type = {
        tm.DenseMatrix: 3,  # includes low-dimension categorical
        tm.SparseMatrix: 2,  # sparse column
        tm.CategoricalMatrix: n_rows,
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
            tm.from_pandas(df, cat_missing_method=cat_missing_method)
    elif cat_missing_method == "zero":
        assert tm.from_pandas(df, cat_missing_method=cat_missing_method).shape == (6, 2)
    elif cat_missing_method == "convert":
        assert tm.from_pandas(df, cat_missing_method=cat_missing_method).shape == (6, 3)


@pytest.mark.parametrize("cat_missing_method", ["fail", "zero", "convert"])
def test_from_polars_missing(cat_missing_method):
    df = pl.DataFrame(
        {"cat": pl.Series(["1", "2", None, "1", "2", None], dtype=pl.Categorical)}
    )

    if cat_missing_method == "fail":
        with pytest.raises(
            ValueError, match="Categorical data can't have missing values"
        ):
            tm.from_polars(df, cat_missing_method=cat_missing_method)
    elif cat_missing_method == "zero":
        assert tm.from_polars(df, cat_missing_method=cat_missing_method).shape == (6, 2)
    elif cat_missing_method == "convert":
        assert tm.from_polars(df, cat_missing_method=cat_missing_method).shape == (6, 3)


@pytest.mark.parametrize("prefix_sep", ["_", ": "])
@pytest.mark.parametrize("drop_first", [True, False])
def test_names_pandas(prefix_sep, drop_first):
    n_rows = 50
    dense_column = np.linspace(-10, 10, num=n_rows, dtype=np.float64)
    dense_column_with_lots_of_zeros = dense_column.copy()
    dense_column_with_lots_of_zeros[:44] = 0.0
    sparse_column = np.zeros(n_rows, dtype=np.float64)
    sparse_column[0] = 1.0
    cat_column_lowdim = np.tile(["a", "b"], n_rows // 2)
    cat_column_highdim = np.arange(n_rows)

    dense_ser = pd.Series(dense_column)
    lowdense_ser = pd.Series(dense_column_with_lots_of_zeros)
    sparse_ser = pd.Series(sparse_column, dtype=pd.SparseDtype("float", 0.0))
    cat_ser_lowdim = pd.Categorical(cat_column_lowdim)
    cat_ser_highdim = pd.Categorical(cat_column_highdim)

    df = pd.DataFrame(
        data={
            "d": dense_ser,
            "cl_obj": cat_ser_lowdim.astype(object),
            "ch": cat_ser_highdim,
            "ds": lowdense_ser,
            "s": sparse_ser,
        }
    )

    categorical_format = "{name}" + prefix_sep + "{category}"
    mat_end = tm.from_pandas(
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

    mat_expand = tm.from_pandas(
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
    n_rows = 50
    dense_column = np.linspace(-10, 10, num=n_rows, dtype=np.float64)
    dense_column_with_lots_of_zeros = dense_column.copy()
    dense_column_with_lots_of_zeros[:44] = 0.0
    sparse_column = np.zeros(n_rows, dtype=np.float64)
    sparse_column[0] = 1.0
    cat_column_lowdim = np.tile(["a", "b"], n_rows // 2)
    cat_column_highdim = np.arange(n_rows).astype("str")

    dense_ser = pl.Series(dense_column)
    lowdense_ser = pl.Series(dense_column_with_lots_of_zeros)
    sparse_ser = pl.Series(sparse_column)
    cat_ser_lowdim = pl.Series(cat_column_lowdim, dtype=pl.Categorical)
    cat_ser_highdim = pl.Series(cat_column_highdim, dtype=pl.Categorical)

    df = pl.DataFrame(
        data={
            "d": dense_ser,
            "ds": lowdense_ser,
            "s": sparse_ser,
            "cl": cat_ser_lowdim,
            "ch": cat_ser_highdim,
        }
    )

    categorical_format = "{name}" + prefix_sep + "{category}"
    mat_end = tm.from_polars(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        cat_position="end",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    expanded_df = pd.get_dummies(
        df.to_pandas(), prefix_sep=prefix_sep, drop_first=drop_first
    )
    assert mat_end.column_names == list(expanded_df.columns)

    mat_expand = tm.from_polars(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        cat_position="expand",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    unique_terms = list(dict.fromkeys(mat_expand.term_names))
    assert unique_terms == list(df.columns)
