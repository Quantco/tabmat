import re

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pyarrow
import pytest

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
    with pl.StringCache():
        _ = pl.Series(["beagle", "poodle", "labrador"], dtype=pl.Categorical)
        cat_series = pl.Series(str_series, dtype=pl.Categorical)

    indices, categories = _extract_codes_and_categories(cat_series)
    np.testing.assert_array_equal(str_series, categories[indices].tolist())
