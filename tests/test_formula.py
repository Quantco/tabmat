import pickle
import re
from io import BytesIO

import formulaic
import numpy as np
import pandas as pd
import pytest
from formulaic.materializers import FormulaMaterializer
from formulaic.materializers.types import EvaluatedFactor, FactorValues
from formulaic.parser.types import Factor
from scipy import sparse as sps

import tabmat as tm
from tabmat.formula import (
    TabmatMaterializer,
    _interact,
    _InteractableCategoricalVector,
    _InteractableDenseVector,
    _InteractableSparseVector,
)


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "num_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat_1": pd.Categorical(["a", "b", "c", "b", "a"]),
            "cat_2": pd.Categorical(["x", "y", "z", "x", "y"]),
            "cat_3": pd.Categorical(["1", "2", "1", "2", "1"]),
            "str_1": ["a", "b", "c", "b", "a"],
        }
    )
    return df


def test_retrieval():
    assert FormulaMaterializer.for_materializer("tabmat") is TabmatMaterializer
    assert (
        FormulaMaterializer.for_data(pd.DataFrame(), output="tabmat")
        is TabmatMaterializer
    )


@pytest.mark.parametrize(
    "formula, expected",
    [
        pytest.param(
            "1 + num_1",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(
                        np.array(
                            [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
                        ).T
                    )
                ]
            ),
            id="numeric",
        ),
        pytest.param(
            "1 + cat_1",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T),
                    tm.CategoricalMatrix(
                        pd.Categorical(
                            [
                                "__drop__",
                                "cat_1[b]",
                                "cat_1[c]",
                                "cat_1[b]",
                                "__drop__",
                            ],
                            categories=["__drop__", "cat_1[b]", "cat_1[c]"],
                        ),
                        drop_first=True,
                    ),
                ]
            ),
            id="categorical",
        ),
        pytest.param(
            "{np.where(num_1 >= 2, num_1, 0)} * {np.where(num_2 <= 2, num_2, 0)}",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(np.array([[0.0, 2.0, 3.0, 4.0, 5.0]]).T),
                    tm.SparseMatrix(
                        sps.csc_matrix(
                            np.array(
                                [
                                    [1.0, 2.0, 0.0, 0.0, 0.0],
                                    [0.0, 2.0, 0.0, 0.0, 0.0],
                                ]
                            )
                        ).T
                    ),
                ]
            ),
            id="numeric_sparse",
        ),
        pytest.param(
            "1 + num_1 : cat_1",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T),
                    tm.SparseMatrix(
                        sps.csc_matrix(
                            np.array(
                                [
                                    [1.0, 0.0, 0.0, 0.0, 5.0],
                                    [0.0, 2.0, 0.0, 4.0, 0.0],
                                    [0.0, 0.0, 3.0, 0.0, 0.0],
                                ]
                            ).T
                        )
                    ),
                ]
            ),
            id="interaction_cat_num",
        ),
        pytest.param(
            "cat_1 : cat_3 - 1",
            tm.SplitMatrix(
                [
                    tm.CategoricalMatrix(
                        pd.Categorical(
                            [
                                "cat_1[a]:cat_3[1]",
                                "cat_1[b]:cat_3[2]",
                                "cat_1[c]:cat_3[1]",
                                "cat_1[b]:cat_3[2]",
                                "cat_1[a]:cat_3[1]",
                            ],
                            categories=[
                                "cat_1[a]:cat_3[1]",
                                "cat_1[b]:cat_3[1]",
                                "cat_1[c]:cat_3[1]",
                                "cat_1[a]:cat_3[2]",
                                "cat_1[c]:cat_3[2]",
                                "cat_1[b]:cat_3[2]",
                            ],
                        ),
                        drop_first=False,
                    ),
                ]
            ),
            id="interaction_cat_cat",
        ),
    ],
)
def test_matrix_against_expectation(df, formula, expected):
    model_df = tm.from_formula(
        formula, df, ensure_full_rank=True, cat_threshold=1, sparse_threshold=0.5
    )
    assert len(model_df.matrices) == len(expected.matrices)
    for res, exp in zip(model_df.matrices, expected.matrices):
        assert type(res) == type(exp)
        if isinstance(res, (tm.DenseMatrix, tm.SparseMatrix)):
            np.testing.assert_array_equal(res.A, res.A)
        elif isinstance(res, tm.CategoricalMatrix):
            assert (exp.cat == res.cat).all()
            assert exp.drop_first == res.drop_first


@pytest.mark.parametrize(
    "formula, expected",
    [
        pytest.param(
            "1 + num_1",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(
                        np.array(
                            [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
                        ).T
                    )
                ]
            ),
            id="numeric",
        ),
        pytest.param(
            "1 + cat_1",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T),
                    tm.CategoricalMatrix(
                        pd.Categorical(
                            [
                                "__drop__",
                                "cat_1__b",
                                "cat_1__c",
                                "cat_1__b",
                                "__drop__",
                            ],
                            categories=["__drop__", "cat_1__b", "cat_1__c"],
                        ),
                        drop_first=True,
                    ),
                ]
            ),
            id="categorical",
        ),
        pytest.param(
            "1 + num_1 : cat_1",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T),
                    tm.SparseMatrix(
                        sps.csc_matrix(
                            np.array(
                                [
                                    [1.0, 0.0, 0.0, 0.0, 5.0],
                                    [0.0, 2.0, 0.0, 4.0, 0.0],
                                    [0.0, 0.0, 3.0, 0.0, 0.0],
                                ]
                            ).T
                        )
                    ),
                ]
            ),
            id="interaction_cat_num",
        ),
        pytest.param(
            "cat_1 : cat_3 - 1",
            tm.SplitMatrix(
                [
                    tm.CategoricalMatrix(
                        pd.Categorical(
                            [
                                "cat_1__a__x__cat_3__1",
                                "cat_1__b__x__cat_3__2",
                                "cat_1__c__x__cat_3__1",
                                "cat_1__b__x__cat_3__2",
                                "cat_1__a__x__cat_3__1",
                            ],
                            categories=[
                                "cat_1__a__x__cat_3__1",
                                "cat_1__b__x__cat_3__1",
                                "cat_1__c__x__cat_3__1",
                                "cat_1__a__x__cat_3__2",
                                "cat_1__c__x__cat_3__2",
                                "cat_1__b__x__cat_3__2",
                            ],
                        ),
                        drop_first=False,
                    ),
                ]
            ),
            id="interaction_cat_cat",
        ),
    ],
)
def test_matrix_against_expectation_qcl(df, formula, expected):
    model_df = tm.from_formula(
        formula,
        df,
        cat_threshold=1,
        sparse_threshold=0.5,
        ensure_full_rank=True,
        interaction_separator="__x__",
        categorical_format="{name}__{category}",
        intercept_name="intercept",
    )
    assert len(model_df.matrices) == len(expected.matrices)
    for res, exp in zip(model_df.matrices, expected.matrices):
        assert type(res) == type(exp)
        if isinstance(res, (tm.DenseMatrix, tm.SparseMatrix)):
            np.testing.assert_array_equal(res.A, res.A)
        elif isinstance(res, tm.CategoricalMatrix):
            assert (exp.cat == res.cat).all()
            assert exp.drop_first == res.drop_first


@pytest.mark.parametrize(
    "ensure_full_rank", [True, False], ids=["full_rank", "all_levels"]
)
@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("num_1 + num_2", id="numeric"),
        pytest.param("cat_1 + cat_2", id="categorical"),
        pytest.param("cat_1 * cat_2 * cat_3", id="interaction"),
        pytest.param("num_1 + cat_1 * num_2 * cat_2", id="mixed"),
        pytest.param("{np.log(num_1)} + {num_in_scope * num_2}", id="functions"),
        pytest.param("{num_1 * num_in_scope}", id="variable_in_scope"),
        pytest.param("bs(num_1, 3)", id="spline"),
        pytest.param(
            "poly(num_1, 3, raw=True) + poly(num_2, 3, raw=False)", id="polynomial"
        ),
        pytest.param(
            "C(num_1)",
            id="convert_to_categorical",
        ),
        pytest.param(
            "C(cat_1, spans_intercept=False) * cat_2 * cat_3",
            id="custom_contrasts",
        ),
        pytest.param("str_1", id="string_as_categorical"),
    ],
)
def test_matrix_against_pandas(df, formula, ensure_full_rank):
    num_in_scope = 2  # noqa
    model_df = formulaic.model_matrix(formula, df, ensure_full_rank=ensure_full_rank)
    model_tabmat = tm.from_formula(
        formula, df, ensure_full_rank=ensure_full_rank, include_intercept=True
    )
    np.testing.assert_array_equal(model_df.to_numpy(), model_tabmat.A)


@pytest.mark.parametrize(
    "formula, expected_names",
    [
        pytest.param(
            "1 + num_1 + num_2", ("Intercept", "num_1", "num_2"), id="numeric"
        ),
        pytest.param("num_1 + num_2 - 1", ("num_1", "num_2"), id="no_intercept"),
        pytest.param(
            "1 + cat_1", ("Intercept", "cat_1[b]", "cat_1[c]"), id="categorical"
        ),
        pytest.param(
            "1 + cat_2 * cat_3",
            (
                "Intercept",
                "cat_2[y]",
                "cat_2[z]",
                "cat_3[2]",
                "cat_2[y]:cat_3[2]",
                "cat_2[z]:cat_3[2]",
            ),
            id="interaction",
        ),
        pytest.param(
            "poly(num_1, 3) - 1",
            ("poly(num_1, 3)[1]", "poly(num_1, 3)[2]", "poly(num_1, 3)[3]"),
            id="polynomial",
        ),
        pytest.param(
            "1 + {np.log(num_1 ** 2)}",
            ("Intercept", "np.log(num_1 ** 2)"),
            id="functions",
        ),
    ],
)
def test_names_against_expectation(df, formula, expected_names):
    model_tabmat = tm.from_formula(formula, df, ensure_full_rank=True)
    assert model_tabmat.model_spec.column_names == expected_names
    assert model_tabmat.column_names == list(expected_names)


@pytest.mark.parametrize(
    "formula, expected_names",
    [
        pytest.param(
            "1 + cat_1", ("intercept", "cat_1__b", "cat_1__c"), id="categorical"
        ),
        pytest.param(
            "1 + cat_2 * cat_3",
            (
                "intercept",
                "cat_2__y",
                "cat_2__z",
                "cat_3__2",
                "cat_2__y__x__cat_3__2",
                "cat_2__z__x__cat_3__2",
            ),
            id="interaction",
        ),
        pytest.param(
            "poly(num_1, 3) - 1",
            ("poly(num_1, 3)[1]", "poly(num_1, 3)[2]", "poly(num_1, 3)[3]"),
            id="polynomial",
        ),
        pytest.param(
            "1 + {np.log(num_1 ** 2)}",
            ("intercept", "np.log(num_1 ** 2)"),
            id="functions",
        ),
    ],
)
def test_names_against_expectation_qcl(df, formula, expected_names):
    model_tabmat = tm.from_formula(
        formula,
        df,
        ensure_full_rank=True,
        categorical_format="{name}__{category}",
        interaction_separator="__x__",
        intercept_name="intercept",
    )
    assert model_tabmat.model_spec.column_names == expected_names
    assert model_tabmat.column_names == list(expected_names)


@pytest.mark.parametrize(
    "formula, expected_names",
    [
        pytest.param("1 + cat_1", ("1", "cat_1", "cat_1"), id="categorical"),
        pytest.param(
            "1 + cat_2 * cat_3",
            (
                "1",
                "cat_2",
                "cat_2",
                "cat_3",
                "cat_2:cat_3",
                "cat_2:cat_3",
            ),
            id="interaction",
        ),
        pytest.param(
            "poly(num_1, 3) - 1",
            ("poly(num_1, 3)", "poly(num_1, 3)", "poly(num_1, 3)"),
            id="polynomial",
        ),
        pytest.param(
            "1 + {np.log(num_1 ** 2)}",
            ("1", "np.log(num_1 ** 2)"),
            id="functions",
        ),
    ],
)
def test_term_names_against_expectation(df, formula, expected_names):
    model_tabmat = tm.from_formula(
        formula,
        df,
        ensure_full_rank=True,
        intercept_name="intercept",
    )
    assert model_tabmat.term_names == list(expected_names)


@pytest.mark.parametrize(
    "categorical_format",
    ["{name}[{category}]", "{name}__{category}"],
    ids=["brackets", "double_underscore"],
)
def test_all_names_against_from_pandas(df, categorical_format):
    mat_from_pandas = tm.from_pandas(
        df, drop_first=False, object_as_cat=True, categorical_format=categorical_format
    )
    mat_from_formula = tm.from_formula(
        "num_1 + num_2 + cat_1 + cat_2 + cat_3 + str_1 - 1",
        data=df,
        ensure_full_rank=False,
        categorical_format=categorical_format,
    )

    assert mat_from_formula.column_names == mat_from_pandas.column_names
    assert mat_from_formula.term_names == mat_from_pandas.term_names


@pytest.mark.parametrize(
    "ensure_full_rank", [True, False], ids=["full_rank", "all_levels"]
)
@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("1 + num_1 + num_2", id="numeric"),
        pytest.param("1 + cat_1 + cat_2", id="categorical"),
        pytest.param("1 + cat_1 * cat_2 * cat_3", id="interaction"),
        pytest.param("1 + num_1 + cat_1 * num_2 * cat_2", id="mixed"),
        pytest.param("1 + {np.log(num_1)} + {num_in_scope * num_2}", id="functions"),
        pytest.param("1 + {num_1 * num_in_scope}", id="variable_in_scope"),
        pytest.param("1 + bs(num_1, 3)", id="spline"),
        pytest.param(
            "1 + poly(num_1, 3, raw=True) + poly(num_2, 3, raw=False)", id="polynomial"
        ),
        pytest.param(
            "1 + C(num_1)",
            id="convert_to_categorical",
        ),
        pytest.param(
            "1 + C(cat_1, spans_intercept=False) * cat_2 * cat_3",
            id="custom_contrasts",
        ),
    ],
)
def test_names_against_pandas(df, formula, ensure_full_rank):
    num_in_scope = 2  # noqa
    model_df = formulaic.model_matrix(formula, df, ensure_full_rank=ensure_full_rank)
    model_tabmat = tm.from_formula(
        formula,
        df,
        ensure_full_rank=ensure_full_rank,
        categorical_format="{name}[T.{category}]",
    )
    assert model_tabmat.model_spec.column_names == model_df.model_spec.column_names
    assert model_tabmat.model_spec.column_names == tuple(model_df.columns)
    assert model_tabmat.column_names == list(model_df.columns)


@pytest.mark.parametrize(
    "ensure_full_rank", [True, False], ids=["full_rank", "all_levels"]
)
@pytest.mark.parametrize(
    "formula, formula_with_intercept, formula_wo_intercept",
    [
        ("num_1", "1 + num_1", "num_1 - 1"),
        ("cat_1", "1 + cat_1", "cat_1 - 1"),
        (
            "num_1 * cat_1 * cat_2",
            "1 + num_1 * cat_1 * cat_2",
            "num_1 * cat_1 * cat_2 - 1",
        ),
    ],
    ids=["numeric", "categorical", "mixed"],
)
def test_include_intercept(
    df, formula, formula_with_intercept, formula_wo_intercept, ensure_full_rank
):
    model_no_include = tm.from_formula(
        formula, df, include_intercept=False, ensure_full_rank=ensure_full_rank
    )
    model_no_intercept = tm.from_formula(
        formula_wo_intercept,
        df,
        include_intercept=True,
        ensure_full_rank=ensure_full_rank,
    )
    np.testing.assert_array_equal(model_no_include.A, model_no_intercept.A)
    assert (
        model_no_include.model_spec.column_names
        == model_no_intercept.model_spec.column_names
    )

    model_include = tm.from_formula(
        formula, df, include_intercept=True, ensure_full_rank=ensure_full_rank
    )
    model_intercept = tm.from_formula(
        formula_with_intercept,
        df,
        include_intercept=False,
        ensure_full_rank=ensure_full_rank,
    )
    np.testing.assert_array_equal(model_include.A, model_intercept.A)
    assert (
        model_no_include.model_spec.column_names
        == model_no_intercept.model_spec.column_names
    )


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("str_1 : cat_1", id="implicit"),
        pytest.param("C(str_1) : C(cat_1, spans_intercept=False)", id="explicit"),
    ],
)
@pytest.mark.parametrize(
    "ensure_full_rank", [True, False], ids=["full_rank", "all_levels"]
)
def test_C_state(df, formula, ensure_full_rank):
    model_tabmat = tm.from_formula(
        "str_1 : cat_1 + 1", df, cat_threshold=0, ensure_full_rank=ensure_full_rank
    )
    model_tabmat_2 = model_tabmat.model_spec.get_model_matrix(df[:2])
    np.testing.assert_array_equal(model_tabmat.A[:2, :], model_tabmat_2.A)
    np.testing.assert_array_equal(
        model_tabmat.matrices[1].cat.categories,
        model_tabmat_2.matrices[1].cat.categories,
    )


VECTORS = [
    _InteractableDenseVector(np.array([1, 2, 3, 4, 5], dtype=np.float64)).set_name(
        "dense"
    ),
    _InteractableSparseVector(
        sps.csc_matrix(np.array([[1, 0, 0, 0, 2]], dtype=np.float64).T)
    ).set_name("sparse"),
    _InteractableCategoricalVector.from_categorical(
        pd.Categorical(["a", "b", "c", "b", "a"]), reduced_rank=True
    ).set_name("cat_reduced"),
    _InteractableCategoricalVector.from_categorical(
        pd.Categorical(["a", "b", "c", "b", "a"]), reduced_rank=False
    ).set_name("cat_full"),
]


@pytest.mark.parametrize(
    "left", VECTORS, ids=["dense", "sparse", "cat_full", "cat_reduced"]
)
@pytest.mark.parametrize(
    "right", VECTORS, ids=["dense", "sparse", "cat_full", "cat_reduced"]
)
@pytest.mark.parametrize("reverse", [False, True], ids=["not_reversed", "reversed"])
def test_interactable_vectors(left, right, reverse):
    n = left.to_tabmat().shape[0]
    left_np = left.to_tabmat().A.reshape((n, -1))
    right_np = right.to_tabmat().A.reshape((n, -1))

    if reverse:
        left_np, right_np = right_np, left_np

    if isinstance(left, _InteractableCategoricalVector) and isinstance(
        right, _InteractableCategoricalVector
    ):
        result_np = np.zeros((n, left_np.shape[1] * right_np.shape[1]))
        for i in range(right_np.shape[1]):
            for j in range(left_np.shape[1]):
                result_np[:, i * left_np.shape[1] + j] = left_np[:, j] * right_np[:, i]
    else:
        result_np = left_np * right_np

    result_vec = _interact(left, right, reverse=reverse)

    # Test types
    if isinstance(left, _InteractableCategoricalVector) or isinstance(
        right, _InteractableCategoricalVector
    ):
        assert isinstance(result_vec, _InteractableCategoricalVector)
    elif isinstance(left, _InteractableSparseVector) or isinstance(
        right, _InteractableSparseVector
    ):
        assert isinstance(result_vec, _InteractableSparseVector)
    else:
        assert isinstance(result_vec, _InteractableDenseVector)

    # Test values
    np.testing.assert_array_equal(
        result_vec.to_tabmat().A.squeeze(), result_np.squeeze()
    )

    # Test names
    if not reverse:
        assert result_vec.name == left.name + ":" + right.name
    else:
        assert result_vec.name == right.name + ":" + left.name


@pytest.mark.parametrize("cat_missing_method", ["zero", "convert"])
@pytest.mark.parametrize(
    "cat_missing_name",
    ["__missing__", "(MISSING)"],
)
def test_cat_missing_handling(cat_missing_method, cat_missing_name):
    df = pd.DataFrame(
        {
            "cat_1": pd.Categorical(["a", "b", None, "b", "a"]),
        }
    )

    mat_from_pandas = tm.from_pandas(
        df,
        cat_threshold=0,
        cat_missing_method=cat_missing_method,
        cat_missing_name=cat_missing_name,
    )

    mat_from_formula = tm.from_formula(
        "cat_1 - 1",
        df,
        cat_threshold=0,
        cat_missing_method=cat_missing_method,
        cat_missing_name=cat_missing_name,
    )

    assert mat_from_pandas.column_names == mat_from_formula.column_names
    assert mat_from_pandas.term_names == mat_from_formula.term_names
    np.testing.assert_array_equal(mat_from_pandas.A, mat_from_formula.A)

    mat_from_formula_new = mat_from_formula.model_spec.get_model_matrix(df)
    assert mat_from_pandas.column_names == mat_from_formula_new.column_names
    assert mat_from_pandas.term_names == mat_from_formula_new.term_names
    np.testing.assert_array_equal(mat_from_pandas.A, mat_from_formula_new.A)


def test_cat_missing_C():
    df = pd.DataFrame(
        {
            "cat_1": pd.Categorical(["a", "b", None, "b", "a"]),
            "cat_2": pd.Categorical(["1", "2", None, "1", "2"]),
        }
    )
    formula = (
        "C(cat_1, missing_method='convert', missing_name='M') "
        "+ C(cat_2, missing_method='zero')"
    )
    expected_names = [
        "C(cat_1, missing_method='convert', missing_name='M')[a]",
        "C(cat_1, missing_method='convert', missing_name='M')[b]",
        "C(cat_1, missing_method='convert', missing_name='M')[M]",
        "C(cat_2, missing_method='zero')[1]",
        "C(cat_2, missing_method='zero')[2]",
    ]

    result = tm.from_formula(formula, df)

    assert result.column_names == expected_names
    assert result.model_spec.get_model_matrix(df).column_names == expected_names
    np.testing.assert_equal(result.model_spec.get_model_matrix(df).A, result.A)
    np.testing.assert_equal(
        result.model_spec.get_model_matrix(df[:2]).A, result.A[:2, :]
    )


@pytest.mark.parametrize(
    "cat_missing_method", ["zero", "convert"], ids=["zero", "convert"]
)
def test_cat_missing_unseen(cat_missing_method):
    df = pd.DataFrame(
        {
            "cat_1": pd.Categorical(["a", "b", None, "b", "a"]),
        }
    )
    df_unseen = pd.DataFrame(
        {
            "cat_1": pd.Categorical(["a", None]),
        }
    )
    result_seen = tm.from_formula(
        "cat_1 - 1", df, cat_missing_method=cat_missing_method
    )
    result_unseen = result_seen.model_spec.get_model_matrix(df_unseen)

    assert result_seen.column_names == result_unseen.column_names
    if cat_missing_method == "convert":
        expected_array = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float64)
    elif cat_missing_method == "zero":
        expected_array = np.array([[1, 0], [0, 0]], dtype=np.float64)

    np.testing.assert_array_equal(result_unseen.A, expected_array)


def test_cat_missing_interactions():
    df = pd.DataFrame(
        {
            "cat_1": pd.Categorical(["a", "b", None, "b", "a"]),
            "cat_2": pd.Categorical(["1", "2", None, "1", "2"]),
        }
    )
    formula = "C(cat_1, missing_method='convert') : C(cat_2, missing_method='zero') - 1"
    expected_names = [
        "C(cat_1, missing_method='convert')[a]:C(cat_2, missing_method='zero')[1]",
        "C(cat_1, missing_method='convert')[b]:C(cat_2, missing_method='zero')[1]",
        "C(cat_1, missing_method='convert')[(MISSING)]:C(cat_2, missing_method='zero')[1]",
        "C(cat_1, missing_method='convert')[a]:C(cat_2, missing_method='zero')[2]",
        "C(cat_1, missing_method='convert')[b]:C(cat_2, missing_method='zero')[2]",
        "C(cat_1, missing_method='convert')[(MISSING)]:C(cat_2, missing_method='zero')[2]",
    ]

    assert tm.from_formula(formula, df).column_names == expected_names


# Tests from formulaic's test suite
# ---------------------------------

FORMULAIC_TESTS = {
    # '<formula>': (<full_rank_names>, <names>, <full_rank_null_names>, <null_rows>)
    "a": (["Intercept", "a"], ["Intercept", "a"], ["Intercept", "a"], 2),
    "A": (
        ["Intercept", "A[b]", "A[c]"],
        ["Intercept", "A[a]", "A[b]", "A[c]"],
        ["Intercept", "A[c]"],
        2,
    ),
    "C(A)": (
        ["Intercept", "C(A)[b]", "C(A)[c]"],
        ["Intercept", "C(A)[a]", "C(A)[b]", "C(A)[c]"],
        ["Intercept", "C(A)[c]"],
        2,
    ),
    "A:a": (
        ["Intercept", "A[a]:a", "A[b]:a", "A[c]:a"],
        ["Intercept", "A[a]:a", "A[b]:a", "A[c]:a"],
        ["Intercept", "A[a]:a"],
        1,
    ),
    "A:B": (
        [
            "Intercept",
            "B[b]",
            "B[c]",
            "A[b]:B[a]",
            "A[c]:B[a]",
            "A[b]:B[b]",
            "A[c]:B[b]",
            "A[b]:B[c]",
            "A[c]:B[c]",
        ],
        [
            "Intercept",
            "A[a]:B[a]",
            "A[b]:B[a]",
            "A[c]:B[a]",
            "A[a]:B[b]",
            "A[b]:B[b]",
            "A[c]:B[b]",
            "A[a]:B[c]",
            "A[b]:B[c]",
            "A[c]:B[c]",
        ],
        ["Intercept"],
        1,
    ),
}


class TestFormulaicTests:
    @pytest.fixture
    def data(self):
        return pd.DataFrame(
            {"a": [1, 2, 3], "b": [1, 2, 3], "A": ["a", "b", "c"], "B": ["a", "b", "c"]}
        )

    @pytest.fixture
    def data_with_nulls(self):
        return pd.DataFrame(
            {"a": [1, 2, None], "A": ["a", None, "c"], "B": ["a", "b", None]}
        )

    @pytest.fixture
    def materializer(self, data):
        return TabmatMaterializer(data)

    @pytest.mark.parametrize("formula,tests", FORMULAIC_TESTS.items())
    def test_get_model_matrix(self, materializer, formula, tests):
        mm = materializer.get_model_matrix(formula, ensure_full_rank=True)
        assert isinstance(mm, tm.MatrixBase)
        assert mm.shape == (3, len(tests[0]))
        assert list(mm.model_spec.column_names) == tests[0]

        mm = materializer.get_model_matrix(formula, ensure_full_rank=False)
        assert isinstance(mm, tm.MatrixBase)
        assert mm.shape == (3, len(tests[1]))
        assert list(mm.model_spec.column_names) == tests[1]

    def test_get_model_matrix_edge_cases(self, materializer):
        mm = materializer.get_model_matrix(("a",), ensure_full_rank=True)
        assert isinstance(mm, formulaic.ModelMatrices)
        assert isinstance(mm[0], tm.MatrixBase)

        mm = materializer.get_model_matrix("a ~ A", ensure_full_rank=True)
        assert isinstance(mm, formulaic.ModelMatrices)
        assert "lhs" in mm.model_spec
        assert "rhs" in mm.model_spec

        mm = materializer.get_model_matrix(("a ~ A",), ensure_full_rank=True)
        assert isinstance(mm, formulaic.ModelMatrices)
        assert isinstance(mm[0], formulaic.ModelMatrices)

    def test_get_model_matrix_invalid_output(self, materializer):
        with pytest.raises(
            formulaic.errors.FormulaMaterializationError,
            match=r"Nominated output .* is invalid\. Available output types are: ",
        ):
            materializer.get_model_matrix(
                "a", ensure_full_rank=True, output="invalid_output"
            )

    @pytest.mark.parametrize("formula,tests", FORMULAIC_TESTS.items())
    def test_na_handling(self, data_with_nulls, formula, tests):
        mm = TabmatMaterializer(data_with_nulls).get_model_matrix(formula)
        assert isinstance(mm, tm.MatrixBase)
        assert mm.shape == (tests[3], len(tests[2]))
        assert list(mm.model_spec.column_names) == tests[2]

        if formula != "a":
            pytest.skip("Tabmat does not allo NAs in categoricals")

        mm = TabmatMaterializer(data_with_nulls).get_model_matrix(
            formula, na_action="ignore"
        )
        assert isinstance(mm, tm.MatrixBase)
        assert mm.shape == (3, len(tests[0]) + (-1 if "A" in formula else 0))

    def test_state(self, materializer):
        mm = materializer.get_model_matrix("center(a) - 1")
        assert isinstance(mm, tm.MatrixBase)
        assert list(mm.model_spec.column_names) == ["center(a)"]
        assert np.allclose(mm.getcol(0).unpack().squeeze(), [-1, 0, 1])

        mm2 = TabmatMaterializer(pd.DataFrame({"a": [4, 5, 6]})).get_model_matrix(
            mm.model_spec
        )
        assert isinstance(mm2, tm.MatrixBase)
        assert list(mm2.model_spec.column_names) == ["center(a)"]
        assert np.allclose(mm2.getcol(0).unpack().squeeze(), [2, 3, 4])

        mm3 = mm.model_spec.get_model_matrix(pd.DataFrame({"a": [4, 5, 6]}))
        assert isinstance(mm3, tm.MatrixBase)
        assert list(mm3.model_spec.column_names) == ["center(a)"]
        assert np.allclose(mm3.getcol(0).unpack().squeeze(), [2, 3, 4])

    def test_factor_evaluation_edge_cases(self, materializer):
        # Test that categorical kinds are set if type would otherwise be numerical
        ev_factor = materializer._evaluate_factor(
            Factor("a", eval_method="lookup", kind="categorical"),
            formulaic.model_spec.ModelSpec(formula=[]),
            drop_rows=set(),
        )
        assert ev_factor.metadata.kind.value == "categorical"

        # Test that other kind mismatches result in an exception
        materializer.factor_cache = {}
        with pytest.raises(
            formulaic.errors.FactorEncodingError,
            match=re.escape(
                "Factor `A` is expecting values of kind 'numerical', "
                "but they are actually of kind 'categorical'."
            ),
        ):
            materializer._evaluate_factor(
                Factor("A", eval_method="lookup", kind="numerical"),
                formulaic.model_spec.ModelSpec(formula=[]),
                drop_rows=set(),
            )

        # Test that if an encoding has already been determined, that an exception is raised
        # if the new encoding does not match
        materializer.factor_cache = {}
        with pytest.raises(
            formulaic.errors.FactorEncodingError,
            match=re.escape(
                "The model specification expects factor `a` to have values of kind "
                "`categorical`, but they are actually of kind `numerical`."
            ),
        ):
            materializer._evaluate_factor(
                Factor("a", eval_method="lookup", kind="numerical"),
                formulaic.model_spec.ModelSpec(
                    formula=[], encoder_state={"a": ("categorical", {})}
                ),
                drop_rows=set(),
            )

    def test__is_categorical(self, materializer):
        assert materializer._is_categorical([1, 2, 3]) is False
        assert materializer._is_categorical(pd.Series(["a", "b", "c"])) is True
        assert materializer._is_categorical(pd.Categorical(["a", "b", "c"])) is True
        assert materializer._is_categorical(FactorValues({}, kind="categorical"))

    def test_encoding_edge_cases(self, materializer):
        # Verify that constant encoding works well
        encoded_factor = materializer._encode_evaled_factor(
            factor=EvaluatedFactor(
                factor=Factor("10", eval_method="literal", kind="constant"),
                values=FactorValues(10, kind="constant"),
            ),
            spec=formulaic.model_spec.ModelSpec(formula=[]),
            drop_rows=[],
        )
        np.testing.assert_array_equal(encoded_factor["10"].values, [10, 10, 10])

        # Verify that unencoded dictionaries with drop-fields work
        encoded_factor = materializer._encode_evaled_factor(
            factor=EvaluatedFactor(
                factor=Factor("a", eval_method="lookup", kind="numerical"),
                values=FactorValues(
                    {"a": pd.Series([1, 2, 3]), "b": pd.Series([4, 5, 6])},
                    kind="numerical",
                    spans_intercept=True,
                    drop_field="a",
                ),
            ),
            spec=formulaic.model_spec.ModelSpec(formula=[]),
            drop_rows=set(),
        )
        np.testing.assert_array_equal(encoded_factor["a[a]"].values, [1, 2, 3])
        np.testing.assert_array_equal(encoded_factor["a[b]"].values, [4, 5, 6])

        encoded_factor = materializer._encode_evaled_factor(
            factor=EvaluatedFactor(
                factor=Factor("a", eval_method="lookup", kind="numerical"),
                values=FactorValues(
                    {"a": pd.Series([1, 2, 3]), "b": pd.Series([4, 5, 6])},
                    kind="numerical",
                    spans_intercept=True,
                    drop_field="a",
                ),
            ),
            spec=formulaic.model_spec.ModelSpec(formula=[]),
            drop_rows=set(),
            reduced_rank=True,
        )
        np.testing.assert_array_equal(encoded_factor["a[b]"].values, [4, 5, 6])

        # Verify that encoding of nested dictionaries works well
        encoded_factor = materializer._encode_evaled_factor(
            factor=EvaluatedFactor(
                factor=Factor("A", eval_method="python", kind="numerical"),
                values=FactorValues(
                    {
                        "a": pd.Series([1, 2, 3]),
                        "b": pd.Series([4, 5, 6]),
                        "__metadata__": None,
                    },
                    kind="numerical",
                ),
            ),
            spec=formulaic.model_spec.ModelSpec(formula=[]),
            drop_rows=[],
        )
        np.testing.assert_array_equal(encoded_factor["A[a]"].values, [1, 2, 3])

        encoded_factor = materializer._encode_evaled_factor(
            factor=EvaluatedFactor(
                factor=Factor("B", eval_method="python", kind="categorical"),
                values=FactorValues(
                    {"a": pd.Series(["a", "b", "c"])}, kind="categorical"
                ),
            ),
            spec=formulaic.model_spec.ModelSpec(formula=[]),
            drop_rows=[],
        )
        encoded_matrix = (
            encoded_factor["B[a]"].set_name("B[a]").to_tabmat(cat_threshold=1)
        )
        assert list(encoded_matrix.cat) == ["B[a][a]", "B[a][b]", "B[a][c]"]

    def test_empty(self, materializer):
        mm = materializer.get_model_matrix("0", ensure_full_rank=True)
        assert mm.shape[1] == 0
        mm = materializer.get_model_matrix("0", ensure_full_rank=False)
        assert mm.shape[1] == 0

    def test_category_reordering(self):
        data = pd.DataFrame({"A": ["a", "b", "c"]})
        data2 = pd.DataFrame({"A": ["c", "b", "a"]})
        data3 = pd.DataFrame(
            {"A": pd.Categorical(["c", "b", "a"], categories=["c", "b", "a"])}
        )

        m = TabmatMaterializer(data).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m.model_spec.column_names) == ["A[a]", "A[b]", "A[c]"]

        m2 = TabmatMaterializer(data2).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m2.model_spec.column_names) == ["A[a]", "A[b]", "A[c]"]

        m3 = TabmatMaterializer(data3).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m3.model_spec.column_names) == ["A[c]", "A[b]", "A[a]"]

    def test_term_clustering(self, materializer):
        assert materializer.get_model_matrix(
            "a + b + a:A + b:A"
        ).model_spec.column_names == (
            "Intercept",
            "a",
            "b",
            "a:A[b]",
            "a:A[c]",
            "b:A[b]",
            "b:A[c]",
        )
        assert materializer.get_model_matrix(
            "a + b + a:A + b:A", cluster_by="numerical_factors"
        ).model_spec.column_names == (
            "Intercept",
            "a",
            "a:A[b]",
            "a:A[c]",
            "b",
            "b:A[b]",
            "b:A[c]",
        )

    def test_model_spec_pickleable(self, materializer):
        o = BytesIO()
        ms = materializer.get_model_matrix("a ~ a:A")
        pickle.dump(ms.model_spec, o)
        o.seek(0)
        ms2 = pickle.load(o)
        assert isinstance(ms, formulaic.parser.types.Structured)
        assert ms2.lhs.formula.root == ["a"]
