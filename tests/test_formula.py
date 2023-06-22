import formulaic
import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sps

import tabmat as tm


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "num_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat_1": pd.Categorical(["a", "b", "c", "b", "a"]),
            "cat_2": pd.Categorical(["x", "y", "z", "x", "y"]),
            "cat_3": pd.Categorical(["1", "2", "1", "2", "1"]),
        }
    )
    return df


@pytest.mark.parametrize(
    "formula, expected",
    [
        pytest.param(
            "num_1",
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
            "cat_1",
            tm.SplitMatrix(
                [
                    tm.DenseMatrix(np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T),
                    tm.CategoricalMatrix(
                        pd.Categorical(
                            [
                                "__drop__",
                                "cat_1[T.b]",
                                "cat_1[T.c]",
                                "cat_1[T.b]",
                                "__drop__",
                            ],
                            categories=["__drop__", "cat_1[T.b]", "cat_1[T.c]"],
                        ),
                        drop_first=True,
                    ),
                ]
            ),
            id="categorical",
        ),
        pytest.param(
            "num_1 : cat_1",
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
                                "cat_1[T.a]:cat_3[T.1]",
                                "cat_1[T.b]:cat_3[T.2]",
                                "cat_1[T.c]:cat_3[T.1]",
                                "cat_1[T.b]:cat_3[T.2]",
                                "cat_1[T.a]:cat_3[T.1]",
                            ],
                            categories=[
                                "cat_1[T.a]:cat_3[T.1]",
                                "cat_1[T.b]:cat_3[T.1]",
                                "cat_1[T.c]:cat_3[T.1]",
                                "cat_1[T.a]:cat_3[T.2]",
                                "cat_1[T.c]:cat_3[T.2]",
                                "cat_1[T.b]:cat_3[T.2]",
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
    model_df = tm.from_formula(formula, df, ensure_full_rank=True)
    assert len(model_df.matrices) == len(expected.matrices)
    for res, exp in zip(model_df.matrices, expected.matrices):
        assert type(res) == type(exp)
        if isinstance(res, tm.DenseMatrix):
            np.testing.assert_array_equal(res, exp)
        elif isinstance(res, tm.SparseMatrix):
            np.testing.assert_array_equal(res.A, res.A)
        elif isinstance(res, tm.CategoricalMatrix):
            assert (exp.cat == res.cat).all()
            assert exp.drop_first == res.drop_first


@pytest.mark.parametrize(
    "formula, expected",
    [
        pytest.param(
            "num_1",
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
            "cat_1",
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
            "num_1 : cat_1",
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
        ensure_full_rank=True,
        interaction_separator="__x__",
        categorical_format="{name}__{category}",
        intercept_name="intercept",
    )
    assert len(model_df.matrices) == len(expected.matrices)
    for res, exp in zip(model_df.matrices, expected.matrices):
        assert type(res) == type(exp)
        if isinstance(res, tm.DenseMatrix):
            np.testing.assert_array_equal(res, exp)
        elif isinstance(res, tm.SparseMatrix):
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
            "C(cat_1, spans_intercept=False) * cat_2 * cat_3",
            id="custom_contrasts",
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_matrix_against_pandas(df, formula, ensure_full_rank):
    num_in_scope = 2  # noqa
    model_df = formulaic.model_matrix(formula, df, ensure_full_rank=ensure_full_rank)
    model_tabmat = tm.from_formula(formula, df, ensure_full_rank=ensure_full_rank)
    np.testing.assert_array_equal(model_df.to_numpy(), model_tabmat.A)


@pytest.mark.parametrize(
    "formula, expected_names",
    [
        pytest.param("num_1 + num_2", ("Intercept", "num_1", "num_2"), id="numeric"),
        pytest.param("num_1 + num_2 - 1", ("num_1", "num_2"), id="no_intercept"),
        pytest.param(
            "cat_1", ("Intercept", "cat_1[T.b]", "cat_1[T.c]"), id="categorical"
        ),
        pytest.param(
            "cat_2 * cat_3",
            (
                "Intercept",
                "cat_2[T.y]",
                "cat_2[T.z]",
                "cat_3[T.2]",
                "cat_2[T.y]:cat_3[T.2]",
                "cat_2[T.z]:cat_3[T.2]",
            ),
            id="interaction",
        ),
        pytest.param(
            "poly(num_1, 3) - 1",
            ("poly(num_1, 3)[1]", "poly(num_1, 3)[2]", "poly(num_1, 3)[3]"),
            id="polynomial",
        ),
        pytest.param(
            "{np.log(num_1 ** 2)}", ("Intercept", "np.log(num_1 ** 2)"), id="functions"
        ),
    ],
)
def test_names_against_expectation(df, formula, expected_names):
    model_tabmat = tm.from_formula(formula, df, ensure_full_rank=True)
    assert model_tabmat.model_spec.column_names == expected_names


@pytest.mark.parametrize(
    "formula, expected_names",
    [
        pytest.param("cat_1", ("intercept", "cat_1__b", "cat_1__c"), id="categorical"),
        pytest.param(
            "cat_2 * cat_3",
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
            "{np.log(num_1 ** 2)}", ("intercept", "np.log(num_1 ** 2)"), id="functions"
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
    ],
)
def test_names_against_pandas(df, formula, ensure_full_rank):
    num_in_scope = 2  # noqa
    model_df = formulaic.model_matrix(formula, df, ensure_full_rank=ensure_full_rank)
    model_tabmat = tm.from_formula(formula, df, ensure_full_rank=ensure_full_rank)
    assert model_tabmat.model_spec.column_names == model_df.model_spec.column_names
    assert model_tabmat.model_spec.column_names == tuple(model_df.columns)
