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
from tabmat.formula import TabmatMaterializer


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
        pytest.param(
            "C(num_1)",
            id="convert_to_categorical",
        ),
        pytest.param(
            "C(cat_1, spans_intercept=False) * cat_2 * cat_3",
            id="custom_contrasts",
        ),
    ],
)
def test_names_against_pandas(df, formula, ensure_full_rank):
    num_in_scope = 2  # noqa
    model_df = formulaic.model_matrix(formula, df, ensure_full_rank=ensure_full_rank)
    model_tabmat = tm.from_formula(formula, df, ensure_full_rank=ensure_full_rank)
    assert model_tabmat.model_spec.column_names == model_df.model_spec.column_names
    assert model_tabmat.model_spec.column_names == tuple(model_df.columns)


FORMULAIC_TESTS = {
    # '<formula>': (<full_rank_names>, <names>, <full_rank_null_names>, <null_rows>)
    "a": (["Intercept", "a"], ["Intercept", "a"], ["Intercept", "a"], 2),
    "A": (
        ["Intercept", "A[T.b]", "A[T.c]"],
        ["Intercept", "A[T.a]", "A[T.b]", "A[T.c]"],
        ["Intercept", "A[T.c]"],
        2,
    ),
    "C(A)": (
        ["Intercept", "C(A)[T.b]", "C(A)[T.c]"],
        ["Intercept", "C(A)[T.a]", "C(A)[T.b]", "C(A)[T.c]"],
        ["Intercept", "C(A)[T.c]"],
        2,
    ),
    "A:a": (
        ["Intercept", "A[T.a]:a", "A[T.b]:a", "A[T.c]:a"],
        ["Intercept", "A[T.a]:a", "A[T.b]:a", "A[T.c]:a"],
        ["Intercept", "A[T.a]:a"],
        1,
    ),
    "A:B": (
        [
            "Intercept",
            "B[T.b]",
            "B[T.c]",
            "A[T.b]:B[T.a]",
            "A[T.c]:B[T.a]",
            "A[T.b]:B[T.b]",
            "A[T.c]:B[T.b]",
            "A[T.b]:B[T.c]",
            "A[T.c]:B[T.c]",
        ],
        [
            "Intercept",
            "A[T.a]:B[T.a]",
            "A[T.b]:B[T.a]",
            "A[T.c]:B[T.a]",
            "A[T.a]:B[T.b]",
            "A[T.b]:B[T.b]",
            "A[T.c]:B[T.b]",
            "A[T.a]:B[T.c]",
            "A[T.b]:B[T.c]",
            "A[T.c]:B[T.c]",
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

        # Tabmat does not allo NAs in categoricals
        if formula == "a":
            mm = TabmatMaterializer(data_with_nulls).get_model_matrix(
                formula, na_action="ignore"
            )
            assert isinstance(mm, tm.MatrixBase)
            assert mm.shape == (3, len(tests[0]) + (-1 if "A" in formula else 0))

            if formula != "C(A)":  # C(A) pre-encodes the data, stripping out nulls.
                with pytest.raises(ValueError):
                    TabmatMaterializer(data_with_nulls).get_model_matrix(
                        formula, na_action="raise"
                    )

    def test_state(self, materializer):
        mm = materializer.get_model_matrix("center(a) - 1")
        assert isinstance(mm, tm.MatrixBase)
        assert list(mm.model_spec.column_names) == ["center(a)"]
        assert np.allclose(mm.getcol(0).squeeze(), [-1, 0, 1])

        mm2 = TabmatMaterializer(pd.DataFrame({"a": [4, 5, 6]})).get_model_matrix(
            mm.model_spec
        )
        assert isinstance(mm2, tm.MatrixBase)
        assert list(mm2.model_spec.column_names) == ["center(a)"]
        assert np.allclose(mm2.getcol(0).squeeze(), [2, 3, 4])

        mm3 = mm.model_spec.get_model_matrix(pd.DataFrame({"a": [4, 5, 6]}))
        assert isinstance(mm3, tm.MatrixBase)
        assert list(mm3.model_spec.column_names) == ["center(a)"]
        assert np.allclose(mm3.getcol(0).squeeze(), [2, 3, 4])

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
        encoded_matrix = encoded_factor["B[a]"].set_name("B[a]").to_tabmat()
        assert list(encoded_matrix.cat) == ["B[a][T.a]", "B[a][T.b]", "B[a][T.c]"]

    @pytest.mark.xfail(reason="Cannot create an empty SplitMatrix in tabmat")
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
        assert list(m.model_spec.column_names) == ["A[T.a]", "A[T.b]", "A[T.c]"]

        m2 = TabmatMaterializer(data2).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m2.model_spec.column_names) == ["A[T.a]", "A[T.b]", "A[T.c]"]

        m3 = TabmatMaterializer(data3).get_model_matrix("A + 0", ensure_full_rank=False)
        assert list(m3.model_spec.column_names) == ["A[T.c]", "A[T.b]", "A[T.a]"]

    def test_term_clustering(self, materializer):
        assert materializer.get_model_matrix(
            "a + b + a:A + b:A"
        ).model_spec.column_names == (
            "Intercept",
            "a",
            "b",
            "a:A[T.b]",
            "a:A[T.c]",
            "b:A[T.b]",
            "b:A[T.c]",
        )
        assert materializer.get_model_matrix(
            "a + b + a:A + b:A", cluster_by="numerical_factors"
        ).model_spec.column_names == (
            "Intercept",
            "a",
            "a:A[T.b]",
            "a:A[T.c]",
            "b",
            "b:A[T.b]",
            "b:A[T.c]",
        )

    def test_model_spec_pickleable(self, materializer):
        o = BytesIO()
        ms = materializer.get_model_matrix("a ~ a:A")
        pickle.dump(ms.model_spec, o)
        o.seek(0)
        ms2 = pickle.load(o)
        assert isinstance(ms, formulaic.parser.types.Structured)
        assert ms2.lhs.formula.root == ["a"]
