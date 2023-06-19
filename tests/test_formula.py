import formulaic
import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("ensure_full_rank", [True, False])
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
def test_against_pandas(df, formula, ensure_full_rank):
    num_in_scope = 2  # noqa
    model_df = formulaic.model_matrix(formula, df, ensure_full_rank=ensure_full_rank)
    model_tabmat = tm.from_formula(formula, df, ensure_full_rank=ensure_full_rank)
    np.testing.assert_array_equal(model_df.to_numpy(), model_tabmat.A)
