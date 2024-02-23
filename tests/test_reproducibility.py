import numpy as np
import pandas as pd
import pytest

import tabmat as tm

N = 100
K = 5


@pytest.fixture
def df():
    rng = np.random.default_rng(1234)
    return pd.DataFrame(
        pd.Categorical(rng.integers(low=0, high=K - 1, size=N), categories=range(K))
    )


@pytest.mark.parametrize("cat_threshold", [K, K + 1])
def test_mat_transpose_vec(df, cat_threshold):
    rng = np.random.default_rng(1234)
    vec = rng.normal(size=N)
    mat = tm.from_pandas(df, cat_threshold=cat_threshold)
    np.testing.assert_equal(mat.transpose_matvec(vec), mat.transpose_matvec(vec))
