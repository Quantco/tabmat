import numpy as np
import pytest

import tabmat as tm


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_float32(dtype):
    rng = np.random.default_rng(1234)
    input = rng.random([1000, 10]).astype(dtype)
    X = tm.DenseMatrix(input)
    beta = rng.random([10]).astype(dtype)
    out = X.sandwich(beta)
    assert not np.isnan(out).any()
