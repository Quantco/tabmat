import numpy as np

import tabmat as tm


def test_float32():
    rng = np.random.default_rng(1234)
    input = rng.random([1000, 10]).astype(np.float32)
    X = tm.DenseMatrix(input)
    beta = rng.random([10]).astype(np.float32)
    out = X.sandwich(beta)
    assert not np.isnan(out).any()
