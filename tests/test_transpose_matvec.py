import narwhals as nw
import numpy as np
import pandas as pd

import tabmat as tm


def test_transpose_matvec_does_not_crash():
    # With high n_categories CategoricalMatrix.indices is read-only.
    # As a result, transpose_matvec method crashed with "buffer
    # source array is read-only".

    n = 797_586
    n_categories = 58_059
    categories = [f"cat[{i}]" for i in range(n_categories)]
    indices = np.linspace(0, n_categories - 1, n).round().astype(int)
    cat_vec = pd.Series(pd.Categorical.from_codes(indices, categories=categories))
    cat_vec_nw = nw.from_native(cat_vec, allow_series=True)
    weights = np.ones(n)
    cat_matrix_tm = tm.CategoricalMatrix(cat_vec_nw)

    result = cat_matrix_tm.transpose_matvec(weights)

    assert result is not None
