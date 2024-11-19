import numpy as np

import tabmat

np.set_printoptions(suppress=True)

for dtype in [np.float32, np.float64]:
    X = np.array(
        [
            [46.231056, 126.05263, 144.46439],
            [46.231224, 128.66818, 0.7667693],
            [46.231186, 104.97506, 193.8872],
            [46.230835, 130.10156, 143.88954],
            [46.230896, 116.76007, 7.5629334],
        ],
        dtype=dtype,
    )
    v = np.array(
        [0.12428328, 0.67062443, 0.6471895, 0.6153851, 0.38367754], dtype=dtype
    )

    weights = np.full(X.shape[0], 1 / X.shape[0], dtype=dtype)

    stmat, out_means, col_stds = tabmat.DenseMatrix(X).standardize(weights, True, True)

    print(stmat.toarray().T @ v)
    print(stmat.transpose_matvec(v))

    # compute by hand
    res = np.zeros(X.shape[1], dtype=dtype)
    for col in range(X.shape[1]):
        res[col] += (
            stmat.shift[col] + stmat.mult[col] * stmat.mat.toarray()[:, col]
        ) @ v

    print(res)
    print("\n")
