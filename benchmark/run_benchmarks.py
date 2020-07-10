import time
from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

import quantcore.matrix as mx


def sandwich(mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray):
    if isinstance(mat, mx.MatrixBase):
        return mat.sandwich(vec)
    elif isinstance(mat, np.ndarray):
        return (mat * vec[:, None]).T @ mat
    else:
        return mat.T @ sps.diags(vec) @ mat


def transpose_dot(
    mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray
):
    if isinstance(mat, mx.MatrixBase):
        return mat.transpose_dot(vec)
    return mat.T @ vec


def run_benchmarks(matrices: dict) -> pd.DataFrame:
    vec = np.random.random(next(iter(matrices.values())).shape[1])
    vec2 = np.random.random(next(iter(matrices.values())).shape[0])

    times = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["dot", "sandwich", "transpose_dot"], matrices.keys()],
            names=["operation", "storage"],
        ),
        columns=["time"],
    ).reset_index()
    for i, row in times.iterrows():
        mat_ = matrices[row["storage"]]
        op = row["operation"]
        start = time.time()
        if op == "dot":
            mat_.dot(vec)
        elif op == "transpose_dot":
            transpose_dot(mat_, vec2)
        else:
            sandwich(mat_, vec2)

        end = time.time()
        times.iloc[i, -1] = end - start
    return times


def main():
    n_rows = int(1e6)
    n_cat_cols_1 = int(1e4)
    n_cat_cols_2 = int(1e3)
    two_cat_matrices = {
        "custom": mx.SplitMatrix(
            [
                mx.CategoricalMatrix(
                    np.random.choice(np.arange(n_cat_cols_1, dtype=int), n_rows)
                ),
                mx.CategoricalMatrix(
                    np.random.choice(np.arange(n_cat_cols_2, dtype=int), n_rows)
                ),
            ]
        )
    }
    two_cat_matrices["csr"] = sps.hstack(
        [elt.tocsr() for elt in two_cat_matrices["custom"].matrices]
    )
    two_cat_matrices["csc"] = two_cat_matrices["csr"].tocsc()
    print(two_cat_matrices["custom"].shape)

    times = run_benchmarks(two_cat_matrices)
    times.to_csv("benchmark/two_cat_times.csv", index=False)

    dense_block = np.random.random((n_rows, 10))
    dense_cat_matrices = {
        "custom": mx.SplitMatrix(
            two_cat_matrices["custom"].matrices + [mx.DenseMatrix(dense_block)]
        ),
        "csr": sps.hstack([two_cat_matrices["csr"], sps.csr_matrix(dense_block)]),
    }
    dense_cat_matrices["csc"] = dense_cat_matrices["csr"].tocsc()
    print(dense_cat_matrices["custom"].shape)

    dense_cat_times = run_benchmarks(dense_cat_matrices)
    dense_cat_times.to_csv("benchmark/dense_cat_times.csv", index=False)


if __name__ == "__main__":
    main()
