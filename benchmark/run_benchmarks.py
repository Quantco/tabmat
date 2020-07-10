import time
import tracemalloc
from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

import quantcore.matrix as mx


def track_peak_mem(f):
    def g(*args, **kwargs):
        tracemalloc.start()
        f(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    return g


@track_peak_mem
def sandwich(mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray):
    if isinstance(mat, mx.MatrixBase):
        mat.sandwich(vec)
    elif isinstance(mat, np.ndarray):
        (mat * vec[:, None]).T @ mat
    else:
        mat.T @ sps.diags(vec) @ mat
    return


@track_peak_mem
def transpose_dot(
    mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray
):
    if isinstance(mat, mx.MatrixBase):
        return mat.transpose_dot(vec)
    return mat.T @ vec


@track_peak_mem
def dot(mat, vec):
    return mat.dot(vec)


def run_benchmarks(matrices: dict) -> pd.DataFrame:
    vec = np.random.random(next(iter(matrices.values())).shape[1])
    vec2 = np.random.random(next(iter(matrices.values())).shape[0])

    times = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["dot", "sandwich", "transpose_dot"], matrices.keys()],
            names=["operation", "storage"],
        ),
        columns=["memory", "time"],
    ).reset_index()
    for i, row in times.iterrows():
        mat_ = matrices[row["storage"]]
        op = row["operation"]
        start = time.time()
        if op == "dot":
            peak_mem = dot(mat_, vec)
        elif op == "transpose_dot":
            peak_mem = transpose_dot(mat_, vec2)
        else:
            peak_mem = sandwich(mat_, vec2)

        end = time.time()
        times.iloc[i, -1] = end - start
        times.iloc[i, -2] = peak_mem
    return times


def make_dense_matrices(n_rows: int, n_cols: int) -> dict:
    dense_matrices = {"np_C": np.random.random((n_rows, n_cols))}
    dense_matrices["np_F"] = dense_matrices["np_C"].copy(order="F")
    assert dense_matrices["np_F"].flags["F_CONTIGUOUS"]
    dense_matrices["custom"] = mx.DenseMatrix(dense_matrices["np_C"])
    return dense_matrices


def make_cat_matrices(n_rows: int, n_cat_cols_1: int, n_cat_cols_2: int) -> dict:
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
    return two_cat_matrices


def main():
    n_rows = int(1e6)

    dense_times = run_benchmarks(make_dense_matrices(int(1e5), 1000))
    dense_times.to_csv("benchmark/dense_times.csv", index=False)

    two_cat_matrices = make_cat_matrices(n_rows, int(1e4), int(1e3))
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
