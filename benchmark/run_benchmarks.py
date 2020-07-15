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
            [["matrix-vector", "sandwich", "matrix-transpose-vector"], matrices.keys()],
            names=["operation", "storage"],
        ),
        columns=["memory", "time"],
    ).reset_index()

    for i, row in times.iterrows():
        mat_ = matrices[row["storage"]]
        op = row["operation"]
        start = time.time()
        if op == "matrix-vector":
            peak_mem = dot(mat_, vec)
        elif op == "matrix-transpose-vector":
            peak_mem = transpose_dot(mat_, vec2)
        else:
            peak_mem = sandwich(mat_, vec2)

        end = time.time()
        times.iloc[i, -1] = end - start
        times.iloc[i, -2] = peak_mem
    return times


def make_dense_matrices(n_rows: int, n_cols: int) -> dict:
    dense_matrices = {"numpy_C": np.random.random((n_rows, n_cols))}
    dense_matrices["numpy_F"] = dense_matrices["numpy_C"].copy(order="F")
    assert dense_matrices["numpy_F"].flags["F_CONTIGUOUS"]
    dense_matrices["quantcore.matrix"] = mx.DenseMatrix(dense_matrices["numpy_C"])
    return dense_matrices


def make_cat_matrix(n_rows: int, n_cats: int) -> mx.CategoricalMatrix:
    mat = mx.CategoricalMatrix(np.random.choice(np.arange(n_cats, dtype=int), n_rows))
    return mat


def make_cat_matrix_all_formats(n_rows: int, n_cats: int) -> dict:
    mat = make_cat_matrix(n_rows, n_cats)
    d = {
        "quantcore.matrix": mat,
        "scipy.sparse csr": mat.tocsr(),
    }
    d["scipy.sparse csc"] = d["scipy.sparse csr"].tocsc()
    return d


def make_cat_matrices(n_rows: int, n_cat_cols_1: int, n_cat_cols_2: int) -> dict:
    two_cat_matrices = {
        "quantcore.matrix": mx.SplitMatrix(
            [
                make_cat_matrix(n_rows, n_cat_cols_1),
                make_cat_matrix(n_rows, n_cat_cols_2),
            ]
        )
    }
    two_cat_matrices["scipy.sparse csr"] = sps.hstack(
        [elt.tocsr() for elt in two_cat_matrices["quantcore.matrix"].matrices]
    )
    two_cat_matrices["scipy.sparse csc"] = two_cat_matrices["scipy.sparse csr"].tocsc()
    return two_cat_matrices


def make_dense_cat_matrices(
    n_rows: int, n_dense_cols: int, n_cats_1: int, n_cats_2: int
) -> dict:

    dense_block = np.random.random((n_rows, n_dense_cols))
    two_cat_matrices = [
        make_cat_matrix(n_rows, n_cats_1),
        make_cat_matrix(n_rows, n_cats_2),
    ]
    dense_cat_matrices = {
        "quantcore.matrix": mx.SplitMatrix(
            two_cat_matrices + [mx.DenseMatrix(dense_block)]
        ),
        "scipy.sparse csr": sps.hstack(
            [elt.tocsr() for elt in two_cat_matrices] + [sps.csr_matrix(dense_block)]
        ),
    }
    dense_cat_matrices["scipy.sparse csc"] = dense_cat_matrices[
        "scipy.sparse csr"
    ].tocsc()
    return dense_cat_matrices


def main():
    n_rows = int(1e6)
    benchmark_matrices = {
        "dense": lambda: make_dense_matrices(int(1e5), 1000),
        "one_cat": lambda: make_cat_matrix(n_rows, int(1e5)),
        "two_cat": lambda: make_cat_matrices(n_rows, int(1e3), int(1e3)),
        "dense_cat": lambda: make_dense_cat_matrices(n_rows, 5, int(1e3), int(1e3)),
    }
    for name, f in benchmark_matrices.items():
        times = run_benchmarks(f())
        times.to_csv(f"benchmark/{name}_times.csv", index=False)


if __name__ == "__main__":
    main()
