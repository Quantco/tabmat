import time
import tracemalloc
from threading import Thread
from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

import quantcore.matrix as mx

should_bench_memory = False
n_iterations = 200


class MemoryPoller:
    """
    Example usage:

    with MemoryPoller() as mp:
        do some stuff here
        print('initial memory usage', mp.initial_memory)
        print('max memory usage', mp.max_memory)
        excess_memory_used = mp.max_memory - mp.initial_memory
    """

    def poll_max_memory_usage(self):
        while not self.stop_polling:
            self.snapshots.append(tracemalloc.take_snapshot())
            time.sleep(1e-3)

    def __enter__(self):
        tracemalloc.start()
        self.stop_polling = False
        self.snapshots = [tracemalloc.take_snapshot()]
        self.t = Thread(target=self.poll_max_memory_usage)
        self.t.start()
        return self

    def __exit__(self, *excargs):
        self.stop_polling = True
        self.t.join()
        self.final_usage, self.peak_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()


def track_peak_mem(f):
    def g(*args, **kwargs):
        # The memory benchmark slows down the runtime and makes it less
        # consistent so it's nice to be able to turn it off.
        # Also, weirdly, the MemoryPoller causes ipdb to not work correctly?!
        # Maybe something about the polling frequency and time to store a
        # snapshot being too high.
        if should_bench_memory:
            with MemoryPoller() as mp:
                f(*args, **kwargs)
            for s in mp.snapshots:
                top_stats = s.statistics("lineno")
                print("[ Top 2 ]")
                for stat in top_stats[:2]:
                    print(stat)
            return mp.peak_usage
        else:
            f(*args, **kwargs)
            return 0

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


def run_benchmarks(ops, matrices: dict) -> pd.DataFrame:
    assert isinstance(matrices, dict)
    vec = np.random.random(next(iter(matrices.values())).shape[1])
    vec2 = np.random.random(next(iter(matrices.values())).shape[0])

    times = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [ops, matrices.keys()], names=["operation", "storage"],
        ),
        columns=["memory", "time"],
    ).reset_index()

    for i, row in times.iterrows():
        mat_ = matrices[row["storage"]]
        op = row["operation"]
        runtimes = []
        for j in range(n_iterations):
            start = time.time()
            if op == "matrix-vector":
                peak_mem = dot(mat_, vec)
            elif op == "matrix-transpose-vector":
                peak_mem = transpose_dot(mat_, vec2)
            else:
                peak_mem = sandwich(mat_, vec2)
            end = time.time()
            runtimes.append(end - start)
        times["time"].iloc[i] = np.min(runtimes)
        times["memory"].iloc[i] = peak_mem
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


def make_sparse_matrices(n_rows: int, n_cols: int) -> dict:
    mat = sps.random(n_rows, n_cols).tocsc()
    matrices = {
        "scipy.sparse csc": mat,
        "scipy.sparse csr": mat.tocsr(),
        "quantcore.matrix": mx.SparseMatrix(mat),
    }
    return matrices


def main():
    n_rows = int(1e6)
    benchmark_matrices = {
        # "dense": lambda: make_dense_matrices(int(1e5), 1000),
        # "one_cat": lambda: make_cat_matrix_all_formats(n_rows, int(1e5)),
        # "sparse": lambda: make_sparse_matrices(n_rows, int(1e3)),
        # "two_cat": lambda: make_cat_matrices(n_rows, int(1e3), int(1e3)),
        "dense_cat": lambda: make_dense_cat_matrices(n_rows, 5, int(1e3), int(1e3)),
        # "dense_smallcat": lambda: make_dense_cat_matrices(n_rows, 5, 10, int(1e3)),
    }
    ops = ["matrix-vector", "sandwich", "matrix-transpose-vector"]
    ops = ["matrix-vector", "matrix-transpose-vector"]
    for name, f in benchmark_matrices.items():
        mats = f()
        del mats["scipy.sparse csr"]
        del mats["scipy.sparse csc"]
        times = run_benchmarks(ops, mats)
        print(times)
        times.to_csv(f"benchmark/{name}_times.csv", index=False)


if __name__ == "__main__":
    main()
