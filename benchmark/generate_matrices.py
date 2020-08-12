import pickle

import numpy as np
from scipy import sparse as sps

import quantcore.matrix as mx


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


def get_matrix_path(name):
    return f"benchmark/data/{name}_data.pkl"


def get_all_benchmark_matrices():
    return {
        "dense": lambda: make_dense_matrices(int(4e4), 1000),
        "sparse": lambda: make_sparse_matrices(int(4e5), int(1e2)),
        "one_cat": lambda: make_cat_matrix_all_formats(int(1e6), int(1e5)),
        "two_cat": lambda: make_cat_matrices(int(1e6), int(1e3), int(1e3)),
        "dense_cat": lambda: make_dense_cat_matrices(int(3e6), 5, int(1e3), int(1e3)),
        "dense_smallcat": lambda: make_dense_cat_matrices(int(3e6), 5, 10, int(1e3)),
    }


def generate_matrices():
    benchmark_matrices = get_all_benchmark_matrices()
    for name, f in benchmark_matrices.items():
        mats = f()
        with open(get_matrix_path(name), "wb") as f:
            pickle.dump(mats, f)


if __name__ == "__main__":
    generate_matrices()
