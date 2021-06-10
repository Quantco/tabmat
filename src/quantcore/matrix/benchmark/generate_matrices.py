import pickle

import click
import numpy as np
from scipy import sparse as sps

import quantcore.matrix as mx


def _make_dense_matrices(n_rows: int, n_cols: int) -> dict:
    dense_matrices = {"numpy_C": np.random.random((n_rows, n_cols))}
    dense_matrices["numpy_F"] = dense_matrices["numpy_C"].copy(order="F")
    assert dense_matrices["numpy_F"].flags["F_CONTIGUOUS"]
    dense_matrices["quantcore.matrix"] = mx.DenseMatrix(dense_matrices["numpy_C"])
    return dense_matrices


def _make_cat_matrix(n_rows: int, n_cats: int) -> mx.CategoricalMatrix:
    mat = mx.CategoricalMatrix(np.random.choice(np.arange(n_cats, dtype=int), n_rows))
    return mat


def _make_cat_matrix_all_formats(n_rows: int, n_cats: int) -> dict:
    mat = _make_cat_matrix(n_rows, n_cats)
    d = {
        "quantcore.matrix": mat,
        "scipy.sparse csr": mat.tocsr(),
    }
    d["scipy.sparse csc"] = d["scipy.sparse csr"].tocsc()
    return d


def _make_cat_matrices(n_rows: int, n_cat_cols_1: int, n_cat_cols_2: int) -> dict:
    two_cat_matrices = {
        "quantcore.matrix": mx.SplitMatrix(
            [
                _make_cat_matrix(n_rows, n_cat_cols_1),
                _make_cat_matrix(n_rows, n_cat_cols_2),
            ]
        )
    }
    two_cat_matrices["scipy.sparse csr"] = sps.hstack(
        [elt.tocsr() for elt in two_cat_matrices["quantcore.matrix"].matrices]
    )
    two_cat_matrices["scipy.sparse csc"] = two_cat_matrices["scipy.sparse csr"].tocsc()
    return two_cat_matrices


def _make_dense_cat_matrices(
    n_rows: int, n_dense_cols: int, n_cats_1: int, n_cats_2: int
) -> dict:

    dense_block = np.random.random((n_rows, n_dense_cols))
    two_cat_matrices = [
        _make_cat_matrix(n_rows, n_cats_1),
        _make_cat_matrix(n_rows, n_cats_2),
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


def _make_sparse_matrices(n_rows: int, n_cols: int) -> dict:
    mat = sps.random(n_rows, n_cols).tocsc()
    matrices = {
        "scipy.sparse csc": mat,
        "scipy.sparse csr": mat.tocsr(),
        "quantcore.matrix": mx.SparseMatrix(mat),
    }
    return matrices


def _get_matrix_path(name):
    return f"benchmark/data/{name}_data.pkl"


def _get_all_benchmark_matrices():
    return {
        "dense": lambda: _make_dense_matrices(int(4e4), 1000),
        "sparse": lambda: _make_sparse_matrices(int(4e5), int(1e2)),
        "sparse_narrow": lambda: _make_sparse_matrices(int(3e6), 3),
        "sparse_wide": lambda: _make_sparse_matrices(int(4e4), int(1e4)),
        "one_cat": lambda: _make_cat_matrix_all_formats(int(1e6), int(1e5)),
        "two_cat": lambda: _make_cat_matrices(int(1e6), int(1e3), int(1e3)),
        "dense_cat": lambda: _make_dense_cat_matrices(int(3e6), 5, int(1e3), int(1e3)),
        "dense_smallcat": lambda: _make_dense_cat_matrices(int(3e6), 5, 10, int(1e3)),
    }


# TODO: duplication with glm_benchmarks
def _get_comma_sep_names(xs: str):
    return [x.strip() for x in xs.split(",")]


def _get_matrix_names():
    return ",".join(_get_all_benchmark_matrices().keys())


@click.command()
@click.option(
    "--matrix_name",
    type=str,
    help=(
        f"Specify a comma-separated list of matrices you want to build. "
        f"Leaving this blank will default to building all matrices. "
        f"Matrix options: {_get_matrix_names()}"
    ),
)
def generate_matrices(matrix_name: str) -> None:
    """Generate example matrices for benchmarks."""
    all_benchmark_matrices = _get_all_benchmark_matrices()

    if matrix_name is None:
        benchmark_matrices = list(all_benchmark_matrices.keys())
    else:
        benchmark_matrices = _get_comma_sep_names(matrix_name)

    for name in benchmark_matrices:
        f = all_benchmark_matrices[name]
        mats = f()
        with open(_get_matrix_path(name), "wb") as fname:
            pickle.dump(mats, fname)


if __name__ == "__main__":
    generate_matrices()
