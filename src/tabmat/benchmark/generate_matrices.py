import os
import pickle

import click
import numpy as np
from scipy import sparse as sps

import tabmat as tm


def make_dense_matrices(n_rows: int, n_cols: int) -> dict:
    """Make dense matrices for benchmarks."""
    dense_matrices = {"numpy_C": np.random.random((n_rows, n_cols))}
    dense_matrices["numpy_F"] = dense_matrices["numpy_C"].copy(order="F")
    assert dense_matrices["numpy_F"].flags["F_CONTIGUOUS"]
    dense_matrices["tabmat"] = tm.DenseMatrix(dense_matrices["numpy_C"])
    return dense_matrices


def make_cat_matrix(n_rows: int, n_cats: int) -> tm.CategoricalMatrix:
    """Make categorical matrix for benchmarks."""
    mat = tm.CategoricalMatrix(np.random.choice(np.arange(n_cats, dtype=int), n_rows))
    return mat


def make_cat_matrix_all_formats(n_rows: int, n_cats: int) -> dict:
    """Make categorical matrix with all formats for benchmarks."""
    mat = make_cat_matrix(n_rows, n_cats)
    d = {
        "tabmat": mat,
        "scipy.sparse csr": mat.tocsr(),
    }
    d["scipy.sparse csc"] = d["scipy.sparse csr"].tocsc()
    return d


def make_cat_matrices(n_rows: int, n_cat_cols_1: int, n_cat_cols_2: int) -> dict:
    """Make two categorical matrices for benchmarks."""
    two_cat_matrices = {
        "tabmat": tm.SplitMatrix(
            [
                make_cat_matrix(n_rows, n_cat_cols_1),
                make_cat_matrix(n_rows, n_cat_cols_2),
            ]
        )
    }
    two_cat_matrices["scipy.sparse csr"] = sps.hstack(
        [elt.tocsr() for elt in two_cat_matrices["tabmat"].matrices]
    )
    two_cat_matrices["scipy.sparse csc"] = two_cat_matrices[
        "scipy.sparse csr"
    ].tocsc()  # type: ignore
    return two_cat_matrices


def make_dense_cat_matrices(
    n_rows: int, n_dense_cols: int, n_cats_1: int, n_cats_2: int
) -> dict:
    """Make dense categorical matrices for benchmarks."""
    dense_block = np.random.random((n_rows, n_dense_cols))
    two_cat_matrices = [
        make_cat_matrix(n_rows, n_cats_1),
        make_cat_matrix(n_rows, n_cats_2),
    ]
    dense_cat_matrices = {
        "tabmat": tm.SplitMatrix(two_cat_matrices + [tm.DenseMatrix(dense_block)]),
        "scipy.sparse csr": sps.hstack(
            [elt.tocsr() for elt in two_cat_matrices] + [sps.csr_matrix(dense_block)]
        ),
    }
    dense_cat_matrices["scipy.sparse csc"] = dense_cat_matrices[
        "scipy.sparse csr"
    ].tocsc()
    return dense_cat_matrices


def make_sparse_matrices(n_rows: int, n_cols: int) -> dict:
    """Make sparse matrices for benchmarks."""
    mat = sps.random(n_rows, n_cols).tocsc()
    matrices = {
        "scipy.sparse csc": mat,
        "scipy.sparse csr": mat.tocsr(),
        "tabmat": tm.SparseMatrix(mat),
    }
    return matrices


def _get_matrix_path(name):
    return f"benchmark/data/{name}_data.pkl"


def get_all_benchmark_matrices():
    """Get all matrices used in benchmarks."""
    return {
        "dense": lambda: make_dense_matrices(int(4e6), 10),
        "sparse": lambda: make_sparse_matrices(int(4e5), int(1e2)),
        "sparse_narrow": lambda: make_sparse_matrices(int(3e6), 3),
        "sparse_wide": lambda: make_sparse_matrices(int(4e4), int(1e4)),
        "one_cat": lambda: make_cat_matrix_all_formats(int(1e6), int(1e5)),
        "two_cat": lambda: make_cat_matrices(int(1e6), int(1e3), int(1e3)),
        "dense_cat": lambda: make_dense_cat_matrices(int(3e6), 5, int(1e3), int(1e3)),
        "dense_smallcat": lambda: make_dense_cat_matrices(int(3e6), 5, 10, int(1e3)),
    }


# TODO: duplication with glm_benchmarks
def get_comma_sep_names(xs: str):
    """Return comma separated names from names in input string."""
    return [x.strip() for x in xs.split(",")]


def get_matrix_names():
    """Return names for benchmark_matrices."""
    return ",".join(get_all_benchmark_matrices().keys())


@click.command()
@click.option(
    "--matrix_name",
    type=str,
    help=(
        f"Specify a comma-separated list of matrices you want to build. "
        f"Leaving this blank will default to building all matrices. "
        f"Matrix options: {get_matrix_names()}"
    ),
)
def generate_matrices(matrix_name: str) -> None:
    """Generate example matrices for benchmarks."""
    all_benchmark_matrices = get_all_benchmark_matrices()

    if matrix_name is None:
        benchmark_matrices = list(all_benchmark_matrices.keys())
    else:
        benchmark_matrices = get_comma_sep_names(matrix_name)

    for name in benchmark_matrices:
        f = all_benchmark_matrices[name]
        mats = f()
        save_path = _get_matrix_path(name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as fname:
            pickle.dump(mats, fname)


if __name__ == "__main__":
    generate_matrices()
