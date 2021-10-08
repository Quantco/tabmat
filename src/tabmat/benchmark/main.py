import pickle
import time
from typing import Dict, List, Union

import click
import numpy as np
import pandas as pd
from scipy import sparse as sps

import tabmat as tm
from tabmat.benchmark.generate_matrices import (
    get_all_benchmark_matrices,
    get_comma_sep_names,
    get_matrix_names,
    make_cat_matrices,
    make_cat_matrix_all_formats,
    make_dense_cat_matrices,
    make_dense_matrices,
    make_sparse_matrices,
)
from tabmat.benchmark.memory_tools import track_peak_mem


def _sandwich(mat: Union[tm.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray):
    if isinstance(mat, (tm.MatrixBase, tm.StandardizedMatrix)):
        mat.sandwich(vec)
    elif isinstance(mat, np.ndarray):
        (mat * vec[:, None]).T @ mat
    else:
        mat.T @ sps.diags(vec) @ mat
    return


def _transpose_matvec(
    mat: Union[tm.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray
):
    if isinstance(mat, (tm.MatrixBase, tm.StandardizedMatrix)):
        return mat.transpose_matvec(vec)
    else:
        return mat.T.dot(vec)


def _matvec(mat, vec: np.ndarray) -> np.ndarray:
    if isinstance(mat, (tm.MatrixBase, tm.StandardizedMatrix)):
        return mat.matvec(vec)
    else:
        return mat.dot(vec)


def _lvec_setup(matrices):
    return (np.random.random(next(iter(matrices.values())).shape[0]),)


def _rvec_setup(matrices):
    return (np.random.random(next(iter(matrices.values())).shape[1]),)


ops = {
    "matvec": (_rvec_setup, _matvec),
    "transpose-matvec": (_lvec_setup, _transpose_matvec),
    "sandwich": (_lvec_setup, _sandwich),
}


def get_op_names():
    """Get names of operations."""
    return ",".join(ops.keys())


def run_one_benchmark_set(
    matrices: Dict[
        str, Union[tm.MatrixBase, tm.StandardizedMatrix, np.ndarray, sps.spmatrix]
    ],
    include_baseline: bool,
    name: str,
    standardized: bool,
    ops_to_run,
    n_iterations: int,
    bench_memory: bool,
) -> pd.DataFrame:
    """Run a single round of benchmarks."""
    if not include_baseline:
        for k in list(matrices.keys()):
            if k != "tabmat":
                del matrices[k]

    if standardized:

        def _to_standardized_mat(mat):
            if isinstance(mat, tm.MatrixBase):
                return tm.StandardizedMatrix(mat, np.zeros(mat.shape[1]))
            print(
                f"""For benchmarking a {type(mat)}, the baseline matrix will not
                be standardized."""
            )
            return mat

        matrices = {k: _to_standardized_mat(v) for k, v in matrices.items()}

    times = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [ops_to_run, matrices.keys()],
            names=["operation", "storage"],
        ),
        columns=["memory", "time"],
    ).reset_index()

    for i, row in times.iterrows():
        mat_ = matrices[row["storage"]]
        setup_fnc, op_fnc = ops[row["operation"]]
        setup_data = setup_fnc(matrices)
        runtimes = []
        peak_mems = []
        for _ in range(n_iterations):
            start = time.time()
            if bench_memory:
                peak_mem = track_peak_mem(op_fnc, mat_, *setup_data)
            else:
                op_fnc(mat_, *setup_data)
                peak_mem = 0
            end = time.time()
            peak_mems.append(peak_mem)
            runtimes.append(end - start)

        # We want to get a consistent measure of runtime so we take the
        # minimum. Any increase in runtime is due to warmup or other
        # processes running at the same time.
        times["time"].iloc[i] = np.min(runtimes)

        # On the other hand, we want the maximum memory usage because this
        # metric is isolated to our current python process. Any lower
        # values will be because the highest memory usage was "missed" by
        # the tracker
        times["memory"].iloc[i] = np.max(peak_mems)

    times["design"] = name
    return times


@click.command()
@click.option(
    "--operation_name",
    type=str,
    help=(
        f"Specify a comma-separated list of operations you want to run. Leaving this blank "
        f"will default to running all operations. Operation options: {get_op_names()}"
    ),
)
@click.option(
    "--matrix_name",
    type=str,
    help=(
        f"Specify a comma-separated list of matrices you want to run or specify. "
        f"Leaving this blank will default to running all predefined matrices. "
        f"Matrix options: {get_matrix_names()} OR custom. If custom, specify details using "
        f"additional custom matrix options. See --dense, --sparse, --one_cat, --two_cat, "
        f"and --dense_cat options for more details"
    ),
)
@click.option(
    "--dense",
    nargs=2,
    multiple=True,
    help=(
        "Specify n_rows, n_cols for custom dense matrix. "
        "Only used if 'custom' included in matrix_name."
    ),
    default=None,
)
@click.option(
    "--sparse",
    nargs=2,
    multiple=True,
    help=(
        "Specify n_rows, n_cols for custom sparse matrix. "
        "Only used if 'custom' included in matrix_name."
    ),
    default=None,
)
@click.option(
    "--one_cat",
    nargs=2,
    multiple=True,
    help=(
        "Specify n_rows, n_cols for custom one_cat matrix. "
        "Only used if 'custom' included in matrix_name."
    ),
    default=None,
)
@click.option(
    "--two_cat",
    nargs=3,
    multiple=True,
    help=(
        "Specify n_rows, n_cols for custom two_cat matrix. "
        "Only used if 'custom' included in matrix_name."
    ),
    default=None,
)
@click.option(
    "--dense_cat",
    nargs=4,
    multiple=True,
    help=(
        "Specify n_rows, n_cols for custom dense_cat matrix. "
        "Only used if 'custom' included in matrix_name."
    ),
    default=None,
)
@click.option(
    "--bench_memory",
    type=bool,
    is_flag=True,
    help=(
        "Should we benchmark memory usage with tracemalloc. Turning this on will make "
        "the runtime benchmarks less useful due to memory benchmarking overhead. "
        "Also, when memory benchmarking is on, debuggers like pdb and ipdb seem to fail."
    ),
    default=False,
)
@click.option(
    "--n_iterations",
    type=int,
    help=(
        "How many times to re-run the benchmark. The maximum memory usage and minimum "
        "runtime will be reported. Higher numbers of iterations reduce noise. This defaults "
        "to 100 unless memory benchmarking is turned on in which case it will be 1."
    ),
    default=None,
)
@click.option(
    "--include_baseline",
    type=bool,
    is_flag=True,
    help="Should we include a numpy/scipy baseline performance benchmark.",
    default=False,
)
@click.option(
    "--standardized",
    type=bool,
    is_flag=True,
    help="Should we test with a tabmat.StandardizedMatrix?",
    default=False,
)
def run_all_benchmarks(
    operation_name: str,
    matrix_name: str,
    dense: List,
    sparse: List,
    one_cat: List,
    two_cat: List,
    dense_cat: List,
    bench_memory: bool,
    n_iterations: int,
    include_baseline: bool,
    standardized: bool,
):
    """
    Usage examples.

    python benchmark/main.py --operation_name matvec,transpose-matvec --matrix_name sparse --include_baseline\n
              operation           storage memory         time

    0            matvec  scipy.sparse csc      0   0.00129819\n
    1            matvec  scipy.sparse csr      0   0.00266385\n
    2            matvec  tabmat      0   0.00199628\n
    3  transpose-matvec  scipy.sparse csc      0  0.000838518\n
    4  transpose-matvec  scipy.sparse csr      0   0.00239468\n
    5  transpose-matvec  tabmat      0  0.000296116\n

    python benchmark/main.py --operation_name sandwich --matrix_name dense_cat --bench_memory\n

      operation           storage    memory      time\n
    0  sandwich  tabmat  52244505  0.159682\n

    python benchmark/main.py --operation_name matvec --matrix_name custom --sparse 3e6 1 --sparse 3e6 10 --dense 10 10\n
    operation           storage memory      time                            design \n
    0    matvec  tabmat      0  0.000006  dense, #rows:10, #cols:10      \n
    operation           storage memory      time                            design \n
    0    matvec  tabmat      0  0.046355  sparse, #rows:3000000, #cols:1 \n
    operation           storage memory      time                            design \n
    0    matvec  tabmat      0  0.048141  sparse, #rows:3000000, #cols:10\n
    """  # noqa
    if n_iterations is None:
        if bench_memory:
            n_iterations = 1
        else:
            n_iterations = 100

    if operation_name is None:
        ops_to_run = list(ops.keys())
    else:
        ops_to_run = get_comma_sep_names(operation_name)

    all_benchmark_matrices = get_all_benchmark_matrices()

    benchmark_matrices = {}
    if matrix_name is None:
        for k in all_benchmark_matrices.keys():
            with open(f"benchmark/data/{k}_data.pkl", "rb") as f:
                benchmark_matrices[k] = pickle.load(f)

    elif "custom" in matrix_name:
        if dense:
            for params in dense:
                n_rows, n_cols = (int(float(x)) for x in params)
                benchmark_matrices[
                    f"dense, #rows:{n_rows}, #cols:{n_cols}"
                ] = make_dense_matrices(n_rows, n_cols)
        if sparse:
            for params in sparse:
                n_rows, n_cols = (int(float(x)) for x in params)
                benchmark_matrices[
                    f"sparse, #rows:{n_rows}, #cols:{n_cols}"
                ] = make_sparse_matrices(n_rows, n_cols)
        if one_cat:
            for params in one_cat:
                n_rows, n_cats = (int(float(x)) for x in params)
                benchmark_matrices[
                    f"one_cat, #rows:{n_rows}, #cats:{n_cats}"
                ] = make_cat_matrix_all_formats(n_rows, n_cats)
        if two_cat:
            for params in two_cat:
                n_rows, n_cat_cols_1, n_cat_cols_2 = (int(float(x)) for x in params)
                benchmark_matrices[
                    f"two_cat #rows:{n_rows}, #cats_1:{n_cat_cols_1}, #cats_2:{n_cat_cols_2}"
                ] = make_cat_matrices(n_rows, n_cat_cols_1, n_cat_cols_2)
        if dense_cat:
            for params in dense_cat:
                n_rows, n_dense_cols, n_cat_cols_1, n_cat_cols_2 = (
                    int(float(x)) for x in params
                )
                benchmark_matrices[
                    f"dense_cat #rows:{n_rows}, #dense:{n_dense_cols}, "
                    f" cats_1:{n_cat_cols_1}, #cats_2:{n_cat_cols_2}"
                ] = make_dense_cat_matrices(
                    n_rows, n_dense_cols, n_cat_cols_1, n_cat_cols_2
                )
    else:
        for k in get_comma_sep_names(matrix_name):
            with open(f"benchmark/data/{k}_data.pkl", "rb") as f:
                benchmark_matrices[k] = pickle.load(f)

    for name, matrices in benchmark_matrices.items():
        time_bench = run_one_benchmark_set(
            matrices,
            include_baseline,
            name,
            standardized,
            ops_to_run,
            n_iterations,
            False,
        )

        if bench_memory:
            memory_bench = run_one_benchmark_set(
                matrices,
                include_baseline,
                name,
                standardized,
                ops_to_run,
                1,
                True,
            )

            full_bench = pd.merge(
                memory_bench[["operation", "storage", "memory"]],
                time_bench[["operation", "storage", "time", "design"]],
                on=["operation", "storage"],
            )
        else:
            full_bench = time_bench[["operation", "storage", "time", "design"]]
        print(full_bench)

        full_bench.to_csv(f"benchmark/data/{name}_bench.csv", index=False)


if __name__ == "__main__":
    run_all_benchmarks()
