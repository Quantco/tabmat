import pickle
import time
from typing import Union

import click
import numpy as np
import pandas as pd
from generate_matrices import (
    get_all_benchmark_matrices,
    get_comma_sep_names,
    get_matrix_names,
)
from memory_tools import track_peak_mem
from scipy import sparse as sps

import quantcore.matrix as mx


def sandwich(mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray):
    if isinstance(mat, (mx.MatrixBase, mx.StandardizedMatrix)):
        mat.sandwich(vec)
    elif isinstance(mat, np.ndarray):
        (mat * vec[:, None]).T @ mat
    else:
        mat.T @ sps.diags(vec) @ mat
    return


def transpose_matvec(
    mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray
):
    if isinstance(mat, (mx.MatrixBase, mx.StandardizedMatrix)):
        out = np.zeros(mat.shape[1])
        return mat.transpose_matvec(vec, out=out)
    else:
        return mat.T.dot(vec)


def matvec(mat, vec: np.ndarray) -> np.ndarray:
    if isinstance(mat, (mx.MatrixBase, mx.StandardizedMatrix)):
        out = np.zeros(mat.shape[0])
        return mat.matvec(vec, out=out)
    else:
        return mat.dot(vec)


def lvec_setup(matrices):
    return (np.random.random(next(iter(matrices.values())).shape[0]),)


def rvec_setup(matrices):
    return (np.random.random(next(iter(matrices.values())).shape[1]),)


ops = {
    "matvec": (rvec_setup, matvec),
    "transpose-matvec": (lvec_setup, transpose_matvec),
    "sandwich": (lvec_setup, sandwich),
}


def get_op_names():
    return ",".join(ops.keys())


@click.command()
@click.option(
    "--operation_name",
    type=str,
    help=f"Specify a comma-separated list of operations you want to run. Leaving this blank will default to running all operations. Operation options: {get_op_names()}",
)
@click.option(
    "--matrix_name",
    type=str,
    help=f"Specify a comma-separated list of matrices you want to run. Leaving this blank will default to running all matrices. Matrix options: {get_matrix_names()}",
)
@click.option(
    "--bench_memory",
    type=bool,
    is_flag=True,
    help="Should we benchmark memory usage with tracemalloc. Turning this on will make the runtime benchmarks less useful due to memory benchmarking overhead. Also, when memory benchmarking is on, debuggers like pdb and ipdb seem to fail.",
    default=False,
)
@click.option(
    "--n_iterations",
    type=int,
    help="How many times to re-run the benchmark. The maximum memory usage and minimum runtime will be reported. Higher numbers of iterations reduce noise. This defaults to 100 unless memory benchmarking is turned on in which case it will be 1.",
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
    help="Should we test with a quantcore.matrix.StandardizedMatrix?",
    default=False,
)
def run_all_benchmarks(
    operation_name: str,
    matrix_name: str,
    bench_memory: bool,
    n_iterations: int,
    include_baseline: bool,
    standardized: bool,
):
    """
    Usage examples:

    python benchmark/main.py --operation_name matvec,transpose-matvec --matrix_name sparse --include_baseline\n
              operation           storage memory         time\n
    0            matvec  scipy.sparse csc      0   0.00129819\n
    1            matvec  scipy.sparse csr      0   0.00266385\n
    2            matvec  quantcore.matrix      0   0.00199628\n
    3  transpose-matvec  scipy.sparse csc      0  0.000838518\n
    4  transpose-matvec  scipy.sparse csr      0   0.00239468\n
    5  transpose-matvec  quantcore.matrix      0  0.000296116\n

    python benchmark/main.py --operation_name sandwich --matrix_name dense_cat --bench_memory

      operation           storage    memory      time\n
    0  sandwich  quantcore.matrix  52244505  0.159682
    """
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

    if matrix_name is None:
        benchmark_matrices = list(all_benchmark_matrices.keys())
    else:
        benchmark_matrices = get_comma_sep_names(matrix_name)

    for name in benchmark_matrices:
        with open(f"benchmark/data/{name}_data.pkl", "rb") as f:
            matrices = pickle.load(f)

        if not include_baseline:
            for k in list(matrices.keys()):
                if k != "quantcore.matrix":
                    del matrices[k]

        # ES note: Mysterious legacy code.
        if name not in ["dense"]:
            del matrices["scipy.sparse csr"]

        if standardized:

            def _to_standardized_mat(mat):
                if isinstance(mat, mx.MatrixBase):
                    return mx.StandardizedMatrix(mat, np.zeros(mat.shape[1]))
                print(
                    f"""For benchmarking a {type(mat)}, the baseline matrix will not
                    be standardized."""
                )
                return mat

            matrices = {k: _to_standardized_mat(v) for k, v in matrices.items()}

        times = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [ops_to_run, matrices.keys()], names=["operation", "storage"],
            ),
            columns=["memory", "time"],
        ).reset_index()

        for i, row in times.iterrows():
            mat_ = matrices[row["storage"]]
            setup_fnc, op_fnc = ops[row["operation"]]
            setup_data = setup_fnc(matrices)
            runtimes = []
            peak_mems = []
            for j in range(n_iterations):
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
        print(times)

        times.to_csv(f"benchmark/{name}_times.csv", index=False)


if __name__ == "__main__":
    run_all_benchmarks()
