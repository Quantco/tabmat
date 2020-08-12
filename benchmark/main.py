import pickle
import time
from typing import Union

import click
import numpy as np
import pandas as pd
from scipy import sparse as sps

import quantcore.matrix as mx

from .generate_matrices import get_all_benchmark_matrices
from .memory_tools import track_peak_mem


def sandwich(mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray):
    if isinstance(mat, mx.MatrixBase):
        mat.sandwich(vec)
    elif isinstance(mat, np.ndarray):
        (mat * vec[:, None]).T @ mat
    else:
        mat.T @ sps.diags(vec) @ mat
    return


def transpose_matvec(
    mat: Union[mx.MatrixBase, np.ndarray, sps.csc_matrix], vec: np.ndarray
):
    if isinstance(mat, mx.MatrixBase):
        out = np.zeros(mat.shape[1])
        return mat.transpose_matvec(vec, out=out)
    else:
        return mat.T.dot(vec)


def matvec(mat, vec):
    if isinstance(mat, mx.MatrixBase):
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
    "tranpose-matvec": (lvec_setup, transpose_matvec),
    "sandwich": (lvec_setup, sandwich),
}


# TODO: duplication with glm_benchmarks
def get_comma_sep_names(xs: str):
    return [x.strip() for x in xs.split(",")]


def get_problem_names():
    return ",".join(get_all_benchmark_matrices().keys())


def get_op_names():
    return ",".join(ops.keys())


@click.command()
@click.option(
    "--operation_name",
    type=str,
    help=f"Specify a comma-separated list of operations you want to run. Leaving this blank will default to running all operations. Operation options: {get_op_names()}",
)
@click.option(
    "--problem_name",
    type=str,
    help=f"Specify a comma-separated list of problems you want to run. Leaving this blank will default to running all problems. Problems options: {get_problem_names()}",
)
def run_all_benchmarks(operation_name, problem_name):
    # The memory benchmark slows down the runtime and makes it less
    # consistent so it's nice to be able to turn it off.
    # Also, weirdly, the MemoryPoller causes ipdb to not work correctly?!
    # Maybe something about the polling frequency and time to store a
    # snapshot being too high.

    should_bench_memory = False

    if should_bench_memory:
        n_iterations = 1
    else:
        n_iterations = 100

    include_baseline = True

    ops_to_run = get_comma_sep_names(operation_name)

    benchmark_matrices = get_all_benchmark_matrices()

    for name, f in benchmark_matrices.items():
        with open(f"benchmark/data/{name}_data.pkl", "rb") as f:
            matrices = pickle.load(f)

        if not include_baseline:
            for k in list(matrices.keys()):
                if k != "quantcore.matrix":
                    del matrices[k]

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
                if should_bench_memory:
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

        print(times)

        times.to_csv(f"benchmark/{name}_times.csv", index=False)


if __name__ == "__main__":
    run_all_benchmarks()
