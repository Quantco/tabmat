import numpy as np
import pytest
from click.testing import CliRunner

import tabmat as tm
from tabmat.benchmark.generate_matrices import (
    generate_matrices,
    get_all_benchmark_matrices,
)
from tabmat.benchmark.main import (
    get_comma_sep_names,
    get_op_names,
    run_one_benchmark_set,
)


@pytest.mark.parametrize(
    "cli_input",
    [
        "",
        "dense,sparse,sparse_narrow, sparse_wide,one_cat,two_cat,dense_cat,"
        "dense_smallcat",
    ],
)
def test_generate_matrices(cli_input: str):
    runner = CliRunner()
    runner.invoke(generate_matrices, cli_input)


@pytest.mark.parametrize("matrix_name", ["sparse,dense_cat", "dense"])
@pytest.mark.parametrize("include_baseline", [False, True])
@pytest.mark.parametrize("standardized", [False, True])
@pytest.mark.parametrize("bench_memory", [False, True])
def test_run_all_benchmarks(
    matrix_name: str, include_baseline: bool, standardized: bool, bench_memory: bool
):
    """
    Run 'run_all_benchmarks'.

    This only runs a few tests since running all of them is slow and memory-intensive.
    """
    benchmark_matrices = get_comma_sep_names(matrix_name)
    for name in benchmark_matrices:
        matrices = get_all_benchmark_matrices()[name]()
        run_one_benchmark_set(
            matrices,
            include_baseline,
            name,
            standardized,
            get_comma_sep_names(get_op_names()),
            1,
            bench_memory,
        )


def test_run_one_benchmark_set_records_timings():
    """Timings must actually be written to the results frame.

    Regression test: under pandas copy-on-write the chained assignment
    ``times["time"].iloc[i] = ...`` is rejected and the value silently dropped,
    leaving every timing at its initial value.
    """
    rng = np.random.default_rng(0)
    matrices = {"tabmat": tm.DenseMatrix(rng.standard_normal((50, 4)))}
    times = run_one_benchmark_set(
        matrices,
        include_baseline=False,
        name="dense",
        standardized=False,
        ops_to_run=["matvec", "transpose-matvec", "sandwich"],
        n_iterations=1,
        bench_memory=False,
    )
    assert times["time"].notna().all()
    assert times["memory"].notna().all()
