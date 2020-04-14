import os
import pickle

import click
import numpy as np

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.main import execute_problem_library, get_limited_problems

intercept = -3.366495969747295
coef = np.array(
    [
        0.27608816,
        0.35549287,
        0.32686169,
        -0.26129063,
        0.05631971,
        1.66835335,
        0.0,
        0.0,
        -0.25640353,
        0.25240303,
        -0.04934933,
        0.00838973,
        -0.08514918,
        0.01641579,
        0.0,
        0.0,
        0.54944044,
        0.0,
        0.0,
        -0.03864504,
        -0.03402387,
        -0.05737517,
        -0.00961314,
        0.16551897,
        0.41539209,
        -0.08141213,
        0.03119013,
        -0.04565951,
        0.12508365,
        0.0,
        -0.05832944,
        0.0,
        -0.33902941,
        0.0,
        0.0,
        0.0,
        0.03173001,
        0.36435994,
        0.0,
        -0.33060842,
        0.13377216,
        0.0,
        0.09336626,
        -0.41353699,
        0.0,
        0.10139023,
        0.19088491,
        0.13317361,
    ]
)

# For line-by-line profiling, use line_profiler:
# kernprof -lbv src/glm_benchmarks/profile_entry.py
#
# For stack sampling profiling, use py-spy:
# py-spy py-spy record -o profile.svg -- python src/glm_benchmarks/profile_entry.py
# py-spy top -- python src/glm_benchmarks/profile_entry.py


@click.command()
@click.option(
    "--num_rows",
    type=int,
    default=50000,
    help="Integer number of rows to run profiling on.",
)
@click.option(
    "--problem_names",
    default="simple_insurance_no_weights_lasso_poisson",
    help="Specify a comma-separated list of benchmark problems you want to run.",
)
@click.option(
    "--sparsify",
    is_flag=True,
    help="Convert an originally dense problem into a sparse one.",
)
@click.option(
    "--save_result",
    is_flag=True,
    help="Save the estimates for later golden master testing.",
)
@click.option(
    "--save_dir",
    default="golden_master",
    help="Where to find saved estimates for checking that estimates haven't changed.",
)
def main(num_rows, problem_names, sparsify, save_result, save_dir):
    problems = get_limited_problems(problem_names)
    for Pn in problems:
        print(f"benchmarking {Pn}")
        result = execute_problem_library(
            problems[Pn], sklearn_fork_bench, num_rows, sparsify
        )

        path = os.path.join(save_dir, Pn, str(num_rows) + ".npy")
        if save_result:
            save_baseline(path, result)
        else:
            test_against_baseline(path, result)
        print("")


def save_baseline(path, data):
    print("saving baseline estimates for later testing.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_against_baseline(path, data):
    print("loading baseline estimates for testing")
    with open(path, "rb") as f:
        baseline = pickle.load(f)
    np.testing.assert_almost_equal(data["intercept"], baseline["intercept"])
    np.testing.assert_almost_equal(data["coef"], baseline["coef"])
    print("test passed")
    print(f"baseline runtime = {baseline['runtime']}")
    print(f"current runtime = {data['runtime']}")


if __name__ == "__main__":
    main()
