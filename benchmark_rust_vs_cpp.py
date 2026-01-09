#!/usr/bin/env python
"""Benchmark comparing C++/Cython vs Rust implementations of tabmat.

Usage:
    # Benchmark current implementation
    python benchmark_rust_vs_cpp.py

    # To compare both, run with C++/Cython first, then with Rust:
    # 1. Install conda tabmat: pixi run pip uninstall -y tabmat && pixi install
    # 2. Run: python benchmark_rust_vs_cpp.py --save cpp_results.json
    # 3. Install Rust tabmat: pixi run pip install -e .
    # 4. Run: python benchmark_rust_vs_cpp.py --save rust_results.json
    #         --compare cpp_results.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import tabmat as tm


def get_implementation_info():
    """Detect which implementation is being used."""
    try:
        from tabmat.ext import dense

        # Check if it's the Rust compatibility layer
        if hasattr(dense, "__file__") and "rust_compat" in str(dense.__file__):
            return "Rust"
        return "C++/Cython"
    except ImportError:
        # If we can't import from ext, try to detect rust_compat
        try:
            import tabmat.ext.rust_compat  # noqa: F401

            return "Rust"
        except ImportError:
            return "Unknown"


def create_dense_matrix(n_rows, n_cols, seed=42):
    """Create a dense matrix."""
    rng = np.random.RandomState(seed)
    arr = rng.randn(n_rows, n_cols)
    return tm.DenseMatrix(np.asfortranarray(arr))


def create_sparse_matrix(n_rows, n_cols, density=0.1, seed=42):
    """Create a sparse matrix."""
    from scipy import sparse as sp

    rng = np.random.RandomState(seed)
    arr = rng.randn(n_rows, n_cols)
    mask = rng.rand(n_rows, n_cols) < density
    arr = arr * mask
    return tm.SparseMatrix(sp.csc_matrix(arr))


def create_categorical_matrix(n_rows, n_categories, seed=42):
    """Create a categorical matrix."""
    rng = np.random.RandomState(seed)
    codes = rng.randint(0, n_categories, size=n_rows)
    categories = np.array([f"cat_{i}" for i in range(n_categories)])
    return tm.CategoricalMatrix(codes, categories)


def create_split_matrix(n_rows, n_dense_cols, n_sparse_cols, n_cat_cols, seed=42):
    """Create a split matrix with mixed types."""
    matrices = []

    if n_dense_cols > 0:
        matrices.append(create_dense_matrix(n_rows, n_dense_cols, seed))

    if n_sparse_cols > 0:
        matrices.append(create_sparse_matrix(n_rows, n_sparse_cols, 0.1, seed + 1))

    if n_cat_cols > 0:
        # Create categorical columns (10 categories each)
        for i in range(n_cat_cols):
            matrices.append(create_categorical_matrix(n_rows, 10, seed + 2 + i))

    return tm.SplitMatrix(matrices)


def create_standardized_matrix(base_matrix):
    """Create a standardized version of a matrix."""
    n_cols = base_matrix.shape[1]
    shift = np.random.randn(n_cols) * 0.1
    mult = np.abs(np.random.randn(n_cols)) + 0.5
    return tm.StandardizedMatrix(base_matrix, shift=shift, mult=mult)


def benchmark_operation(name, func, n_iterations=10, warmup=2):
    """Benchmark an operation."""
    # Warmup
    for _ in range(warmup):
        func()

    # Actual timing
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "name": name,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def run_benchmarks(matrix_sizes, n_iterations=10):
    """Run comprehensive benchmarks on different matrix types and sizes."""
    impl = get_implementation_info()
    print(f"\n{'=' * 70}")
    print(f"Running benchmarks with {impl} implementation")
    print(f"{'=' * 70}\n")

    results = {"implementation": impl, "benchmarks": []}

    for size_name, config in matrix_sizes.items():
        n_rows = config["n_rows"]
        n_cols = config["n_cols"]

        print(f"\n{size_name}: {n_rows} rows × {n_cols} cols")
        print("-" * 70)

        # Dense matrix benchmarks
        print("  Dense matrix...")
        dense = create_dense_matrix(n_rows, n_cols)
        d = np.abs(np.random.randn(n_rows)) + 0.1
        v = np.random.randn(n_cols)

        result = benchmark_operation(
            f"{size_name}/dense/sandwich",
            lambda: dense.sandwich(d),
            n_iterations=n_iterations,
        )
        mean_ms = result["mean_time"] * 1000
        std_ms = result["std_time"] * 1000
        print(f"    sandwich: {mean_ms:.2f} ± {std_ms:.2f} ms")
        results["benchmarks"].append(result)

        result = benchmark_operation(
            f"{size_name}/dense/matvec",
            lambda: dense.matvec(v),
            n_iterations=n_iterations,
        )
        mean_ms = result["mean_time"] * 1000
        std_ms = result["std_time"] * 1000
        print(f"    matvec:   {mean_ms:.2f} ± {std_ms:.2f} ms")
        results["benchmarks"].append(result)

        # Sparse matrix benchmarks
        print("  Sparse matrix (10% density)...")
        sparse = create_sparse_matrix(n_rows, n_cols, density=0.1)

        result = benchmark_operation(
            f"{size_name}/sparse/sandwich",
            lambda: sparse.sandwich(d),
            n_iterations=n_iterations,
        )
        print(
            f"    sandwich: {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        result = benchmark_operation(
            f"{size_name}/sparse/matvec",
            lambda: sparse.matvec(v),
            n_iterations=n_iterations,
        )
        print(
            f"    matvec:   {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        # Categorical matrix benchmarks
        print("  Categorical matrix (10 categories)...")
        cat = create_categorical_matrix(n_rows, 10)
        d_cat = np.abs(np.random.randn(n_rows)) + 0.1
        v_cat = np.random.randn(10)

        result = benchmark_operation(
            f"{size_name}/categorical/sandwich",
            lambda: cat.sandwich(d_cat),
            n_iterations=n_iterations,
        )
        print(
            f"    sandwich: {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        result = benchmark_operation(
            f"{size_name}/categorical/matvec",
            lambda: cat.matvec(v_cat),
            n_iterations=n_iterations,
        )
        print(
            f"    matvec:   {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        # Split matrix benchmarks
        print("  Split matrix (mixed types)...")
        n_total = n_cols
        n_dense = n_total // 3
        n_sparse = n_total // 3
        n_cat = max(1, (n_total - n_dense - n_sparse) // 10)  # 10 cols per cat

        split = create_split_matrix(n_rows, n_dense, n_sparse, n_cat)
        split_cols = split.shape[1]
        d_split = np.abs(np.random.randn(n_rows)) + 0.1
        v_split = np.random.randn(split_cols)

        result = benchmark_operation(
            f"{size_name}/split/sandwich",
            lambda: split.sandwich(d_split),
            n_iterations=n_iterations,
        )
        print(
            f"    sandwich: {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        result = benchmark_operation(
            f"{size_name}/split/matvec",
            lambda: split.matvec(v_split),
            n_iterations=n_iterations,
        )
        print(
            f"    matvec:   {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        # Standardized matrix benchmarks
        print("  Standardized matrix (dense)...")
        std = create_standardized_matrix(dense)

        result = benchmark_operation(
            f"{size_name}/standardized/sandwich",
            lambda: std.sandwich(d),
            n_iterations=n_iterations,
        )
        print(
            f"    sandwich: {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        result = benchmark_operation(
            f"{size_name}/standardized/matvec",
            lambda: std.matvec(v),
            n_iterations=n_iterations,
        )
        print(
            f"    matvec:   {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

        # Sandwich with row subset (critical for GLM)
        print("  Dense sandwich with row subset...")
        rows = np.arange(0, n_rows, 2)  # Every other row

        result = benchmark_operation(
            f"{size_name}/dense/sandwich_rows",
            lambda: dense.sandwich(d, rows=rows),
            n_iterations=n_iterations,
        )
        print(
            f"    sandwich: {result['mean_time'] * 1000:.2f} ± {result['std_time'] * 1000:.2f} ms"  # noqa: E501
        )
        results["benchmarks"].append(result)

    return results


def compare_results(rust_results, cpp_results):
    """Compare Rust vs C++/Cython results."""
    print(f"\n{'=' * 70}")
    print("Performance Comparison: Rust vs C++/Cython")
    print(f"{'=' * 70}\n")

    # Create lookup for cpp results
    cpp_lookup = {b["name"]: b for b in cpp_results["benchmarks"]}

    speedups = []

    print(f"{'Benchmark':<50} {'Rust (ms)':<12} {'C++ (ms)':<12} {'Speedup':>8}")
    print("-" * 85)

    for rust_bench in rust_results["benchmarks"]:
        name = rust_bench["name"]
        cpp_bench = cpp_lookup.get(name)

        if cpp_bench is None:
            continue

        rust_time = rust_bench["mean_time"] * 1000
        cpp_time = cpp_bench["mean_time"] * 1000
        speedup = cpp_time / rust_time
        speedups.append(speedup)

        status = "✓" if speedup >= 0.95 else "⚠" if speedup >= 0.80 else "✗"

        print(
            f"{name:<50} {rust_time:>10.2f}  {cpp_time:>10.2f}  "
            f"{speedup:>7.2f}x {status}"
        )

    print("-" * 85)
    print(f"{'Mean speedup:':<50} {'':<12} {'':<12} {np.mean(speedups):>7.2f}x")
    print(f"{'Median speedup:':<50} {'':<12} {'':<12} {np.median(speedups):>7.2f}x")
    print(f"{'Min speedup:':<50} {'':<12} {'':<12} {np.min(speedups):>7.2f}x")
    print(f"{'Max speedup:':<50} {'':<12} {'':<12} {np.max(speedups):>7.2f}x")

    print("\n✓ = Rust ≥ 95% of C++ performance")
    print("⚠ = Rust 80-95% of C++ performance")
    print("✗ = Rust < 80% of C++ performance")

    if np.mean(speedups) >= 0.95:
        print("\n🎉 Rust implementation has comparable performance to C++/Cython!")
    elif np.mean(speedups) >= 0.80:
        print("\n⚠️  Rust implementation is slightly slower but still acceptable.")
    else:
        print("\n⚠️  Rust implementation needs performance optimization.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark tabmat implementations")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--compare", type=str, help="Compare with results from JSON file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations",
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--xlarge", action="store_true", help="Benchmark with 1M rows")
    parser.add_argument(
        "--large-narrow",
        action="store_true",
        help="Benchmark with 100K rows × 50 cols",
    )
    args = parser.parse_args()

    if args.large_narrow:
        matrix_sizes = {
            "large-narrow": {"n_rows": 100000, "n_cols": 50},
        }
    elif args.xlarge:
        matrix_sizes = {
            "xlarge": {"n_rows": 1000000, "n_cols": 100},
        }
    elif args.quick:
        matrix_sizes = {
            "small": {"n_rows": 1000, "n_cols": 50},
            "medium": {"n_rows": 10000, "n_cols": 100},
        }
    else:
        matrix_sizes = {
            "small": {"n_rows": 1000, "n_cols": 50},
            "medium": {"n_rows": 10000, "n_cols": 100},
            "large": {"n_rows": 100000, "n_cols": 200},
            "wide": {"n_rows": 10000, "n_cols": 1000},
            "xlarge": {"n_rows": 1000000, "n_cols": 100},
        }

    results = run_benchmarks(matrix_sizes, n_iterations=args.iterations)

    if args.save:
        output_path = Path(args.save)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\n✓ Results saved to {output_path}")

    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.exists():
            previous_results = json.loads(compare_path.read_text())
            compare_results(previous_results, results)
        else:
            print(f"\n⚠️  Comparison file not found: {compare_path}")

    print()


if __name__ == "__main__":
    main()
