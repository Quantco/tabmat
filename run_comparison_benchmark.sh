#!/bin/bash
# Script to compare C++/Cython vs Rust implementations
#
# Usage: ./run_comparison_benchmark.sh [--quick]

set -e

# Workaround for OpenMP library conflict
export KMP_DUPLICATE_LIB_OK=TRUE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

QUICK_FLAG=""
if [ "$1" = "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "Running quick benchmark..."
else
    echo "Running full benchmark (use --quick for faster results)..."
fi

echo "================================================================"
echo "Tabmat Implementation Comparison: C++/Cython vs Rust"
echo "================================================================"

# Step 1: Install C++/Cython version and benchmark
echo ""
echo "Step 1: Benchmarking C++/Cython implementation..."
echo "----------------------------------------------------------------"
pixi run pip uninstall -y tabmat 2>/dev/null || true
pixi install
pixi run pip install tabmat
echo ""
pixi run python benchmark_rust_vs_cpp.py $QUICK_FLAG --save cpp_results.json

# Step 2: Install Rust version and benchmark
echo ""
echo "Step 2: Benchmarking Rust implementation..."
echo "----------------------------------------------------------------"
pixi run postinstall
echo ""
pixi run python benchmark_rust_vs_cpp.py $QUICK_FLAG --save rust_results.json --compare cpp_results.json

echo ""
echo "================================================================"
echo "Benchmark complete!"
echo "Results saved to cpp_results.json and rust_results.json"
echo "================================================================"
