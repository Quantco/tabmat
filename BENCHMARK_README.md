# Tabmat Implementation Benchmark: C++/Cython vs Rust

This directory contains benchmarks to compare the performance of the C++/Cython and Rust implementations of tabmat.

## Quick Start

### Option 1: Automated Comparison (Recommended)

Run the automated benchmark script that tests both implementations:

```bash
# Quick benchmark (~2 minutes)
./run_comparison_benchmark.sh --quick

# Full benchmark (~10-15 minutes, more comprehensive)
./run_comparison_benchmark.sh
```

This will:
1. Install and benchmark the C++/Cython version (from conda)
2. Build and benchmark the Rust version (from local source)
3. Display a comparison table with speedup ratios
4. Save results to `cpp_results.json` and `rust_results.json`

### Option 2: Manual Benchmarking

Benchmark the current implementation:

```bash
pixi run python benchmark_rust_vs_cpp.py --quick --iterations 10
```

Compare two implementations manually:

```bash
# 1. Benchmark C++/Cython version
pixi run pip uninstall -y tabmat
pixi install  # Installs conda tabmat
pixi run python benchmark_rust_vs_cpp.py --save cpp_results.json

# 2. Benchmark Rust version
pixi run postinstall  # Builds local Rust tabmat
pixi run python benchmark_rust_vs_cpp.py --save rust_results.json --compare cpp_results.json
```

## Benchmark Details

The benchmark tests the following operations across different matrix types and sizes:

### Matrix Types
- **DenseMatrix**: Standard dense matrices (Fortran-order)
- **SparseMatrix**: CSC sparse matrices (10% density)
- **CategoricalMatrix**: One-hot encoded categorical data (10 categories)
- **SplitMatrix**: Heterogeneous matrices with mixed dense/sparse/categorical blocks
- **StandardizedMatrix**: Mean-centered and scaled matrices

### Operations
- **sandwich**: `X.T @ diag(d) @ X` - critical for GLM Hessian computation
- **matvec**: `X @ v` - matrix-vector multiplication
- **sandwich_rows**: `X[rows].T @ diag(d[rows]) @ X[rows]` - sandwich with row subset

### Size Categories

**Quick mode** (`--quick`):
- Small: 1,000 × 50
- Medium: 10,000 × 100

**Full mode** (default):
- Small: 1,000 × 50
- Medium: 10,000 × 100
- Large: 100,000 × 200
- Wide: 10,000 × 1,000

## Command-Line Options

```bash
python benchmark_rust_vs_cpp.py [OPTIONS]

Options:
  --quick              Run quick benchmark with smaller matrices
  --iterations N       Number of iterations per benchmark (default: 10)
  --save FILE          Save results to JSON file
  --compare FILE       Compare current results with saved results
```

## Output Format

The benchmark produces:
1. **Timing table**: Mean and standard deviation for each operation
2. **Comparison table**: Side-by-side comparison with speedup ratios
3. **JSON files**: Detailed results for further analysis

### Example Output

```
==================================================================
Performance Comparison: Rust vs C++/Cython
==================================================================

Benchmark                                          Rust (ms)    C++ (ms)   Speedup
---------------------------------------------------------------------------------
small/dense/sandwich                                    0.44        0.42     0.95x ✓
small/dense/matvec                                      0.01        0.01     1.02x ✓
medium/split/sandwich                                 157.13      155.20     0.99x ✓
...
---------------------------------------------------------------------------------
Mean speedup:                                                                1.01x
Median speedup:                                                              1.00x

✓ = Rust ≥ 95% of C++ performance
⚠ = Rust 80-95% of C++ performance
✗ = Rust < 80% of C++ performance
```

## Performance Targets

The Rust implementation aims for:
- **Target**: ≥95% of C++/Cython performance (average)
- **Acceptable**: ≥80% of C++/Cython performance
- **Excellent**: ≥100% of C++/Cython performance

## Files

- `benchmark_rust_vs_cpp.py` - Main benchmark script
- `run_comparison_benchmark.sh` - Automated comparison script
- `cpp_results.json` - Saved C++/Cython benchmark results
- `rust_results.json` - Saved Rust benchmark results

## Notes

- First run may be slower due to compilation and warmup
- The benchmark uses 2 warmup iterations before timing
- Standard deviation indicates timing stability
- Results may vary based on system load and hardware
- For most accurate results, close other applications during benchmarking

## Troubleshooting

**"Unknown implementation" detected:**
- Make sure tabmat is installed: `pixi list | grep tabmat`
- Rebuild if needed: `pixi run postinstall`

**Import errors:**
- Check Python path: `pixi run python -c "import tabmat; print(tabmat.__file__)"`
- Verify dependencies: `pixi install`

**Slow benchmarks:**
- Use `--quick` for faster results
- Reduce iterations: `--iterations 3`
