# tabmat Development Guide for AI Agents

## Project Overview

**tabmat** provides efficient matrix representations for tabular data in statistical/ML applications. It's designed for problems where data is partially sparse, partially dense, and partially categorical - common in economics, insurance, and actuarial modeling.

**Key use case**: Fast computation of operations like sandwich products (`X.T @ diag(d) @ X`), matrix-vector products, and column standardization while preserving sparsity.

## Architecture

### Matrix Types

- **`DenseMatrix`** (`dense_matrix.py`): Wraps numpy ndarray, adds `sandwich()`, `standardize()`, `getcol()`
- **`SparseMatrix`** (`sparse_matrix.py`): Wraps scipy.sparse.csc_matrix, optimized sandwich products
- **`CategoricalMatrix`** (`categorical_matrix.py`): One-hot encoded categories stored as integer indices (not sparse matrix). Exploits that all non-zeros are 1 and each row has exactly one non-zero.
- **`SplitMatrix`** (`split_matrix.py`): Heterogeneous matrix with dense, sparse, and categorical blocks. Delegates operations to appropriate submatrix types.
- **`StandardizedMatrix`** (`standardized_mat.py`): Wraps any matrix type, applies mean-centering and scaling without densifying. Stores shift/scale factors separately.

### Core Operations

All matrix types inherit from `MatrixBase` (`matrix_base.py`) and implement:
- `matvec(v)` and `rmatvec(v)`: Matrix-vector products
- `sandwich(d, rows=None)`: Compute `X[rows].T @ diag(d) @ X[rows]` efficiently
- `getcol(i)`: Extract column as `MatrixBase`
- `standardize(center, scale)`: Return `StandardizedMatrix`
- `transpose_matvec(v)`: Same as `rmatvec` for sklearn compatibility

**Critical insight**: `CategoricalMatrix.sandwich()` is O(n_categories²) instead of O(n_samples * n_categories) by exploiting categorical structure.

### Construction

**Primary constructor**: `from_pandas()` or `from_df()` in `constructor.py`:
```python
import tabmat as tm
mat = tm.from_pandas(df, 
    sparse_threshold=0.1,    # Columns with <10% nonzero → sparse
    cat_threshold=4,         # Categorical columns with ≤4 categories
    object_as_cat=False,     # Treat object dtype as categorical
    drop_first=False         # Drop first category level
)
```

**Returns `SplitMatrix`** with automatic partitioning into dense/sparse/categorical blocks based on thresholds.

### C++ Extensions (Target for Rust Translation)

Performance-critical code in `src/tabmat/ext/`:
- **Mako templates**: `*-tmpl.cpp` files are templated and rendered at build time to generate type-specialized code
  - `dense_helpers-tmpl.cpp`: Dense matrix sandwich products with BLIS/GotoBLAS-like tiling
  - `sparse_helpers-tmpl.cpp`: CSC sparse matrix operations
  - `cat_split_helpers-tmpl.cpp`: Categorical and split matrix operations
- **jemalloc integration**: Uses jemalloc allocator on Linux/macOS for better memory performance (`alloc.h`)
- **xsimd**: SIMD vectorization library (version pinned: `<11|>12.1` to avoid broken release)
- **OpenMP**: Parallel loops with `#pragma omp parallel for` and `prange` in Cython
- **Cython wrappers**: `*.pyx` files expose C++ functions to Python (dense.pyx, sparse.pyx, categorical.pyx, split.pyx)

**Key C++ implementation details**:
- Sandwich products use blocked matrix multiplication (IBLOCK × JBLOCK unrolling, typically 4×4)
- SIMD innermost loops with `xs::load_aligned()` and `xs::fma()` (fused multiply-add)
- Both C-contiguous and F-contiguous (column-major) layout support
- Tuning parameters: `thresh1d`, `kratio`, `innerblock` for cache optimization
- Template metaprogramming via Mako for loop unrolling and type specialization

Build system in `setup.py` renders Mako templates before compilation.

## Development Workflow

### Environment Setup (pixi)

```bash
# Install dependencies and build C++ extensions
pixi run postinstall

# Run tests
pixi run test

# Run specific test module with parallel execution
pixi run test -nauto tests/test_matrices.py

# Linting
pixi run pre-commit-run
```

### Available Environments

- **default**: Main development (dev + test features)
- **py310/py311/py312/py313**: Python version-specific testing
- **oldies**: Tests with oldest supported dependencies
- **nightly**: Tests with numpy/pandas/scipy nightlies
- **docs**: Documentation building with Sphinx
- **benchmark**: Matrix operation benchmarking
- **lint**: Pre-commit hooks (ruff, mypy, cython-lint)

### Building & Testing

**C++ Extensions**: Changes to `.cpp` or `-tmpl.cpp` files require rebuild:
```bash
pixi run postinstall  # Reruns Mako templates and recompiles
```

**Test Organization**:
- `tests/test_matrices.py`: Core matrix operation tests (842 lines, heavily parameterized)
- Use pytest fixtures for matrix creation (see `base_array()`, `dense_matrix_F()`, etc.)
- Test both C-contiguous and Fortran-contiguous layouts
- Mark memory-intensive tests with `@pytest.mark.high_memory`

**Doctests**: README examples tested in CI.

### Benchmarking

Benchmarks in `src/tabmat/benchmark/`:
```bash
pixi run -e benchmark benchmark-generate-matrices  # Create test matrices
pixi run -e benchmark benchmark-run                # Run benchmarks
pixi run -e benchmark benchmark-visualize          # Generate plots
```

Results saved to `benchmark/data/*.csv`, visualizations show tabmat vs numpy/scipy performance.

## Project-Specific Conventions

### Naming & Style

- Matrix classes use PascalCase: `DenseMatrix`, `SplitMatrix`
- Use `ruff` (see `pyproject.toml`): ignore E731, N802/N803/N806 for numpy naming
- Type hints: Use `numpy.typing` for array types, `MatrixBase` for matrices

### Memory Layout

- **Prefer Fortran order** (column-major) for matrices - matches scipy.sparse.csc_matrix and is cache-friendly for column operations
- C++ extensions expect Fortran-contiguous arrays in many cases
- `DenseMatrix` supports both but Fortran is default

### Categorical Handling

- Categories stored as integer codes (0-indexed)
- `drop_first=True` drops reference category, reduces dimensionality
- Missing category handling via `cat_missing_method='fail'` (default) or `'convert'`
- Category naming via `categorical_format="{name}[{category}]"`

### Formula Support (formulaic integration)

`TabmatMaterializer` in `formula.py` enables R-style formulas:
```python
from tabmat.formula import TabmatMaterializer
from formulaic import model_matrix

mat = model_matrix("y ~ C(region) + bs(age, 3)", data=df, 
                   materializer=TabmatMaterializer)
```

Returns `SplitMatrix` with categorical variables as `CategoricalMatrix`, splines as `DenseMatrix`, etc.

### Sparse Matrix Format

- Always use **CSC format** (compressed sparse column) for `SparseMatrix`
- Indices can be 32-bit or 64-bit integers (`int32` or `int64` in `indptr`/`indices`)
- Test both index types (see `sparse_matrix()` vs `sparse_matrix_64()` fixtures)

## Common Pitfalls

1. **Sandwich product dimensions**: `d` must have length matching `X.shape[0]` (or subset if `rows` specified). Result is `(n_cols, n_cols)`.

2. **Standardization state**: `StandardizedMatrix` stores original matrix reference. Changes to original affect standardized view. Clone if needed.

3. **SplitMatrix construction**: Submatrices must have same number of rows. Use `hstack()` or provide lists to `SplitMatrix()` constructor.

4. **Column extraction**: `mat.getcol(i)` returns `MatrixBase`, not numpy array. Use `.toarray()` or `.A` to get array.

5. **Categorical matrix shape**: `n_cols` equals number of categories (minus 1 if `drop_first=True`), NOT number of unique categories in data.

6. **xsimd version**: Must use `<11|>12.1` - version 11.x had critical bugs. This is enforced in `pyproject.toml`.

## Integration with glum

**tabmat was designed for glum's needs**. When modifying tabmat:
- Consider impacts on GLM fitting (sandwich products for Hessians, standardization for regularization)
- Maintain backward compatibility - glum depends on tabmat's API
- Performance matters: profile sandwich products and matvec operations
- Test with `glum`'s benchmarks (see `glum/src/glum_benchmarks/`)

To test tabmat changes with glum:
```bash
# In tabmat directory
pixi run postinstall

# In glum directory  
pixi run -e glum-tabmat postinstall-glum-tabmat  # Installs local tabmat + glum
pixi run -e glum-tabmat test
```

## narwhals Integration

Recent addition: narwhals support for polars/pandas compatibility. `from_df()` uses `@nw.narwhalify` decorator to accept both pandas and polars DataFrames. See `constructor.py` for implementation.

## Documentation

Docs built with Sphinx:
```bash
pixi run -e docs make-docs    # Build HTML
pixi run -e docs serve-docs   # Serve locally on :8000
```

API docs auto-generated via sphinxcontrib-apidoc from docstrings.

## C++ to Rust Translation Guide

### Migration Strategy

**Target**: Replace C++ extensions with Rust + PyO3, maintaining 100% API compatibility and performance parity.

**Rust advantages for this codebase**:
- Memory safety without runtime cost (eliminates jemalloc complexity)
- Built-in package manager (Cargo) vs complex C++ build (Mako templates, CMake)
- Better cross-platform support (simplified Windows builds)
- Modern SIMD via `std::simd` (nightly) or `packed_simd`
- Rayon for parallelism (simpler than OpenMP setup)

### Build System Changes

**Current (C++/Cython)**:
- `setup.py` with Mako template rendering
- Separate xsimd/jemalloc dependencies
- Platform-specific compiler flags in `pyproject.toml`

**Target (Rust/PyO3)**:
- `maturin` build backend (replaces setuptools)
- `Cargo.toml` for Rust dependencies
- PyO3 for Python bindings (replaces Cython)
- Example pyproject.toml:
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### Performance-Critical Functions to Translate

Priority order (based on performance impact):

1. **Dense sandwich** (`dense_helpers-tmpl.cpp` → Rust module):
   - Most CPU-intensive operation in GLM fitting
   - Current: BLIS-style blocked matrix mult with xsimd
   - Rust: Use `std::simd` or `packed_simd` + Rayon
   - Key: Maintain cache blocking parameters (IBLOCK=4, JBLOCK=4)

2. **Sparse sandwich** (`sparse_helpers-tmpl.cpp`):
   - CSC format operations
   - Rust: Consider `sprs` crate for CSC support
   - Challenge: xsimd's sparse patterns need careful translation

3. **Categorical operations** (`cat_split_helpers-tmpl.cpp`):
   - Integer index-based operations (simpler)
   - Good starting point for learning Rust/PyO3

### PyO3 vs Cython Mapping

| Cython Pattern | Rust/PyO3 Equivalent |
|----------------|---------------------|
| `cdef extern from "file.cpp"` | Direct Rust impl |
| `floating[:]` memoryview | `PyReadonlyArray1<f64>` |
| `nogil` block | Natural in Rust |
| `prange` (OpenMP) | `rayon::par_iter()` |
| `np.import_array()` | Auto-handled by PyO3 |

**Example translation** (dense.pyx → Rust):
```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

#[pyfunction]
fn dense_sandwich<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    rows: PyReadonlyArray1<i32>,
    cols: PyReadonlyArray1<i32>,
) -> &'py PyArray2<f64> {
    // Implementation here
}
```

### Testing Strategy

**Critical**: Run benchmarks before/after translation:
```bash
pixi run -e benchmark benchmark-run
```

Compare:
- Runtime (must be ≤5% slower initially)
- Memory usage (should improve without jemalloc)
- Correctness (all 842 tests in test_matrices.py must pass)

**Phased approach**:
1. Translate one matrix type (CategoricalMatrix - simplest)
2. Verify benchmarks match C++
3. Translate remaining types
4. Remove C++/Cython once all tests pass

### SIMD Considerations

**Current xsimd patterns**:
```cpp
auto Xsimd = xs::load_aligned(ptr);
accumsimd = xs::fma(Xtd, Xsimd, accumsimd);
F result = xs::reduce_add(accumsimd);
```

**Rust equivalent** (std::simd nightly):
```rust
use std::simd::{f64x4, Simd, SimdFloat};
let x: Simd<f64, 4> = Simd::from_slice(ptr);
accum = x.mul_add(y, accum);
let result: f64 = accum.reduce_sum();
```

**Portable alternative**: Use `packed_simd` or manual unrolling for stable Rust.

### Memory Alignment

**Current**: jemalloc's `je_aligned_alloc` (64-byte alignment for cache lines)

**Rust**: 
```rust
use std::alloc::{alloc, Layout};
let layout = Layout::from_size_align(size, 64).unwrap();
let ptr = unsafe { alloc(layout) };
```

Or use `aligned-vec` crate for simpler API.

## Release Process

1. Update `CHANGELOG.rst` with changes (use semantic versioning)
2. Open PR with changelog update + date
3. Create GitHub release with version tag - triggers conda-forge deployment
4. conda-forge feedstock at `../tabmat-feedstock` auto-updates via bot
