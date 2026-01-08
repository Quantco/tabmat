# Rust Migration

This branch contains the initial implementation of tabmat's C++ extensions in Rust using PyO3.

## Status

✅ **Implemented**:
- Basic Rust project structure with Cargo.toml
- PyO3 bindings setup
- Dense matrix operations (sandwich, matvec, rmatvec)
- Sparse matrix sandwich product
- Categorical matrix sandwich product
- Maturin build system integration
- Backward compatibility layer

🚧 **In Progress**:
- SIMD optimizations (currently using basic Rayon parallelization)
- Cache-aware blocking optimizations
- Performance benchmarking vs C++ implementation

⏳ **TODO**:
- Add comprehensive tests
- Implement remaining matrix operations
- Optimize memory layout handling
- Add SIMD with packed_simd or std::simd
- Performance tuning to match C++ performance
- CI/CD updates for Rust toolchain

## Building

```bash
# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install in development mode
pixi run postinstall
# or manually:
maturin develop --release

# Run tests
pixi run test
```

## Architecture Changes

### Before (C++/Cython)
- `setup.py` with Mako templates
- Cython `.pyx` files calling C++ code
- Manual memory management with jemalloc
- xsimd for SIMD vectorization
- OpenMP for parallelization

### After (Rust/PyO3)
- `Cargo.toml` for dependencies
- Pure Rust implementation in `rust_src/`
- Automatic memory safety
- Rayon for parallelization
- Future: packed_simd or std::simd for SIMD

## Performance Considerations

The initial Rust implementation focuses on correctness and uses:
- Rayon for parallelization (similar to OpenMP)
- Basic blocked matrix multiplication
- Standard Rust iterators

Future optimizations will add:
- SIMD instructions via packed_simd
- Cache-aware blocking with tuning parameters
- Alignment optimizations
- Custom allocators if needed

## Compatibility

The Rust extensions are designed to be drop-in replacements for the C++ extensions:
- Same function signatures
- Same numpy array handling
- Backward compatibility through `rust_compat.py` wrapper

## Testing

Run the existing test suite to verify compatibility:

```bash
pixi run test tests/test_matrices.py
```

Run benchmarks to compare performance:

```bash
pixi run -e benchmark benchmark-run
```

## Migration Notes

Key differences from C++ implementation:

1. **Memory Management**: Rust's ownership system eliminates need for jemalloc
2. **Parallelization**: Rayon's `par_iter()` replaces OpenMP pragmas
3. **SIMD**: Currently basic; will add packed_simd for vectorization
4. **Build System**: Maturin simplifies build vs setuptools + Mako + Cython
5. **Type Safety**: Rust's type system catches more errors at compile time

## Next Steps

1. Run comprehensive benchmarks against C++ version
2. Implement SIMD optimizations if performance gaps exist
3. Add cache blocking optimizations
4. Update CI/CD pipelines
5. Update documentation
6. Gradual rollout and testing
