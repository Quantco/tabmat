"""
Compatibility wrapper for Rust extensions.

This module provides backward compatibility by wrapping Rust implementations
with the same API as the old Cython extensions.
"""

try:
    from .tabmat_ext import (
        categorical_sandwich as _rust_categorical_sandwich,
    )
    from .tabmat_ext import (
        dense_matvec as _rust_dense_matvec,
    )
    from .tabmat_ext import (
        dense_rmatvec as _rust_dense_rmatvec,
    )
    from .tabmat_ext import (
        dense_sandwich as _rust_dense_sandwich,
    )
    from .tabmat_ext import (
        sparse_sandwich as _rust_sparse_sandwich,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    # Fall back to old Cython extensions if available
    try:
        from .categorical import categorical_sandwich as _rust_categorical_sandwich
        from .dense import (
            dense_matvec as _rust_dense_matvec,
        )
        from .dense import (
            dense_rmatvec as _rust_dense_rmatvec,
        )
        from .dense import (
            dense_sandwich as _rust_dense_sandwich,
        )
        from .sparse import sparse_sandwich as _rust_sparse_sandwich
    except ImportError:
        pass

# Export with original names
dense_sandwich = _rust_dense_sandwich
dense_rmatvec = _rust_dense_rmatvec
dense_matvec = _rust_dense_matvec
sparse_sandwich = _rust_sparse_sandwich
categorical_sandwich = _rust_categorical_sandwich

__all__ = [
    "dense_sandwich",
    "dense_rmatvec",
    "dense_matvec",
    "sparse_sandwich",
    "categorical_sandwich",
    "RUST_AVAILABLE",
]
