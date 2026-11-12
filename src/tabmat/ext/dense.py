"""Kernels for :class:`tabmat.DenseMatrix`.

The implementation lives in the Rust extension module ``tabmat.ext._rust``
(see ``src/tabmat/ext/rust/dense.rs`` for the bindings and
``src/tabmat/ext/rust/kernels/dense.rs`` for the kernels); this module only
re-exports its functions under their historical import path.
"""

from ._rust.dense import (
    dense_matvec,
    dense_rmatvec,
    dense_sandwich,
    transpose_square_dot_weights,
)

__all__ = [
    "dense_matvec",
    "dense_rmatvec",
    "dense_sandwich",
    "transpose_square_dot_weights",
]
