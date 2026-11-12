"""Kernels for :class:`tabmat.SparseMatrix`.

The implementation lives in the Rust extension module ``tabmat.ext._rust``
(see ``src/tabmat/ext/rust/sparse.rs`` for the bindings and
``src/tabmat/ext/rust/kernels/sparse.rs`` for the kernels); this module only
re-exports its functions under their historical import path.
"""

from ._rust.sparse import (
    csc_rmatvec,
    csc_rmatvec_unrestricted,
    csr_dense_sandwich,
    csr_matvec,
    csr_matvec_unrestricted,
    sparse_sandwich,
    transpose_square_dot_weights,
)

__all__ = [
    "csc_rmatvec",
    "csc_rmatvec_unrestricted",
    "csr_dense_sandwich",
    "csr_matvec",
    "csr_matvec_unrestricted",
    "sparse_sandwich",
    "transpose_square_dot_weights",
]
