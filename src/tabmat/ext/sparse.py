"""Kernels for :class:`tabmat.SparseMatrix`.

The implementation lives in the Rust extension module ``tabmat.ext.crabmat``
(see ``src/tabmat/ext/crabmat/sparse.rs`` for the bindings and
``src/tabmat/ext/crabmat/kernels/sparse.rs`` for the kernels); this module only
re-exports its functions under their historical import path.
"""

from .crabmat.sparse import (
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
