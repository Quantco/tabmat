"""Kernels for :class:`tabmat.CategoricalMatrix`.

The implementation lives in the Rust extension module ``tabmat.ext._rust``
(see ``src/tabmat/ext/rust/categorical.rs`` for the bindings and
``src/tabmat/ext/rust/kernels/categorical.rs`` for the kernels); this module
only re-exports its functions under their historical import path.
"""

from ._rust.categorical import (
    get_col_included,
    matvec,
    multiply,
    sandwich_categorical,
    subset_categorical,
    transpose_matvec,
)

__all__ = [
    "get_col_included",
    "matvec",
    "multiply",
    "sandwich_categorical",
    "subset_categorical",
    "transpose_matvec",
]
