"""Kernels for :class:`tabmat.CategoricalMatrix`.

The implementation lives in the Rust extension module ``tabmat.ext._rust``
(see ``src/tabmat/ext/rust/categorical.rs``); this module only re-exports its
functions under their historical import path.
"""

from ._rust.categorical import (
    get_col_included,
    matvec_complex,
    matvec_fast,
    multiply_complex,
    sandwich_categorical_complex,
    sandwich_categorical_fast,
    subset_categorical_complex,
    transpose_matvec_complex,
    transpose_matvec_fast,
)

__all__ = [
    "get_col_included",
    "matvec_complex",
    "matvec_fast",
    "multiply_complex",
    "sandwich_categorical_complex",
    "sandwich_categorical_fast",
    "subset_categorical_complex",
    "transpose_matvec_complex",
    "transpose_matvec_fast",
]
