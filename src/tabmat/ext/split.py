"""Kernels for :class:`tabmat.SplitMatrix` and the cross products of
:class:`tabmat.CategoricalMatrix`.

The implementation lives in the Rust extension module ``tabmat.ext.crabmat``
(see ``src/tabmat/ext/crabmat/split.rs`` for the bindings and
``src/tabmat/ext/crabmat/kernels/split.rs`` for the kernels); this module only
re-exports its functions under their historical import path.
"""

from .crabmat.split import (
    _sandwich_cat_cat_limited_rows_cols,
    is_sorted,
    sandwich_cat_cat,
    sandwich_cat_dense,
    split_col_subsets,
)

__all__ = [
    "_sandwich_cat_cat_limited_rows_cols",
    "is_sorted",
    "sandwich_cat_cat",
    "sandwich_cat_dense",
    "split_col_subsets",
]
