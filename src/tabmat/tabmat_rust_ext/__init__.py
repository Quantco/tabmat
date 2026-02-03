from . import tabmat_rust_ext  # type: ignore[attr-defined]

__all__ = [
    "tabmat_rust_ext",
    "transpose_matvec",
    "matvec",
    "sandwich_categorical",
    "sandwich_cat_cat",
    "sandwich_cat_dense",
]

# Re-export functions for convenience
transpose_matvec = tabmat_rust_ext.transpose_matvec
matvec = tabmat_rust_ext.matvec
sandwich_categorical = tabmat_rust_ext.sandwich_categorical
sandwich_cat_cat = tabmat_rust_ext.sandwich_cat_cat
sandwich_cat_dense = tabmat_rust_ext.sandwich_cat_dense
