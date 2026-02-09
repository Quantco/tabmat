from . import tabmat_rust_ext  # type: ignore[attr-defined]

__all__ = [
    "tabmat_rust_ext",
    # Categorical operations
    "transpose_matvec",
    "matvec",
    "sandwich_categorical",
    "sandwich_cat_cat",
    "sandwich_cat_dense",
    # Sparse operations
    "csr_matvec_unrestricted",
    "csr_matvec",
    "csc_rmatvec_unrestricted",
    "csc_rmatvec",
    "sparse_sandwich",
    "csr_dense_sandwich",
    "transpose_square_dot_weights",
]

# Re-export categorical functions
transpose_matvec = tabmat_rust_ext.transpose_matvec
matvec = tabmat_rust_ext.matvec
sandwich_categorical = tabmat_rust_ext.sandwich_categorical
sandwich_cat_cat = tabmat_rust_ext.sandwich_cat_cat
sandwich_cat_dense = tabmat_rust_ext.sandwich_cat_dense

# Re-export sparse functions
csr_matvec_unrestricted = tabmat_rust_ext.csr_matvec_unrestricted
csr_matvec = tabmat_rust_ext.csr_matvec
csc_rmatvec_unrestricted = tabmat_rust_ext.csc_rmatvec_unrestricted
csc_rmatvec = tabmat_rust_ext.csc_rmatvec
sparse_sandwich = tabmat_rust_ext.sparse_sandwich
csr_dense_sandwich = tabmat_rust_ext.csr_dense_sandwich
transpose_square_dot_weights = tabmat_rust_ext.transpose_square_dot_weights
