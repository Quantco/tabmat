from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sps

from .dense_matrix import DenseMatrix
from .sparse_matrix import SparseMatrix


def _split_sparse_and_dense_parts(
    arg1: sps.csc_matrix,
    threshold: float = 0.1,
    column_names: Optional[Sequence[Optional[str]]] = None,
    term_names: Optional[Sequence[Optional[str]]] = None,
) -> Tuple[DenseMatrix, SparseMatrix, np.ndarray, np.ndarray]:
    """
    Split matrix.

    Return the dense and sparse parts of a matrix and the corresponding indices
    for each at the provided threshold.
    """
    if not isinstance(arg1, sps.csc_matrix):
        raise TypeError(
            f"X must be of type scipy.sparse.csc_matrix or matrix.SparseMatrix,"
            f"not {type(arg1)}"
        )
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    densities = np.diff(arg1.indptr) / arg1.shape[0]
    dense_indices = np.where(densities > threshold)[0]
    sparse_indices = np.setdiff1d(np.arange(densities.shape[0]), dense_indices)

    if column_names is None:
        column_names = [None] * arg1.shape[1]
    if term_names is None:
        term_names = column_names

    X_dense_F = DenseMatrix(
        np.asfortranarray(arg1[:, dense_indices].toarray()),
        column_names=[column_names[i] for i in dense_indices],
        term_names=[term_names[i] for i in dense_indices],
    )
    X_sparse = SparseMatrix(
        arg1[:, sparse_indices],
        column_names=[column_names[i] for i in sparse_indices],
        term_names=[term_names[i] for i in sparse_indices],
    )
    return X_dense_F, X_sparse, dense_indices, sparse_indices
