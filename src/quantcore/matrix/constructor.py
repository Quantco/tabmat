import warnings
from typing import List, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .categorical_matrix import CategoricalMatrix
from .dense_matrix import DenseMatrix
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .split_matrix import SplitMatrix, split_sparse_and_dense_parts


def from_pandas(
    df: pd.DataFrame,
    dtype: np.dtype = np.float64,
    sparse_threshold: float = 0.1,
    cat_threshold: int = 4,
    object_as_cat: bool = False,
) -> MatrixBase:
    """
    Transform a pandas.DataFrame into an efficient SplitMatrix

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame to be converted.
    dtype : np.dtype, default np.float64
        dtype of all sub-matrices of the resulting SplitMatrix.
    sparse_threshold : float, default 0.1
        Density threshold below which numerical columns will be stored in a sparse
        format.
    cat_threshold : int, default 4
        Number of levels of a categorical column under which the column will be stored
        as sparse one-hot-encoded columns instead of CategoricalMatrix
    object_as_cat : bool, default False
        If True, DataFrame columns stored as python objects will be treated as
        categorical columns.

    Returns
    -------
    SplitMatrix
    """
    if object_as_cat:
        for colname in df.select_dtypes("object"):
            df[colname] = df[colname].astype("category")

    matrices: List[Union[DenseMatrix, SparseMatrix, CategoricalMatrix]] = []
    indices: List[List[int]] = []

    dense_dfidx = []  # column index in original DataFrame
    dense_mxidx = []  # index in the new SplitMatrix
    sparse_dfidx = []  # column index in original DataFrame
    sparse_mxidx = []  # index in the new SplitMatrix
    ignored_cols = []

    mxcolidx = 0

    for dfcolidx, (colname, coldata) in enumerate(df.iteritems()):
        # categorical
        if isinstance(coldata.dtype, pd.CategoricalDtype):
            if len(coldata.cat.categories) < cat_threshold:
                (
                    X_dense_F,
                    X_sparse,
                    dense_indices,
                    sparse_indices,
                ) = split_sparse_and_dense_parts(
                    pd.get_dummies(coldata, prefix=colname, sparse=True),
                    threshold=sparse_threshold,
                )
                matrices.append(X_dense_F)
                indices.append(mxcolidx + dense_indices)
                matrices.append(X_sparse)
                indices.append(mxcolidx + sparse_indices)
                mxcolidx += len(dense_indices) + len(sparse_indices)
            else:
                cat = CategoricalMatrix(coldata, dtype=dtype)
                matrices.append(cat)
                indices.append(mxcolidx + np.arange(cat.shape[1]))
                mxcolidx += cat.shape[1]

        # All other numerical dtypes (needs to be after pd.SparseDtype)
        elif is_numeric_dtype(coldata):
            # check if we want to store as sparse
            if (coldata != 0).mean() <= sparse_threshold:
                if not isinstance(coldata.dtype, pd.SparseDtype):
                    sparse_dtype = pd.SparseDtype(coldata.dtype, fill_value=0)
                    df.iloc[:, dfcolidx] = coldata.astype(sparse_dtype)
                sparse_dfidx.append(dfcolidx)
                sparse_mxidx.append(mxcolidx)
                mxcolidx += 1
            else:
                dense_dfidx.append(dfcolidx)
                dense_mxidx.append(mxcolidx)
                mxcolidx += 1

        # dtype not handled yet
        else:
            ignored_cols.append((dfcolidx, colname))

    if len(ignored_cols) > 0:
        warnings.warn(
            f"Columns {ignored_cols} were ignored. Make sure they have a valid dtype."
        )
    if len(dense_dfidx) > 0:
        matrices.append(DenseMatrix(df.iloc[:, dense_dfidx].astype(dtype)))
        indices.append(dense_mxidx)
    if len(sparse_dfidx) > 0:
        matrices.append(
            SparseMatrix(df.iloc[:, sparse_dfidx].sparse.to_coo(), dtype=dtype)
        )
        indices.append(sparse_mxidx)

    if len(matrices) > 1:
        return SplitMatrix(matrices, indices)
    elif len(matrices) == 0:
        raise ValueError("DataFrame contained no valid column")
    else:
        return matrices[0]
