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
    cat_position: str = "expand",
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
    cat_position : str {'end'|'expand'}, default 'expand'
        Position of the categorical variable in the index. If "last", all the
        categoricals (including the ones that did not satisfy cat_threshold)
        will be placed at the end of the index list. If "expand", all the variables
        will remain in the same order.

    Returns
    -------
    SplitMatrix
    """
    matrices: List[Union[DenseMatrix, SparseMatrix, CategoricalMatrix]] = []
    indices: List[List[int]] = []
    is_cat: List[bool] = []

    dense_dfidx = []  # column index in original DataFrame
    dense_mxidx = []  # index in the new SplitMatrix
    sparse_dfcols = []  # column index in original DataFrame
    sparse_mxidx = []  # index in the new SplitMatrix
    ignored_cols = []

    mxcolidx = 0

    for dfcolidx, (colname, coldata) in enumerate(df.iteritems()):
        # categorical
        if object_as_cat and coldata.dtype == object:
            coldata = coldata.astype("category")
        if isinstance(coldata.dtype, pd.CategoricalDtype):
            if len(coldata.cat.categories) < cat_threshold:
                (
                    X_dense_F,
                    X_sparse,
                    dense_indices,
                    sparse_indices,
                ) = split_sparse_and_dense_parts(
                    pd.get_dummies(
                        coldata, prefix=colname, sparse=True, dtype=np.float64
                    )
                    .sparse.to_coo()
                    .tocsc(),
                    threshold=sparse_threshold,
                )
                matrices.append(X_dense_F)
                is_cat.append(True)
                matrices.append(X_sparse)
                is_cat.append(True)
                if cat_position == "expand":
                    indices.append(mxcolidx + dense_indices)
                    indices.append(mxcolidx + sparse_indices)
                    mxcolidx += len(dense_indices) + len(sparse_indices)
                elif cat_position == "end":
                    indices.append(dense_indices)
                    indices.append(sparse_indices)

            else:
                cat = CategoricalMatrix(coldata, dtype=dtype)
                matrices.append(cat)
                is_cat.append(True)
                if cat_position == "expand":
                    indices.append(mxcolidx + np.arange(cat.shape[1]))
                    mxcolidx += cat.shape[1]
                elif cat_position == "end":
                    indices.append(np.arange(cat.shape[1]))
        # All other numerical dtypes (needs to be after pd.SparseDtype)
        elif is_numeric_dtype(coldata):
            # check if we want to store as sparse
            if (coldata != 0).mean() <= sparse_threshold:
                if not isinstance(coldata.dtype, pd.SparseDtype):
                    sparse_dtype = pd.SparseDtype(coldata.dtype, fill_value=0)
                    sparse_dfcols.append(coldata.astype(sparse_dtype))
                else:
                    sparse_dfcols.append(coldata)
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
        is_cat.append(False)
    if len(sparse_dfcols) > 0:
        sparse_dict = {i: v for i, v in enumerate(sparse_dfcols)}
        full_sparse = pd.DataFrame(sparse_dict).sparse.to_coo()
        matrices.append(SparseMatrix(full_sparse, dtype=dtype))
        indices.append(sparse_mxidx)
        is_cat.append(False)

    if cat_position == "end":
        new_indices = []
        for mat_indices, is_cat_ in zip(indices, is_cat):
            if is_cat_:
                new_indices.append(np.asarray(mat_indices) + mxcolidx)
                mxcolidx += len(mat_indices)
            else:
                new_indices.append(mat_indices)
        indices = new_indices

    if len(matrices) > 1:
        return SplitMatrix(matrices, indices)
    elif len(matrices) == 0:
        raise ValueError("DataFrame contained no valid column")
    else:
        return matrices[0]
