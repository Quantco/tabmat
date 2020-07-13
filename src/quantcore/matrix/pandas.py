import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .categorical_matrix import CategoricalMatrix
from .dense_matrix import DenseMatrix
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .split_matrix import SplitMatrix


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

    matrices = []
    sparse_ohe_comp = []
    sparse_idx = []
    dense_idx = []
    ignored_cols = []
    for colidx, (colname, coldata) in enumerate(df.iteritems()):
        # categorical
        if isinstance(coldata.dtype, pd.CategoricalDtype):
            if len(coldata.cat.categories) < cat_threshold:
                sparse_ohe_comp.append(
                    pd.get_dummies(coldata, prefix=colname, sparse=True)
                )
            else:
                matrices.append(CategoricalMatrix(coldata, dtype=dtype))

        # sparse data, keep in sparse format even if density is larger than threshold
        elif isinstance(coldata.dtype, pd.SparseDtype):
            sparse_idx.append(colidx)

        # All other numerical dtypes (needs to be after pd.SparseDtype)
        elif is_numeric_dtype(coldata):
            # check if we want to store as sparse
            if (coldata != 0).mean() <= sparse_threshold:
                sparse_dtype = pd.SparseDtype(coldata.dtype, fill_value=0)
                df.iloc[:, colidx] = df.iloc[:, colidx].astype(sparse_dtype)
                sparse_idx.append(colidx)
            else:
                dense_idx.append(colidx)

        # dtype not handled yet
        else:
            ignored_cols.append((colidx, colname))

    if len(ignored_cols) > 0:
        warnings.warn(
            f"Columns {ignored_cols} were ignored. Make sure they have a valid dtype."
        )
    if len(dense_idx) > 0:
        dense_comp = DenseMatrix(df.iloc[:, dense_idx].astype(dtype))
        matrices.append(dense_comp)
    if len(sparse_idx) > 0:
        sparse_comp = SparseMatrix(df.iloc[:, sparse_idx].sparse.to_coo(), dtype=dtype)
        matrices.append(sparse_comp)
    if len(sparse_ohe_comp) > 0:
        sparse_ohe_comp = SparseMatrix(
            pd.concat(sparse_ohe_comp, axis=1).sparse.to_coo(), dtype=dtype
        )
        matrices.append(sparse_ohe_comp)

    if len(matrices) > 1:
        return SplitMatrix(matrices)
    else:
        return matrices[0]
