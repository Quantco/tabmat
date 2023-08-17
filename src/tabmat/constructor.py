import sys
import warnings
from typing import Any, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula, ModelSpec
from formulaic.materializers.types import NAAction
from formulaic.parser import DefaultFormulaParser
from formulaic.utils.layered_mapping import LayeredMapping
from pandas.api.types import is_numeric_dtype
from scipy import sparse as sps

from .categorical_matrix import CategoricalMatrix
from .constructor_util import _split_sparse_and_dense_parts
from .dense_matrix import DenseMatrix
from .formula import TabmatMaterializer
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .split_matrix import SplitMatrix


def from_pandas(
    df: pd.DataFrame,
    dtype: np.dtype = np.float64,
    sparse_threshold: float = 0.1,
    cat_threshold: int = 4,
    object_as_cat: bool = False,
    cat_position: str = "expand",
    drop_first: bool = False,
    categorical_format: str = "{name}[{category}]",
    cat_missing_method: str = "fail",
    cat_missing_name: str = "(MISSING)",
) -> MatrixBase:
    """
    Transform a pandas.DataFrame into an efficient SplitMatrix. For most users, this
    will be the primary way to construct tabmat objects from their data.

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
    drop_first : bool, default False
        If true, categoricals variables will have their first category dropped.
        This allows multiple categorical variables to be included in an
        unregularized model. If False, all categories are included.
    cat_missing_method: str {'fail'|'zero'|'convert'}, default 'fail'
        How to handle missing values in categorical columns:
        - if 'fail', raise an error if there are missing values
        - if 'zero', missing values will represent all-zero indicator columns.
        - if 'convert', missing values will be converted to the '(MISSING)' category.
    cat_missing_name: str, default '(MISSING)'
        Name of the category to which missing values will be converted if
        ``cat_missing_method='convert'``.

    Returns
    -------
    SplitMatrix
    """
    matrices: List[Union[DenseMatrix, SparseMatrix, CategoricalMatrix]] = []
    indices: List[List[int]] = []
    is_cat: List[bool] = []

    dense_dfidx = []  # column index in original DataFrame
    dense_mxidx = []  # index in the new SplitMatrix
    sparse_dfcols = []  # sparse columns to join together
    sparse_mxidx = []  # index in the new SplitMatrix
    ignored_cols = []

    mxcolidx = 0

    for dfcolidx, (colname, coldata) in enumerate(df.items()):
        # categorical
        if object_as_cat and coldata.dtype == object:
            coldata = coldata.astype("category")
        if isinstance(coldata.dtype, pd.CategoricalDtype):
            cat = CategoricalMatrix(
                coldata,
                drop_first=drop_first,
                dtype=dtype,
                column_name=colname,
                term_name=colname,
                column_name_format=categorical_format,
                cat_missing_method=cat_missing_method,
                cat_missing_name=cat_missing_name,
            )
            if len(coldata.cat.categories) < cat_threshold:
                (
                    X_dense_F,
                    X_sparse,
                    dense_indices,
                    sparse_indices,
                ) = _split_sparse_and_dense_parts(
                    sps.csc_matrix(cat.tocsr(), dtype=dtype),
                    threshold=sparse_threshold,
                    column_names=cat.get_names("column"),
                    term_names=cat.get_names("term"),
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
        matrices.append(
            DenseMatrix(
                df.iloc[:, dense_dfidx].astype(dtype),
                column_names=df.columns[dense_dfidx],
                term_names=df.columns[dense_dfidx],
            )
        )
        indices.append(dense_mxidx)
        is_cat.append(False)
    if len(sparse_dfcols) > 0:
        sparse_dict = {i: v for i, v in enumerate(sparse_dfcols)}
        full_sparse = pd.DataFrame(sparse_dict).sparse.to_coo()
        matrices.append(
            SparseMatrix(
                full_sparse,
                dtype=dtype,
                column_names=[col.name for col in sparse_dfcols],
                term_names=[col.name for col in sparse_dfcols],
            )
        )
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


def from_csc(mat: sps.csc_matrix, threshold=0.1, column_names=None, term_names=None):
    """
    Convert a CSC-format sparse matrix into a ``SplitMatrix``.

    The ``threshold`` parameter specifies the density below which a column is
    treated as sparse.
    """
    dense, sparse, dense_idx, sparse_idx = _split_sparse_and_dense_parts(mat, threshold)
    return SplitMatrix([dense, sparse], [dense_idx, sparse_idx])


def from_formula(
    formula: Union[str, Formula],
    data: pd.DataFrame,
    ensure_full_rank: bool = False,
    na_action: Union[str, NAAction] = NAAction.IGNORE,
    dtype: np.dtype = np.float64,
    sparse_threshold: float = 0.1,
    cat_threshold: int = 4,
    interaction_separator: str = ":",
    categorical_format: str = "{name}[{category}]",
    cat_missing_method: str = "fail",
    cat_missing_name: str = "(MISSING)",
    intercept_name: str = "Intercept",
    include_intercept: bool = False,
    add_column_for_intercept: bool = True,
    context: Optional[Union[int, Mapping[str, Any]]] = 0,
) -> SplitMatrix:
    """
    Transform a pandas data frame to a SplitMatrix using a Wilkinson formula.

    Parameters
    ----------
    formula: str
        A formula accepted by formulaic.
    data: pd.DataFrame
        pandas data frame to be converted.
    ensure_full_rank: bool, default False
        If True, ensure that the matrix has full structural rank by categories.
    na_action: Union[str, NAAction], default NAAction.IGNORE
        How to handle missing values. Can be one of "drop", "ignore", "raise".
    dtype: np.dtype, default np.float64
        The dtype of the resulting matrix.
    sparse_threshold: float, default 0.1
        The density below which a column is treated as sparse.
    cat_threshold: int, default 4
        The number of categories below which a categorical column is one-hot
        encoded. This is only checked after interactions have been applied.
    interaction_separator: str, default ":"
        The separator between the names of interacted variables.
    categorical_format: str, default "{name}[T.{category}]"
        The format string used to generate the names of categorical variables.
        Has to include the placeholders ``{name}`` and ``{category}``.
    cat_missing_method: str {'fail'|'zero'|'convert'}, default 'fail'
        How to handle missing values in categorical columns:
        - if 'fail', raise an error if there are missing values
        - if 'zero', missing values will represent all-zero indicator columns.
        - if 'convert', missing values will be converted to the '(MISSING)' category.
    cat_missing_name: str, default '(MISSING)'
        Name of the category to which missing values will be converted if
        ``cat_missing_method='convert'``.
    intercept_name: str, default "Intercept"
        The name of the intercept column.
    include_intercept: bool, default False
        Whether to include an intercept term if the formula does not
        include (``+ 1``) or exclude (``+ 0`` or ``- 1``) it explicitly.
    add_column_for_intercept: bool, default = True
        Whether to add a column of ones for the intercept, or just
        have a term without a corresponding column. For advanced use only.
    context: Union[int, Mapping[str, Any]], default 0
        The context to use for evaluating the formula. If an integer, the
        context is taken from the stack frame of the caller at the given
        depth. If None, the context is taken from the stack frame of the
        caller at depth 1. If a dict, it is used as the context directly.
    """
    if isinstance(context, int):
        if hasattr(sys, "_getframe"):
            frame = sys._getframe(context + 1)
            context = LayeredMapping(frame.f_locals, frame.f_globals)
        else:
            context = None
    spec = ModelSpec(
        formula=Formula(
            formula, _parser=DefaultFormulaParser(include_intercept=include_intercept)
        ),
        ensure_full_rank=ensure_full_rank,
        na_action=na_action,
    )
    materializer = TabmatMaterializer(
        data,
        context=context,
        interaction_separator=interaction_separator,
        categorical_format=categorical_format,
        intercept_name=intercept_name,
        dtype=dtype,
        sparse_threshold=sparse_threshold,
        cat_threshold=cat_threshold,
        add_column_for_intercept=add_column_for_intercept,
        cat_missing_method=cat_missing_method,
        cat_missing_name=cat_missing_name,
    )
    result = materializer.get_model_matrix(spec)

    term_names = np.zeros(len(result.term_names), dtype="object")
    for term, indices in result.model_spec.term_indices.items():
        term_names[indices] = str(term)
    result.term_names = term_names.tolist()

    return result
