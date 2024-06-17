import sys
import warnings
from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
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
        - if 'fail', raise an error if there are missing values.
        - if 'zero', missing values will represent all-zero indicator columns.
        - if 'convert', missing values will be converted to the '(MISSING)' category.
    cat_missing_name: str, default '(MISSING)'
        Name of the category to which missing values will be converted if
        ``cat_missing_method='convert'``.

    Returns
    -------
    SplitMatrix
    """
    matrices: list[Union[DenseMatrix, SparseMatrix, CategoricalMatrix]] = []
    indices: list[list[int]] = []
    is_cat: list[bool] = []

    dense_columns = []  # column index in original DataFrame
    dense_indices = []  # index in the new SplitMatrix
    sparse_columns = []  # sparse columns to join together
    sparse_indices = []  # index in the new SplitMatrix
    ignored_cols = []

    mxcolidx = 0

    for colname, coldata in df.items():
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
            if len(cat.categories) < cat_threshold:
                (
                    X_dense_F,
                    X_sparse,
                    dense_idx,
                    sparse_idx,
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
                    indices.append(mxcolidx + dense_idx)
                    indices.append(mxcolidx + sparse_idx)
                    mxcolidx += len(dense_idx) + len(sparse_idx)
                elif cat_position == "end":
                    indices.append(dense_idx)
                    indices.append(sparse_idx)

            else:
                matrices.append(cat)
                is_cat.append(True)
                if cat_position == "expand":
                    indices.append(mxcolidx + np.arange(cat.shape[1]))
                    mxcolidx += cat.shape[1]
                elif cat_position == "end":
                    indices.append(np.arange(cat.shape[1]))
        elif is_numeric_dtype(coldata):
            if (coldata != 0).mean() <= sparse_threshold:
                sparse_columns.append(colname)
                sparse_indices.append(mxcolidx)
                mxcolidx += 1
            else:
                dense_columns.append(colname)
                dense_indices.append(mxcolidx)
                mxcolidx += 1

        else:
            ignored_cols.append(colname)

    if len(ignored_cols) > 0:
        warnings.warn(
            f"Columns {ignored_cols} were ignored. Make sure they have a valid dtype."
        )
    if dense_columns:
        matrices.append(
            DenseMatrix(
                df[dense_columns].astype(dtype),
                column_names=dense_columns,
                term_names=dense_columns,
            )
        )
        indices.append(dense_indices)
        is_cat.append(False)
    if sparse_columns:
        matrices.append(
            SparseMatrix(
                sps.coo_matrix(df[sparse_columns], dtype=dtype),
                dtype=dtype,
                column_names=sparse_columns,
                term_names=sparse_columns,
            )
        )
        indices.append(sparse_indices)
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


def from_polars(
    df: pl.DataFrame,
    dtype: np.dtype = np.float64,
    sparse_threshold: float = 0.1,
    cat_threshold: int = 4,
    cat_position: str = "expand",
    drop_first: bool = False,
    categorical_format: str = "{name}[{category}]",
    cat_missing_method: str = "fail",
    cat_missing_name: str = "(MISSING)",
) -> MatrixBase:
    """
    Transform a polars.DataFrame into an efficient SplitMatrix. For most users, this
    will be the primary way to construct tabmat objects from their data.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to be converted.
    dtype : np.dtype, default np.float64
        dtype of all sub-matrices of the resulting SplitMatrix.
    sparse_threshold : float, default 0.1
        Density threshold below which numerical columns will be stored in a sparse
        format.
    cat_threshold : int, default 4
        Number of levels of a categorical column under which the column will be stored
        as sparse one-hot-encoded columns instead of CategoricalMatrix
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
        - if 'fail', raise an error if there are missing values.
        - if 'zero', missing values will represent all-zero indicator columns.
        - if 'convert', missing values will be converted to the '(MISSING)' category.
    cat_missing_name: str, default '(MISSING)'
        Name of the category to which missing values will be converted if
        ``cat_missing_method='convert'``.

    Returns
    -------
    SplitMatrix
    """
    matrices: list[Union[DenseMatrix, SparseMatrix, CategoricalMatrix]] = []
    indices: list[list[int]] = []
    is_cat: list[bool] = []

    dense_columns = []  # column index in original DataFrame
    dense_indices = []  # index in the new SplitMatrix
    sparse_columns = []  # sparse columns to join together
    sparse_indices = []  # index in the new SplitMatrix
    ignored_cols = []

    mxcolidx = 0

    for coldata in df.iter_columns():
        if isinstance(coldata.dtype, (pl.Categorical, pl.Enum)):
            cat = CategoricalMatrix(
                coldata,
                drop_first=drop_first,
                dtype=dtype,
                column_name=coldata.name,
                term_name=coldata.name,
                column_name_format=categorical_format,
                cat_missing_method=cat_missing_method,
                cat_missing_name=cat_missing_name,
            )
            if len(cat.categories) < cat_threshold:
                (
                    X_dense_F,
                    X_sparse,
                    dense_idx,
                    sparse_idx,
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
                    indices.append(mxcolidx + dense_idx)
                    indices.append(mxcolidx + sparse_idx)
                    mxcolidx += len(dense_idx) + len(sparse_idx)
                elif cat_position == "end":
                    indices.append(dense_idx)
                    indices.append(sparse_idx)

            else:
                matrices.append(cat)
                is_cat.append(True)
                if cat_position == "expand":
                    indices.append(mxcolidx + np.arange(cat.shape[1]))
                    mxcolidx += cat.shape[1]
                elif cat_position == "end":
                    indices.append(np.arange(cat.shape[1]))
        elif coldata.dtype.is_numeric():
            if (coldata != 0).mean() <= sparse_threshold:
                sparse_columns.append(coldata.name)
                sparse_indices.append(mxcolidx)
                mxcolidx += 1
            else:
                dense_columns.append(coldata.name)
                dense_indices.append(mxcolidx)
                mxcolidx += 1

        else:
            ignored_cols.append(coldata.name)

    if len(ignored_cols) > 0:
        warnings.warn(
            f"Columns {ignored_cols} were ignored. Make sure they have a valid dtype."
        )
    if dense_columns:
        matrices.append(
            DenseMatrix(
                df[dense_columns].to_numpy().astype(dtype),
                column_names=dense_columns,
                term_names=dense_columns,
            )
        )
        indices.append(dense_indices)
        is_cat.append(False)
    if sparse_columns:
        matrices.append(
            SparseMatrix(
                sps.coo_matrix(df[sparse_columns], dtype=dtype),
                dtype=dtype,
                column_names=sparse_columns,
                term_names=sparse_columns,
            )
        )
        indices.append(sparse_indices)
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
    context: Optional[Union[int, Mapping[str, Any]]] = None,
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
        - if 'fail', raise an error if there are missing values.
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
    context: Optional[Union[int, Mapping[str, Any]]], default = None
        The context to add to the evaluation context of the formula with,
        e.g., custom transforms. If an integer, the context is taken from
        the stack frame of the caller at the given depth. Otherwise, a
        mapping from variable names to values is expected. By default,
        no context is added. Set ``context=0`` to make the calling scope
        available.
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
