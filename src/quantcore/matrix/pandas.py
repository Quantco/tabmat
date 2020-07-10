import warnings

import pandas as pd
import scipy.sparse as sps

from .categorical_matrix import CategoricalMatrix
from .matrix_base import MatrixBase
from .split_matrix import SplitMatrix, csc_to_split


def from_pandas(
    df: pd.DataFrame,
    sparse_threshold: float = 0.1,
    cat_threshold: int = 4,
    object_as_cat: bool = False,
) -> MatrixBase:
    """
    TODO:
     - docstring
     - tests
     - efficiency
     - consider changing filename
    """
    if object_as_cat:
        for colname in df.select_dtypes("object"):
            df[colname] = df[colname].astype("category")
    else:
        if not df.select_dtypes(include=object).empty:
            warnings.warn("DataFrame contains columns with object dtypes. Ignoring")

    categorical_component = df.select_dtypes(include=pd.CategoricalDtype)
    X_cat = []
    for colname in categorical_component:
        X_cat.append(CategoricalMatrix(categorical_component[colname]))

    numerical_component = df.select_dtypes(include="number")
    X_noncat = csc_to_split(sps.csc_matrix(numerical_component))

    return SplitMatrix([*X_noncat.matrices, *X_cat])
