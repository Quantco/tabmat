import importlib.metadata

from .categorical_matrix import CategoricalMatrix
from .constructor import from_csc, from_pandas
from .dense_matrix import DenseMatrix
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .split_matrix import SplitMatrix
from .standardized_mat import StandardizedMatrix

try:
    __version__ = importlib.metadata.distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = [
    "DenseMatrix",
    "MatrixBase",
    "StandardizedMatrix",
    "SparseMatrix",
    "SplitMatrix",
    "CategoricalMatrix",
    "from_csc",
    "from_pandas",
]
