from typing import List, Optional, Union

import numpy as np
from scipy import sparse as sps

from .ext.sparse import (
    csc_rmatvec,
    csc_rmatvec_unrestricted,
    csr_dense_sandwich,
    csr_matvec,
    csr_matvec_unrestricted,
    sparse_sandwich,
    transpose_square_dot_weights,
)
from .matrix_base import MatrixBase
from .util import (
    _check_indexer,
    check_matvec_dimensions,
    check_matvec_out_shape,
    check_transpose_matvec_out_shape,
    set_up_rows_or_cols,
    setup_restrictions,
)


class SparseMatrix(MatrixBase):
    """
    A scipy.sparse csc matrix subclass that allows such objects to conform
    to the ``MatrixBase`` interface.

    SparseMatrix is instantiated in the same way as scipy.sparse.csc_matrix.
    """

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        self._array = sps.csc_matrix(arg1, shape, dtype, copy)

        self.idx_dtype = max(self._array.indices.dtype, self._array.indptr.dtype)
        if self._array.indices.dtype != self.idx_dtype:
            self._array.indices = self._array.indices.astype(self.idx_dtype)
        if self._array.indptr.dtype != self.idx_dtype:
            self._array.indptr = self._array.indptr.astype(self.idx_dtype)
        assert self.indices.dtype == self.idx_dtype

        if not self._array.has_sorted_indices:
            self._array.sort_indices()
        self._array_csr = None

    def __getitem__(self, key):
        return type(self)(self._array.__getitem__(_check_indexer(key)))

    def __matmul__(self, other):
        return self._array.__matmul__(other)

    def __rmatmul__(self, other):
        return self._array.__rmatmul__(other)

    __array_ufunc__ = None

    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self._array.shape

    @property
    def ndim(self):
        """Number of array dimensions."""  # noqa: D401
        return self._array.ndim

    @property
    def dtype(self):
        """Data-type of the array’s elements."""  # noqa: D401
        return self._array.dtype

    @property
    def indices(self):
        """Indices of the matrix."""  # noqa: D401
        return self._array.indices

    @property
    def indptr(self):
        """Indptr of the matrix."""  # noqa: D401
        return self._array.indptr

    @property
    def data(self):
        """Data of the matrix."""  # noqa: D401
        return self._array.data

    @property
    def array_csc(self):
        """Return the CSC representation of the matrix."""
        return self._array

    @property
    def array_csr(self):
        """Cache the CSR representation of the matrix."""
        if self._array_csr is None:
            self._array_csr = self._array.tocsr(copy=False)
            if self._array_csr.indices.dtype != self.idx_dtype:
                self._array_csr.indices = self._array_csr.indices.astype(self.idx_dtype)
            if self._array_csr.indptr.dtype != self.idx_dtype:
                self._array_csr.indptr = self._array_csr.indptr.astype(self.idx_dtype)

        return self._array_csr

    def tocsc(self, copy=False):
        """Return the matrix in CSC format."""
        return self._array.tocsc(copy=copy)

    def transpose(self):
        """Returns a view of the array with axes transposed."""  # noqa: D401
        return type(self)(self._array.T)

    T = property(transpose)

    def getcol(self, i):
        """Return matrix column at specified index."""
        return type(self)(self._array.getcol(i))

    def unpack(self):
        """Return the underlying scipy.sparse.csc_matrix."""
        return self._array

    def toarray(self):
        """Return a dense ndarray representation of the matrix."""
        return self._array.toarray()

    def dot(self, other):
        """Return the dot product as a scipy sparse matrix."""
        return self._array.dot(other)

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        """Perform a sandwich product: X.T @ diag(d) @ X."""
        d = np.asarray(d)
        if not self.dtype == d.dtype:
            raise TypeError(
                f"""self and d need to be of same dtype, either np.float64
                or np.float32. self is of type {self.dtype}, while d is of type
                {d.dtype}."""
            )

        rows, cols = setup_restrictions(self.shape, rows, cols, dtype=self.idx_dtype)
        return sparse_sandwich(self, self.array_csr, d, rows, cols)

    def _cross_sandwich(
        self,
        other: MatrixBase,
        d: np.ndarray,
        rows: np.ndarray,
        L_cols: Optional[np.ndarray] = None,
        R_cols: Optional[np.ndarray] = None,
    ):
        """Perform a sandwich product: X.T @ diag(d) @ Y."""
        from .categorical_matrix import CategoricalMatrix
        from .dense_matrix import DenseMatrix

        if isinstance(other, DenseMatrix):
            return self.sandwich_dense(other._array, d, rows, L_cols, R_cols)

        if isinstance(other, CategoricalMatrix):
            return other._cross_sandwich(self, d, rows, R_cols, L_cols).T
        raise TypeError

    def sandwich_dense(
        self,
        B: np.ndarray,
        d: np.ndarray,
        rows: np.ndarray,
        L_cols: np.ndarray,
        R_cols: np.ndarray,
    ) -> np.ndarray:
        """Perform a sandwich product: self.T @ diag(d) @ B."""
        if not hasattr(d, "dtype"):
            d = np.asarray(d)

        if self.dtype != d.dtype or B.dtype != d.dtype:
            raise TypeError(
                f"""self, B and d all need to be of same dtype, either
                np.float64 or np.float32. This matrix is of type {self.dtype},
                B is of type {B.dtype}, while d is of type {d.dtype}."""
            )
        if np.issubdtype(d.dtype, np.signedinteger):
            d = d.astype(float)

        rows, L_cols = setup_restrictions(self.shape, rows, L_cols)
        R_cols = set_up_rows_or_cols(R_cols, B.shape[1])
        return csr_dense_sandwich(self.array_csr, B, d, rows, L_cols, R_cols)

    def _matvec_helper(
        self,
        vec: Union[List, np.ndarray],
        rows: Optional[np.ndarray],
        cols: Optional[np.ndarray],
        out: Optional[np.ndarray],
        transpose: bool,
    ):
        vec = np.asarray(vec)
        check_matvec_dimensions(self, vec, transpose)

        unrestricted_rows = rows is None or len(rows) == self.shape[0]
        unrestricted_cols = cols is None or len(cols) == self.shape[1]
        if unrestricted_rows and unrestricted_cols and vec.ndim == 1:
            if transpose:
                return csc_rmatvec_unrestricted(self.array_csc, vec, out, self.indices)
            else:
                return csr_matvec_unrestricted(
                    self.array_csr, vec, out, self.array_csr.indices
                )

        matrix_matvec = lambda x, v: sps.csc_matrix.dot(x, v)
        if transpose:
            matrix_matvec = lambda x, v: sps.csr_matrix.dot(x.T, v)

        rows, cols = setup_restrictions(self.shape, rows, cols, dtype=self.idx_dtype)
        if transpose:
            fast_fnc = lambda v: csc_rmatvec(self.array_csc, v, rows, cols)
        else:
            fast_fnc = lambda v: csr_matvec(self.array_csr, v, rows, cols)
        if vec.ndim == 1:
            res = fast_fnc(vec)
        elif vec.ndim == 2 and vec.shape[1] == 1:
            res = fast_fnc(vec[:, 0])[:, None]
        else:
            res = matrix_matvec(
                self[np.ix_(rows, cols)]._array, vec[rows] if transpose else vec[cols]
            )
        if out is None:
            return res
        if transpose:
            out[cols] += res
        else:
            out[rows] += res
        return out

    def matvec(self, vec, cols: np.ndarray = None, out: np.ndarray = None):
        """Perform self[:, cols] @ other[cols]."""
        check_matvec_out_shape(self, out)
        return self._matvec_helper(vec, None, cols, out, False)

    def transpose_matvec(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """Perform: self[rows, cols].T @ vec[rows]."""
        check_transpose_matvec_out_shape(self, out)
        return self._matvec_helper(vec, rows, cols, out, True)

    def _get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        """Get standard deviations of columns."""
        sqrt_arg = (
            transpose_square_dot_weights(
                self._array.data,
                self._array.indices,
                self._array.indptr,
                weights,
                weights.dtype,
            )
            - col_means**2
        )
        # Minor floating point errors above can result in a very slightly
        # negative sqrt_arg (e.g. -5e-16). We just set those values equal to
        # zero.
        sqrt_arg[sqrt_arg < 0] = 0
        return np.sqrt(sqrt_arg)

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """Return SparseMatrix cast to new type."""
        return type(self)(self._array.astype(dtype, casting, copy))

    def multiply(self, other):
        """Element-wise multiplication.

        See ``scipy.sparse.csc_matrix.multiply``. The method is taken almost directly
        from the parent class except that ``other`` is assumed to be a vector of size
        ``self.shape[0]``.
        """
        if other.ndim == 1:
            return type(self)(self._array.multiply(other[:, np.newaxis]))
        return type(self)(self._array.multiply(other))
