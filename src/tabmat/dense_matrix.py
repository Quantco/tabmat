from typing import List, Optional, Union

import numpy as np

from .ext.dense import (
    dense_matvec,
    dense_rmatvec,
    dense_sandwich,
    transpose_square_dot_weights,
)
from .matrix_base import MatrixBase
from .util import (
    check_matvec_out_shape,
    check_transpose_matvec_out_shape,
    setup_restrictions,
)


class DenseMatrix(np.ndarray, MatrixBase):
    """
    A ``numpy.ndarray`` subclass with several additional functions that allow
    it to share the MatrixBase API with SparseMatrix and CategoricalMatrix.

    In particular, we have added:

    - The ``sandwich`` product
    - ``getcol`` to support the same interface as SparseMatrix for retrieving a
      single column
    - ``toarray``
    - ``matvec``

    """

    def __new__(cls, input_array):  # noqa
        """
        Details of how to subclass np.ndarray are explained here:

        https://docs.scipy.org/doc/numpy/user/basics.subclassing.html\
            #slightly-more-realistic-example-attribute-added-to-existing-array
        """
        obj = np.asarray(input_array).view(cls)
        if not np.issubdtype(obj.dtype, np.floating):
            raise NotImplementedError("DenseMatrix is only implemented for float data")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def getcol(self, i):
        """Return matrix column at specified index."""
        return self[:, [i]]

    def toarray(self):
        """Return array representation of matrix."""
        return np.asarray(self)

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        """Perform a sandwich product: X.T @ diag(d) @ X."""
        d = np.asarray(d)
        rows, cols = setup_restrictions(self.shape, rows, cols)
        return dense_sandwich(self, d, rows, cols)

    def _cross_sandwich(
        self,
        other: MatrixBase,
        d: np.ndarray,
        rows: Optional[np.ndarray] = None,
        L_cols: Optional[np.ndarray] = None,
        R_cols: Optional[np.ndarray] = None,
    ):
        from .categorical_matrix import CategoricalMatrix
        from .sparse_matrix import SparseMatrix

        if isinstance(other, SparseMatrix) or isinstance(other, CategoricalMatrix):
            return other._cross_sandwich(self, d, rows, R_cols, L_cols).T
        raise TypeError

    def _get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        """Get standard deviations of columns."""
        sqrt_arg = transpose_square_dot_weights(self, weights) - col_means ** 2
        # Minor floating point errors above can result in a very slightly
        # negative sqrt_arg (e.g. -5e-16). We just set those values equal to
        # zero.
        sqrt_arg[sqrt_arg < 0] = 0
        return np.sqrt(sqrt_arg)

    def _matvec_helper(
        self,
        vec: Union[List, np.ndarray],
        rows: Optional[np.ndarray],
        cols: Optional[np.ndarray],
        out: Optional[Union[np.ndarray]],
        transpose: bool,
    ):
        # Because the dense_rmatvec takes a row array and col array, it has
        # added overhead compared to a raw matrix vector product. So, when
        # we're not filtering at all, let's just use default numpy dot product.
        #
        # TODO: related to above, it could be nice to have a version that only
        # filters rows and a version that only filters columns. How do we do
        # this without an explosion of code?
        X = self.T if transpose else self
        vec = np.asarray(vec)

        # NOTE: We assume that rows and cols are unique
        unrestricted_rows = rows is None or len(rows) == self.shape[0]
        unrestricted_cols = cols is None or len(cols) == self.shape[1]

        if unrestricted_rows and unrestricted_cols:
            if out is None:
                out = X.dot(vec)
            else:
                out += X.dot(vec)
            return out
        else:
            rows, cols = setup_restrictions(self.shape, rows, cols)
            # TODO: should take 'out' parameter
            fast_fnc = dense_rmatvec if transpose else dense_matvec
            if vec.ndim == 1:
                res = fast_fnc(self, vec, rows, cols)
            elif vec.ndim == 2 and vec.shape[1] == 1:
                res = fast_fnc(self, vec[:, 0], rows, cols)[:, None]
            else:
                subset = self[np.ix_(rows, cols)]
                res = subset.T.dot(vec[rows]) if transpose else subset.dot(vec[cols])
            if out is None:
                return res
            if transpose:
                out[cols] += res
            else:
                # Note that currently 'rows' will always be all rows
                out[rows] += res
            return out

    def transpose_matvec(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """Perform: self[rows, cols].T @ vec."""
        check_transpose_matvec_out_shape(self, out)
        return self._matvec_helper(vec, rows, cols, out, True)

    def matvec(
        self,
        vec: Union[np.ndarray, List],
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """Perform self[:, cols] @ other."""
        check_matvec_out_shape(self, out)
        return self._matvec_helper(vec, None, cols, out, False)
