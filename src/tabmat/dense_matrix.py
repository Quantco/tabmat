import textwrap
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
    _check_indexer,
    check_matvec_dimensions,
    check_matvec_out_shape,
    check_transpose_matvec_out_shape,
    setup_restrictions,
)


class DenseMatrix(MatrixBase):
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

    def __init__(self, input_array, column_names=None, term_names=None):
        input_array = np.asarray(input_array)

        if input_array.ndim == 1:
            input_array = input_array.reshape(-1, 1)
        elif input_array.ndim > 2:
            raise ValueError("Input array must be 1- or 2-dimensional")

        self._array = np.asarray(input_array)
        width = self._array.shape[1]

        if column_names is not None:
            if len(column_names) != width:
                raise ValueError(
                    f"Expected {width} column names, got {len(column_names)}"
                )
            self._colnames = column_names
        else:
            self._colnames = [None] * width

        if term_names is not None:
            if len(term_names) != width:
                raise ValueError(f"Expected {width} term names, got {len(term_names)}")
            self._terms = term_names
        else:
            self._terms = self._colnames

    def __getitem__(self, key):
        row, col = _check_indexer(key)
        colnames = list(np.array(self.column_names)[col].ravel())
        terms = list(np.array(self.term_names)[col].ravel())

        return type(self)(
            self._array.__getitem__((row, col)), column_names=colnames, term_names=terms
        )

    __array_ufunc__ = None

    def __matmul__(self, other):
        return self._array.__matmul__(other)

    def __rmatmul__(self, other):
        return self._array.__rmatmul__(other)

    def __str__(self):
        return "{}x{} DenseMatrix:\n\n".format(*self.shape) + np.array_str(self._array)

    def __repr__(self):
        class_name = type(self).__name__
        array_str = f"{class_name}({np.array2string(self._array, separator=', ')})"
        return textwrap.indent(
            array_str,
            " " * (len(class_name) + 1),
            predicate=lambda line: not line.startswith(class_name),
        )

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

    def transpose(self):
        """Returns a view of the array with axes transposed."""  # noqa: D401
        return type(self)(self._array.T)

    T = property(transpose)

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """Copy of the array, cast to a specified type."""
        return type(self)(
            self._array.astype(dtype, order, casting, copy),
            column_names=self.column_names,
            term_names=self.term_names,
        )

    def getcol(self, i):
        """Return matrix column at specified index."""
        return type(self)(
            self._array[:, [i]],
            column_names=[self.column_names[i]],
            term_names=[self.term_names[i]],
        )

    def toarray(self):
        """Return array representation of matrix."""
        return self._array

    def unpack(self):
        """Return the underlying numpy.ndarray."""
        return self._array

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        """Perform a sandwich product: X.T @ diag(d) @ X."""
        d = np.asarray(d)
        rows, cols = setup_restrictions(self.shape, rows, cols)
        return dense_sandwich(self._array, d, rows, cols)

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
        sqrt_arg = transpose_square_dot_weights(self._array, weights) - col_means**2
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
        out: Optional[np.ndarray],
        transpose: bool,
    ):
        # Because the dense_rmatvec takes a row array and col array, it has
        # added overhead compared to a raw matrix vector product. So, when
        # we're not filtering at all, let's just use default numpy dot product.
        #
        # TODO: related to above, it could be nice to have a version that only
        # filters rows and a version that only filters columns. How do we do
        # this without an explosion of code?
        vec = np.asarray(vec)
        check_matvec_dimensions(self, vec, transpose=transpose)
        X = self._array.T if transpose else self._array

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
                res = fast_fnc(self._array, vec, rows, cols)
            elif vec.ndim == 2 and vec.shape[1] == 1:
                res = fast_fnc(self._array, vec[:, 0], rows, cols)[:, None]
            else:
                subset = self._array[np.ix_(rows, cols)]
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
        """Perform: self[rows, cols].T @ vec[rows]."""
        check_transpose_matvec_out_shape(self, out)
        return self._matvec_helper(vec, rows, cols, out, True)

    def matvec(
        self,
        vec: Union[np.ndarray, List],
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """Perform self[:, cols] @ other[cols]."""
        check_matvec_out_shape(self, out)
        return self._matvec_helper(vec, None, cols, out, False)

    def multiply(self, other):
        """Element-wise multiplication.

        This assumes that ``other`` is a vector of size ``self.shape[0]``.
        """
        if np.asanyarray(other).ndim == 1:
            return type(self)(
                self._array.__mul__(other[:, np.newaxis]),
                column_names=self.column_names,
                term_names=self.term_names,
            )
        return type(self)(
            self._array.__mul__(other),
            column_names=self.column_names,
            term_names=self.term_names,
        )

    def get_names(
        self,
        type: str = "column",
        missing_prefix: Optional[str] = None,
        indices: Optional[List[int]] = None,
    ) -> List[Optional[str]]:
        """Get column names.

        For columns that do not have a name, a default name is created using the
        followig pattern: ``"{missing_prefix}{start_index + i}"`` where ``i`` is
        the index of the column.

        Parameters
        ----------
        type: str {'column'|'term'}
            Whether to get column names or term names. The main difference is that
            a categorical submatrix is counted as a single term, whereas it is
            counted as multiple columns. Furthermore, matrices created from formulas
            have a difference between a column and term (c.f. ``formulaic`` docs).
        missing_prefix: Optional[str], default None
            Prefix to use for columns that do not have a name. If None, then no
            default name is created.
        indices
            The indices used for columns that do not have a name. If ``None``,
            then the indices are ``list(range(self.shape[1]))``.

        Returns
        -------
        List[Optional[str]]
            Column names.
        """
        if type == "column":
            names = np.array(self._colnames)
        elif type == "term":
            names = np.array(self._terms)
        else:
            raise ValueError(f"Type must be 'column' or 'term', got {type}")

        if indices is None:
            indices = list(range(len(self._colnames)))

        if missing_prefix is not None:
            default_names = np.array([f"{missing_prefix}{i}" for i in indices])
            names[names == None] = default_names[names == None]  # noqa: E711

        return list(names)

    def set_names(self, names: Union[str, List[Optional[str]]], type: str = "column"):
        """Set column names.

        Parameters
        ----------
        names: List[Optional[str]]
            Names to set.
        type: str {'column'|'term'}
            Whether to set column names or term names. The main difference is that
            a categorical submatrix is counted as a single term, whereas it is
            counted as multiple columns. Furthermore, matrices created from formulas
            have a difference between a column and term (c.f. ``formulaic`` docs).
        """
        if isinstance(names, str):
            names = [names]

        if len(names) != self.shape[1]:
            raise ValueError(f"Length of names must be {self.shape[1]}")

        if type == "column":
            self._colnames = names
        elif type == "term":
            self._terms = names
        else:
            raise ValueError(f"Type must be 'column' or 'term', got {type}")
