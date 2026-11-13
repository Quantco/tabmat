import textwrap
import warnings

import numpy as np

from .crabmat.dense import (
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
    check_sandwich_compatible,
    check_transpose_matvec_out_shape,
    set_up_rows_or_cols,
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

        # Ensure array is contiguous (C or F order) for the compiled kernels
        # Only copy if necessary
        if (
            not input_array.flags["C_CONTIGUOUS"]
            and not input_array.flags["F_CONTIGUOUS"]
        ):
            warnings.warn(
                "Input array is not contiguous; making a copy.",
                UserWarning,
                stacklevel=2,
            )
            input_array = np.asfortranarray(input_array)

        self._array = input_array
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
        colnames = np.array(self.column_names)[col].ravel().tolist()
        terms = np.array(self.term_names)[col].ravel().tolist()

        sliced = self._array.__getitem__((row, col))
        # Slicing may yield a non-contiguous view; make it contiguous here so that
        # the constructor does not warn about an input the user did not provide.
        # Only copy when needed, to avoid a performance cost.
        if not (sliced.flags["C_CONTIGUOUS"] or sliced.flags["F_CONTIGUOUS"]):
            sliced = np.asfortranarray(sliced)
        return type(self)(sliced, column_names=colnames, term_names=terms)

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
    def shape(self):  # type: ignore
        """Tuple of array dimensions."""
        return self._array.shape

    @property
    def ndim(self):  # type: ignore
        """Number of array dimensions."""  # noqa: D401
        return self._array.ndim

    @property
    def dtype(self):  # type: ignore
        """Data type of the array's elements."""  # noqa: D401
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
        self,
        d: np.ndarray,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
    ) -> np.ndarray:
        """Perform a sandwich product: X.T @ diag(d) @ X."""
        return self._centered_sandwich(d, None, rows, cols)

    def _centered_sandwich(
        self,
        d: np.ndarray,
        center: np.ndarray | None,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ``(X - center).T @ diag(d) @ (X - center)`` without copying X.

        ``center`` has length ``self.shape[1]``; ``None`` means no centering. The
        subtraction is fused into the panel packing of the sandwich kernel.
        """
        d = np.asarray(d)
        check_sandwich_compatible(self, d)
        rows, cols = setup_restrictions(self.shape, rows, cols)
        return dense_sandwich(self._array, d, rows, cols, center=center)

    def _centered_transpose_matvec(
        self,
        vec: np.ndarray | list,
        center: np.ndarray | None,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ``(X - center)[rows, cols].T @ vec[rows]``.

        ``center`` has length ``self.shape[1]``; ``None`` means no centering. The
        subtraction happens inside the kernel, so that a large center does not
        cancel against ``X.T @ vec``.
        """
        check_transpose_matvec_out_shape(self, out)
        return self._matvec_helper(vec, rows, cols, out, True, center)

    def _centered_matvec(
        self,
        vec: np.ndarray | list,
        center: np.ndarray | None,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ``(X - center)[:, cols] @ vec[cols]``.

        See :meth:`_centered_transpose_matvec`.
        """
        check_matvec_out_shape(self, out)
        return self._matvec_helper(vec, None, cols, out, False, center)

    def _cross_sandwich(
        self,
        other: MatrixBase,
        d: np.ndarray,
        rows: np.ndarray | None = None,
        L_cols: np.ndarray | None = None,
        R_cols: np.ndarray | None = None,
    ):
        from .categorical_matrix import CategoricalMatrix
        from .sparse_matrix import SparseMatrix

        if isinstance(other, SparseMatrix) or isinstance(other, CategoricalMatrix):
            return other._cross_sandwich(self, d, rows, R_cols, L_cols).T
        raise TypeError

    def _get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        """Get standard deviations of columns using weights `weights`."""
        sqrt_arg = transpose_square_dot_weights(self._array, weights, col_means)
        # Minor floating point errors above can result in a very slightly
        # negative sqrt_arg (e.g. -5e-16). We just set those values equal to
        # zero.
        sqrt_arg[sqrt_arg < 0] = 0
        return np.sqrt(sqrt_arg)

    def _matvec_helper(
        self,
        vec: list | np.ndarray,
        rows: np.ndarray | None,
        cols: np.ndarray | None,
        out: np.ndarray | None,
        transpose: bool,
        center: np.ndarray | None = None,
    ):
        # Because the dense_rmatvec takes a row array and col array, it has
        # added overhead compared to a raw matrix vector product. So, when
        # we're not filtering or centering at all, let's just use default numpy
        # dot product.
        vec = np.asarray(vec)
        check_matvec_dimensions(self, vec, transpose=transpose)
        X = self._array.T if transpose else self._array

        # NOTE: We assume that rows and cols are unique. All rows in any order
        # give the same result, but the order of cols is the output order of
        # transpose_matvec, so only the identity counts as unrestricted.
        unrestricted_rows = rows is None or len(rows) == self.shape[0]
        unrestricted_cols = cols is None or np.array_equal(
            cols, np.arange(self.shape[1])
        )

        if center is None and unrestricted_rows and unrestricted_cols:
            if out is None:
                out = X.dot(vec)
            else:
                out += X.dot(vec)
            return out

        if center is not None and vec.dtype != self.dtype:
            # The kernels need matching dtypes, so expand the centered product
            # instead. This can cancel if the center is large.
            res = self._matvec_helper(vec, rows, cols, None, transpose)
            all_rows, all_cols = setup_restrictions(self.shape, rows, cols)
            if transpose:
                res -= np.multiply.outer(center[all_cols], vec[all_rows].sum(axis=0))
            else:
                res -= center[all_cols] @ vec[all_cols]
        else:
            # The kernels vectorize along rows or columns that are not restricted.
            kernel_rows = None if unrestricted_rows else rows
            kernel_cols = None if unrestricted_cols else cols
            if kernel_rows is not None:
                kernel_rows = set_up_rows_or_cols(kernel_rows, self.shape[0])
            if kernel_cols is not None:
                kernel_cols = set_up_rows_or_cols(kernel_cols, self.shape[1])
            fast_fnc = dense_rmatvec if transpose else dense_matvec

            def kernel(v):
                v = np.ascontiguousarray(v)
                return fast_fnc(self._array, v, kernel_rows, kernel_cols, center=center)

            # TODO: should take 'out' parameter
            if vec.ndim == 1:
                res = kernel(vec)
            elif vec.ndim == 2 and vec.shape[1] == 1:
                res = kernel(vec[:, 0])[:, None]
            elif vec.ndim == 2 and vec.shape[1] > 1 and center is not None:
                res = np.stack([kernel(v) for v in vec.T], axis=1)
            else:
                all_rows, all_cols = setup_restrictions(self.shape, rows, cols)
                subset = self._array[np.ix_(all_rows, all_cols)]
                if transpose:
                    res = subset.T.dot(vec[all_rows])
                else:
                    res = subset.dot(vec[all_cols])

        if out is None:
            return res
        if transpose and not unrestricted_cols:
            out[set_up_rows_or_cols(cols, self.shape[1])] += res
        else:
            # Note that matvec currently always uses all rows.
            out += res
        return out

    def transpose_matvec(
        self,
        vec: np.ndarray | list,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Perform: self[rows, cols].T @ vec[rows]."""
        check_transpose_matvec_out_shape(self, out)
        return self._matvec_helper(vec, rows, cols, out, True)

    def matvec(
        self,
        vec: np.ndarray | list,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
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
        missing_prefix: str | None = None,
        indices: list[int] | None = None,
    ) -> list[str | None]:
        """Get column names.

        For columns that do not have a name, a default name is created using the
        following pattern: ``"{missing_prefix}{start_index + i}"`` where ``i`` is
        the index of the column.

        Parameters
        ----------
        type: str {'column'|'term'}
            Whether to get column names or term names. The main difference is
            that a categorical submatrix counts as one term, but can count as
            multiple columns. Furthermore, matrices created from formulas
            distinguish between columns and terms (c.f. ``formulaic`` docs).
        missing_prefix: Optional[str], default None
            Prefix to use for columns that do not have a name. If None, then no
            default name is created.
        indices
            The indices used for columns that do not have a name. If ``None``,
            then the indices are ``list(range(self.shape[1]))``.

        Returns
        -------
        list[Optional[str]]
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

        return names.tolist()

    def set_names(self, names: str | list[str | None], type: str = "column"):
        """Set column names.

        Parameters
        ----------
        names: list[Optional[str]]
            Names to set.
        type: str {'column'|'term'}
            Whether to get column names or term names. The main difference is
            that a categorical submatrix counts as one term, but can count as
            multiple columns. Furthermore, matrices created from formulas
            distinguish between columns and terms (c.f. ``formulaic`` docs).
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
