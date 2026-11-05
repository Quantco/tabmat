import numpy as np
from scipy import sparse as sps

from .dense_matrix import DenseMatrix
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .util import (
    check_matvec_dimensions,
    check_sandwich_compatible,
    check_transpose_matvec_out_shape,
    set_up_rows_or_cols,
    setup_restrictions,
)


class StandardizedMatrix:
    """
    StandardizedMatrix allows for storing a matrix standardized to have columns
    that have mean zero and standard deviation one without modifying underlying
    sparse matrices.

    To be precise, for a StandardizedMatrix:

    ::

        self[i, j] = (self.mult[j] * self.mat[i, j]) + self.shift[j]

    This class is returned from
    :meth:`MatrixBase.standardize <tabmat.MatrixBase.standardize>`.
    """

    __array_priority__ = 11

    # The unscaled column centers c, with shift == -c * mult, if known exactly.
    # MatrixBase.standardize sets them; otherwise they are recovered from shift
    # and mult when needed (see _dense_centering).
    _center: np.ndarray | None = None

    def __init__(
        self,
        mat: MatrixBase,
        shift: np.ndarray | list,
        mult: np.ndarray | list | None = None,
    ):
        shift_arr = np.atleast_1d(np.squeeze(shift))
        expected_shape = (mat.shape[1],)
        if not isinstance(mat, MatrixBase):
            raise TypeError("mat should be an instance of a MatrixBase subclass.")
        if not shift_arr.shape == expected_shape:
            raise ValueError(
                f"""Expected shift to be able to conform to shape {expected_shape},
            but it has shape {np.asarray(shift).shape}"""
            )

        if mult is not None:
            mult_arr = np.atleast_1d(np.squeeze(mult))
            if not mult_arr.shape == expected_shape:
                raise ValueError(
                    f"""Expected mult to be able to conform to shape {expected_shape},
                but it has shape {np.asarray(mult).shape}"""
                )
        else:
            mult_arr = None

        self.shift = shift_arr
        self.mult = mult_arr
        self.mat = mat
        self.shape = mat.shape
        self.ndim = mat.ndim
        self.dtype = mat.dtype

    def matvec(
        self,
        other_mat: np.ndarray | list,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Perform self[:, cols] @ other[cols].

        This function returns a dense output, so it is best geared for the
        matrix-vector case.
        """
        cols = set_up_rows_or_cols(cols, self.shape[1])

        other_mat = np.asarray(other_mat)
        check_matvec_dimensions(self, other_mat, transpose=False)
        mult_other = other_mat
        if self.mult is not None:
            mult = self.mult
            # Avoiding an outer product by matching dimensions.
            for _ in range(len(other_mat.shape) - 1):
                mult = mult[:, np.newaxis]
            mult_other = mult * other_mat
        # Center the dense columns inside the kernels, see sandwich.
        center, shift = self._dense_centering()
        mat_part = self.mat._centered_matvec(mult_other, center, cols, out=out)

        # Add shift part to mat_part
        shift_part = shift[cols].dot(other_mat[cols, ...])  # scalar
        mat_part += shift_part
        return mat_part

    def getcol(self, i: int):
        """
        Return matrix column at specified index.

        Returns a StandardizedMatrix.

        >>> from scipy import sparse as sps
        >>> x = StandardizedMatrix(SparseMatrix(sps.eye(3).tocsc()), shift=[0, 1, -2])
        >>> col_1 = x.getcol(1)
        >>> isinstance(col_1, StandardizedMatrix)
        True
        >>> col_1.toarray()
        array([[1.],
               [2.],
               [1.]])
        """
        col = self.mat.getcol(i)
        if isinstance(col, sps.csc_matrix) and not isinstance(col, MatrixBase):
            col = SparseMatrix(col)
        return self._with_cols(col, i)

    def _with_cols(self, mat: MatrixBase, col) -> "StandardizedMatrix":
        """Wrap ``mat`` with the shift, mult and center of the columns ``col``."""
        mult = None if self.mult is None else np.atleast_1d(self.mult[col])
        out = StandardizedMatrix(mat, np.atleast_1d(self.shift[col]), mult)
        if self._center is not None:
            out._center = np.atleast_1d(self._center[col])
        return out

    def _dense_centering(self) -> tuple[np.ndarray | None, np.ndarray]:
        """Return ``(center, shift)`` for centering the dense columns in the kernels.

        ``center`` holds the unscaled means of the columns stored in a dense
        matrix and zero for the others, so that
        ``self.toarray() == (self.mat.toarray() - center) * mult + shift``.
        Centering a column absorbs its shift, so ``shift`` is zero for the
        centered columns. The exact means from :meth:`MatrixBase.standardize` are
        used while they still match ``self.shift``; otherwise they are recovered
        from ``self.shift`` and ``self.mult``. Columns whose mean cannot be
        recovered (a zero multiplier) or is zero keep their shift. ``center`` is
        ``None`` if no column is centered.
        """
        from .split_matrix import SplitMatrix

        dense = np.zeros(self.shape[1], dtype=bool)
        if isinstance(self.mat, DenseMatrix):
            dense[:] = True
        elif isinstance(self.mat, SplitMatrix):
            for mat, idx in zip(self.mat.matrices, self.mat.indices):
                dense[idx] = isinstance(mat, DenseMatrix)
        if not dense.any():
            return None, self.shift

        center = self._center
        if center is not None:
            # Allow for rounding, e.g. after astype casts shift and center
            # separately. A larger difference means shift was changed since.
            expected = -center if self.mult is None else -center * self.mult
            # Same as np.allclose(expected, self.shift, rtol=4 * eps, atol=0), but
            # cheaper; this runs for every product.
            eps = np.finfo(np.result_type(expected, self.shift)).eps
            if not np.all(
                np.abs(expected - self.shift) <= 4 * eps * np.abs(self.shift)
            ):
                center = None
        if center is None:
            if self.mult is None:
                center = -self.shift
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    center = -self.shift / self.mult
        centered = dense & np.isfinite(center) & (center != 0)
        if not centered.any():
            return None, self.shift
        return np.where(centered, center, 0.0), np.where(centered, 0.0, self.shift)

    def sandwich(
        self,
        d: np.ndarray,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
    ) -> np.ndarray:
        """Perform a sandwich product: X.T @ diag(d) @ X."""
        if not hasattr(d, "dtype"):
            d = np.asarray(d)
        check_sandwich_compatible(self, d)
        # stat_mat = mat * mult[newaxis, :] + shift[newaxis, :]
        # stat_mat.T @ d[:, newaxis] * stat_mat
        # = mult[:, newaxis] * mat.T @ d[:, newaxis] * mat * mult[newaxis, :] +   (1)
        #   mult[:, newaxis] * mat.T @ d[:, newaxis] * np.outer(ones, shift) +    (2)
        #   shift[:, newaxis] @ d[:, newaxis] * mat * mult[newaxis, :] +          (3)
        #   shift[:, newaxis] @ d[:, newaxis] * shift[newaxis, :]                 (4)
        #
        # (1) = self.mat.sandwich(d) * np.outer(limited_mult, limited_mult)
        # (2) = mult * self.transpose_matvec(d) * shift[newaxis, :]
        if rows is not None or cols is not None:
            setup_rows, setup_cols = setup_restrictions(self.shape, rows, cols)
            if rows is not None:
                rows = setup_rows
            if cols is not None:
                cols = setup_cols

        # The dense columns are centered inside the kernels, without copying the
        # data: for a DenseMatrix, (1) becomes
        # (mat - center).T @ diag(d) @ (mat - center) and (2) to (4) vanish.
        # Expanding the uncentered product instead forms terms that nearly cancel
        # when a column's mean is large relative to its standard deviation,
        # losing most significant digits.
        center, shift = self._dense_centering()
        term1 = self.mat._centered_sandwich(d, center, rows, cols)
        if isinstance(term1, sps.dia_matrix):
            term1 = np.diag(term1.data[0, :])

        limited_mult = None
        if self.mult is not None:
            limited_mult = self.mult[cols] if cols is not None else self.mult
            term1 *= np.outer(limited_mult, limited_mult)
        limited_shift = shift[cols] if cols is not None else shift
        if not limited_shift.any():
            # (2) to (4) vanish without a shift.
            return term1

        d_mat = self.mat._centered_transpose_matvec(d, center, rows, cols)
        if limited_mult is not None:
            d_mat *= limited_mult
        term2 = np.outer(d_mat, limited_shift)

        limited_d = d[rows] if rows is not None else d
        term3 = np.outer(limited_shift, d_mat)
        term4 = np.outer(limited_shift, limited_shift) * np.sum(limited_d)
        return term1 + term2 + term3 + term4

    def unstandardize(self) -> MatrixBase:
        """Get unstandardized (base) matrix."""
        return self.mat

    def transpose_matvec(
        self,
        other: np.ndarray | list,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Perform: self[rows, cols].T @ vec[rows].

        Let self.shape = (N, K) and other.shape = (M, N).
        Let shift_mat = outer(ones(N), shift)

        (X.T @ other)[k, i] = (X.mat.T @ other)[k, i] + (shift_mat @ other)[k, i]
        (shift_mat @ other)[k, i] = (outer(shift, ones(N)) @ other)[k, i]
        = sum_j outer(shift, ones(N))[k, j] other[j, i]
        = sum_j shift[k] other[j, i]
        = shift[k] other.sum(0)[i]
        = outer(shift, other.sum(0))[k, i]

        With row and col restrictions:

        self.transpose_matvec(other, rows, cols)[i, j]
            = self.mat.transpose_matvec(other, rows, cols)[i, j]
              + (outer(self.shift, ones(N))[rows, cols] @ other[cols])
            = self.mat.transpose_matvec(other, rows, cols)[i, j]
              + shift[cols[i]] other.sum(0)[rows[j]
        """
        check_transpose_matvec_out_shape(self, out)
        other = np.asarray(other)
        check_matvec_dimensions(self, other, transpose=True)
        # Center the dense columns inside the kernels, see sandwich.
        center, shift = self._dense_centering()
        res = self.mat._centered_transpose_matvec(other, center, rows, cols)

        rows, cols = setup_restrictions(self.shape, rows, cols)
        other_sum = np.sum(other[rows], 0)

        shift_part_tmp = np.outer(shift[cols], other_sum)
        output_shape = ((self.shape[1] if cols is None else len(cols)),) + res.shape[1:]
        shift_part = np.reshape(shift_part_tmp, output_shape)

        if self.mult is not None:
            mult = self.mult
            # Avoiding an outer product by matching dimensions.
            for _ in range(res.ndim - 1):
                mult = mult[:, np.newaxis]
            res *= mult[cols]
        res += shift_part

        if out is None:
            return res
        else:
            out[cols] += res
            return out

    def __rmatmul__(self, other: np.ndarray | list) -> np.ndarray:
        """
        Return matrix multiplication with other.

        other @ X = (X.T @ other.T).T = X.transpose_matvec(other.T).T

        Parameters
        ----------
        other: array-like

        Returns
        -------
        array

        """
        if not hasattr(other, "T"):
            other = np.asarray(other)
        return self.transpose_matvec(other.T).T  # type: ignore

    def __matmul__(self, other):
        """Define the behavior of 'self @ other'."""
        return self.matvec(other)

    def multiply(self, other) -> DenseMatrix:
        """Element-wise multiplication.

        Note that the output of this function is always a DenseMatrix and might
        require a lot more memory. This assumes that ``other`` is a vector of
        size ``self.shape[0]``.
        """
        return DenseMatrix(self.toarray()).multiply(other)

    def toarray(self) -> np.ndarray:
        """Return array representation of matrix."""
        mat_part = self.mat.toarray()
        if self.mult is not None:
            mat_part = self.mult[None, :] * mat_part
        return mat_part + self.shift[None, :]

    @property
    def A(self) -> np.ndarray:
        """Return array representation of self."""
        return self.toarray()

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """Return StandardizedMatrix cast to new type."""
        mult = None
        if self.mult is not None:
            mult = self.mult.astype(dtype, order=order, casting=casting, copy=copy)
        out = type(self)(
            self.mat.astype(dtype, casting=casting, copy=copy),
            self.shift.astype(dtype, order=order, casting=casting, copy=copy),
            mult,
        )
        if self._center is not None:
            out._center = self._center.astype(
                dtype, order=order, casting=casting, copy=copy
            )
        return out

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
        else:
            row = item
            col = slice(None, None, None)

        out = self._with_cols(self.mat.__getitem__(item), col)
        # A single row is returned as an array.
        return out.toarray() if isinstance(row, int) else out

    def __repr__(self):
        out = f"""StandardizedMat. Mat: {type(self.mat)} of shape {self.mat.shape}.
        Shift: {self.shift}
        Mult: {self.mult}
        """
        return out

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
        return self.mat.get_names(type, missing_prefix, indices)

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
        self.mat.set_names(names, type)

    @property
    def column_names(self):
        """Column names of the matrix."""
        return self.get_names(type="column")

    @column_names.setter
    def column_names(self, names: list[str | None]):
        self.set_names(names, type="column")

    @property
    def term_names(self):
        """Term names of the matrix.

        For differences between column names and term names, see ``get_names``.
        """
        return self.get_names(type="term")

    @term_names.setter
    def term_names(self, names: list[str | None]):
        self.set_names(names, type="term")
