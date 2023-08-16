"""
Categorical data.

One-hot encoding a feature creates a sparse matrix that has some special properties:
All of its nonzero elements are ones, and since each element starts a new row, it's ``indptr``,
which indicates where rows start and end, will increment by 1 every time.

Storage
^^^^^^^

csr storage
-----------

::

    >>> import numpy as np
    >>> from scipy import sparse
    >>> import pandas as pd

    >>> arr = [1, 0, 1]
    >>> dummies = pd.get_dummies(arr, dtype="uint8")
    >>> csr = sparse.csr_matrix(dummies.values)
    >>> csr.data
    array([1, 1, 1], dtype=uint8)
    >>> csr.indices
    array([1, 0, 1], dtype=int32)
    >>> csr.indptr
    array([0, 1, 2, 3], dtype=int32)


The size of this matrix, if the original array is of length ``n``, is ``n`` bytes for the
data (stored as quarter-precision integers), ``4n`` for ``indices``, and ``4(n+1)`` for
``indptr``. However, if we know the matrix results from one-hot encoding, we only need to
store the ``indices``, so we can reduce memory usage to slightly less than 4/9 of the
original.

csc storage
-----------
The case is not quite so simple for csc (column-major) sparse matrices.
However, we still do not need to store the data.

::

    >>> import numpy as np
    >>> from scipy import sparse
    >>> import pandas as pd

    >>> arr = [1, 0, 1]
    >>> dummies = pd.get_dummies(arr, dtype="uint8")
    >>> csc = sparse.csc_matrix(dummies.values)
    >>> csc.data
    array([1, 1, 1], dtype=uint8)
    >>> csc.indices
    array([1, 0, 2], dtype=int32)
    >>> csc.indptr
    array([0, 1, 3], dtype=int32)

Computations
^^^^^^^^^^^^

Matrix vector products
----------------------

A general sparse CSR matrix-vector products in pseudocode,
modeled on [scipy sparse](https://github.com/scipy/scipy/blob/1dc960a33b000b95b1e399582c154efc0360a576/scipy/sparse/sparsetools/csr.h#L1120):  # noqa:

::

    def matvec(mat, vec):
        n_row = mat.shape[0]
        res = np.zeros(n_row)
        for i in range(n_row):
            for j in range(mat.indptr[i], mat.indptr[i+1]):
                res[i] += mat.data[j] * vec[mat.indices[j]]
        return res

With a CSR categorical matrix, ``data`` is all 1 and ``j`` always equals ``i``, so we can
simplify this function to be

::

    def matvec(mat, vec):
        n_row = mat.shape[0]
        res = np.zeros(n_row)
        for i in range(n_row):
            res[i] = vec[mat.indices[j]]
        return res

The original function involved ``6N`` lookups, ``N`` multiplications, and ``N`` additions,
while the new function involves only ``3N`` lookups. It thus has the potential to be
significantly faster.

sandwich: X.T @ diag(d) @ X
---------------------------

![Narrow data set](images/narrow_data_sandwich.png)

![Medium-width data set](images/intermediate_data_sandwich.png)

![Wide data set](images/wide_data_sandwich.png)

Sandwich products can be computed very efficiently.

::

    sandwich(X, d)[i, j] = sum_k X[k, i] d[k] X[k, j]

If ``i != j``, ``sum_k X[k, i] d[k] X[k, j]`` = 0. In other words, since
 categorical matrices have only one nonzero per row, the sandwich product is diagonal.

 If ``i = j``,

::

    sandwich(X, d)[i, j] = sum_k X[k, i] d[k] X[k, i]
    = sum_k X[k, i] d[k]
    = d[X[:, i]].sum()
    = (X.T @ d)[i]

So ``sandwich(X, d) = diag(X.T @ d)``. This will be especially efficient if ``X`` is
available in CSC format. Pseudocode for this sandwich product is

::

    res = np.zeros(n_cols)
    for i in range(n_cols):
        for j in range(X.indptr[i], X.indptr[i + 1]):
            val += d[indices[j]]
    return np.diag(res)


This function is in ``ext/categorical/sandwich_categorical``

Cross-sandwich: X.T @ diag(d) @ Y where Y is categorical
--------------------------------------------------------

If X and Y are different categorical matrices in csr format,
X.T @ diag(d) @ Y is given by

::

    res = np.zeros((X.shape[1], Y.shape[1]))
    for k in range(len(d)):
        res[X.indices[k], Y.indices[k]] += d[k]


So the result will be sparse with at most N elements.
This function is given by ``ext/split/_sandwich_cat_cat``.

Cross-sandwich: X.T @ diag(d) @ Y where Y is dense
--------------------------------------------------

This is `ext/split/sandwich_cat_dense`

::

    res = np.zeros((X.shape[1], Y.shape[1]))
    for k in range(n_rows):
        for j in range(Y.shape[1]):
            res[X.indices[k], j] += d[k] * Y[k, j]

"""

import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from .ext.categorical import (
    matvec_complex,
    matvec_fast,
    multiply_complex,
    sandwich_categorical_complex,
    sandwich_categorical_fast,
    subset_categorical_complex,
    transpose_matvec_complex,
    transpose_matvec_fast,
)
from .ext.split import sandwich_cat_cat, sandwich_cat_dense
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .util import (
    _check_indexer,
    check_matvec_dimensions,
    check_matvec_out_shape,
    check_transpose_matvec_out_shape,
    set_up_rows_or_cols,
    setup_restrictions,
)


def _is_indexer_full_length(full_length: int, indexer: Union[slice, np.ndarray]):
    if isinstance(indexer, np.ndarray):
        if (indexer > full_length - 1).any():
            raise IndexError("Index out-of-range.")
        # Order is important in indexing. Could achieve similar results
        # by rearranging categories.
        return np.array_equal(indexer.ravel(), np.arange(full_length))
    elif isinstance(indexer, slice):
        return len(range(*indexer.indices(full_length))) == full_length


def _row_col_indexing(
    arr: np.ndarray, rows: Optional[np.ndarray], cols: Optional[np.ndarray]
) -> np.ndarray:
    if isinstance(rows, slice) and rows == slice(None, None, None):
        rows = None
    if isinstance(cols, slice) and cols == slice(None, None, None):
        cols = None

    is_row_indexed = not (rows is None or len(rows) == arr.shape[0])
    is_col_indexed = not (cols is None or len(cols) == arr.shape[1])

    if is_row_indexed and is_col_indexed:
        return arr[np.ix_(rows, cols)]
    elif is_row_indexed:
        return arr[rows]
    elif is_col_indexed:
        return arr[:, cols]
    else:
        return arr


class CategoricalMatrix(MatrixBase):
    """
    A faster, more memory efficient sparse matrix adapted to the specific
    settings of a one-hot encoded categorical variable.

    Parameters
    ----------
    cat_vec:
        array-like vector of categorical data.

    drop_first:
        drop the first level of the dummy encoding. This allows a CategoricalMatrix
        to be used in an unregularized setting.

    cat_missing_method: str {'fail'|'zero'|'convert'}, default 'fail'
        - if 'fail', raise an error if there are missing values.
        - if 'zero', missing values will represent all-zero indicator columns.
        - if 'convert', missing values will be converted to the ``cat_missing_name``
          category.

    cat_missing_name: str, default '(MISSING)'
        Name of the category to which missing values will be converted if
        ``cat_missing_method='convert'``. If this category already exists, an error
        will be raised.

    dtype:
        data type
    """

    def __init__(
        self,
        cat_vec: Union[List, np.ndarray, pd.Categorical],
        drop_first: bool = False,
        dtype: np.dtype = np.float64,
        column_name: Optional[str] = None,
        term_name: Optional[str] = None,
        column_name_format: str = "{name}[{category}]",
        cat_missing_method: str = "fail",
        cat_missing_name: str = "(MISSING)",
    ):
        if cat_missing_method not in ["fail", "zero", "convert"]:
            raise ValueError(
                "cat_missing_method must be one of 'fail' 'zero' or 'convert', "
                f" got {cat_missing_method}"
            )
        self._missing_method = cat_missing_method
        self._missing_category = cat_missing_name

        if isinstance(cat_vec, pd.Categorical):
            self.cat = cat_vec
        else:
            self.cat = pd.Categorical(cat_vec)

        if pd.isnull(self.cat).any():
            if self._missing_method == "fail":
                raise ValueError(
                    "Categorical data can't have missing values "
                    "if cat_missing_method='fail'."
                )

            elif self._missing_method == "convert":
                if self._missing_category in self.cat.categories:
                    raise ValueError(
                        f"Missing category {self._missing_category} already exists."
                    )

                self.cat = self.cat.add_categories([self._missing_category])

                self.cat[pd.isnull(self.cat)] = self._missing_category
                self._has_missings = False

            else:
                self._has_missings = True

        else:
            self._has_missings = False

        self.drop_first = drop_first
        self.shape = (len(self.cat), len(self.cat.categories) - int(drop_first))
        self.indices = self.cat.codes.astype(np.int32)
        self.x_csc: Optional[Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]] = None
        self.dtype = np.dtype(dtype)

        self._colname = column_name
        if term_name is None:
            self._term = self._colname
        else:
            self._term = term_name
        self._colname_format = column_name_format

    __array_ufunc__ = None

    def recover_orig(self) -> np.ndarray:
        """
        Return 1d numpy array with same data as what was initially fed to __init__.

        Test: matrix/test_categorical_matrix::test_recover_orig
        """
        orig = self.cat.categories[self.cat.codes].to_numpy()

        if self._has_missings:
            orig = orig.view(np.ma.MaskedArray)
            orig.mask = self.cat.codes == -1
        elif (
            self._missing_method == "convert"
            and self._missing_category in self.cat.categories
        ):
            orig = orig.view(np.ma.MaskedArray)
            missing_code = self.cat.categories.get_loc(self._missing_category)
            orig.mask = self.cat.codes == missing_code

        return orig

    def _matvec_setup(
        self,
        other: Union[List, np.ndarray],
        cols: np.ndarray = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        other = np.asarray(other)
        if other.ndim > 1:
            raise NotImplementedError(
                """CategoricalMatrix.matvec is only implemented for 1d arrays."""
            )
        check_matvec_dimensions(self, other, transpose=False)

        if cols is not None:
            if len(cols) == self.shape[1]:
                cols = None
            else:
                # Needs to be single-precision for compatibility with cython 'int' type
                # Since we have the same restriction on self.indices, this is not an
                # additional restriction (as column numbers can't exceed 2^32 anyway)
                cols = set_up_rows_or_cols(cols, self.shape[1])

        return other, cols

    def matvec(
        self,
        other: Union[List, np.ndarray],
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """
        Multiply self with vector 'other', and add vector 'out' if it is present.

        out[i] += sum_j mat[i, j] other[j] = other[mat.indices[i]]

        The cols parameter allows restricting to a subset of the
        matrix without making a copy.

        If out is None, then a new array will be returned.

        Test:
        test_matrices::test_matvec
        """
        check_matvec_out_shape(self, out)
        other, cols = self._matvec_setup(other, cols)
        is_int = np.issubdtype(other.dtype, np.signedinteger)

        if is_int:
            other_m = other.astype(float)
        else:
            other_m = other

        if out is None:
            out = np.zeros(self.shape[0], dtype=other_m.dtype)

        if self.drop_first or self._has_missings:
            matvec_complex(
                self.indices,
                other_m,
                self.shape[0],
                cols,
                self.shape[1],
                out,
                self.drop_first,
            )
        else:
            matvec_fast(self.indices, other_m, self.shape[0], cols, self.shape[1], out)

        if is_int:
            return out.astype(int)
        return out

    def transpose_matvec(
        self,
        vec: Union[np.ndarray, List],
        rows: Optional[np.ndarray] = None,
        cols: Optional[np.ndarray] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """

        Perform: self[rows, cols].T @ vec[rows].

        ::

            for i in cols: out[i] += sum_{j in rows} self[j, i] vec[j]
            self[j, i] = 1(indices[j] == i)


            for j in rows:
                for i in cols:
                    out[i] += (indices[j] == i) * vec[j]

        If cols == range(self.shape[1]), then for every row j, there will be exactly
            one relevant column, so you can do

        ::

            for j in rows,
                out[indices[j]] += vec[j]

        The rows and cols parameters allow restricting to a subset of the
        matrix without making a copy.

        If out is None, then a new array will be returned.

        Test: tests/test_matrices::test_transpose_matvec
        """
        # TODO: write a function that doesn't reference the data
        # TODO: this should look more like the cat_cat_sandwich
        vec = np.asarray(vec)
        check_matvec_dimensions(self, vec, transpose=True)
        if vec.ndim > 1:
            raise NotImplementedError(
                "CategoricalMatrix.transpose_matvec is only implemented for 1d arrays."
            )

        out_is_none = out is None
        if out_is_none:
            out = np.zeros(self.shape[1], dtype=self.dtype)
        else:
            check_transpose_matvec_out_shape(self, out)

        if rows is not None:
            rows = set_up_rows_or_cols(rows, self.shape[0])
        if cols is not None:
            cols = set_up_rows_or_cols(cols, self.shape[1])

        if self.drop_first or self._has_missings:
            transpose_matvec_complex(
                self.indices,
                vec,
                self.shape[1],
                vec.dtype,
                rows,
                cols,
                out,
                self.drop_first,
            )
        else:
            transpose_matvec_fast(
                self.indices, vec, self.shape[1], vec.dtype, rows, cols, out
            )

        if out_is_none and cols is not None:
            return out[cols, ...]  # type: ignore
        return out

    def sandwich(
        self,
        d: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> sps.dia_matrix:
        """
        Perform a sandwich product: X.T @ diag(d) @ X.

        .. code-block:: python

            sandwich(self, d)[i, j] = (self.T @ diag(d) @ self)[i, j]
                = sum_k (self[k, i] (diag(d) @ self)[k, j])
                = sum_k self[k, i] sum_m diag(d)[k, m] self[m, j]
                = sum_k self[k, i] d[k] self[k, j]
                = 0 if i != j
            sandwich(self, d)[i, i] = sum_k self[k, i] ** 2 * d(k)

        The rows and cols parameters allow restricting to a subset of the
        matrix without making a copy.
        """
        d = np.asarray(d)
        rows = set_up_rows_or_cols(rows, self.shape[0])
        if self.drop_first or self._has_missings:
            res_diag = sandwich_categorical_complex(
                self.indices, d, rows, d.dtype, self.shape[1], self.drop_first
            )
        else:
            res_diag = sandwich_categorical_fast(
                self.indices, d, rows, d.dtype, self.shape[1]
            )

        if cols is not None and len(cols) < self.shape[1]:
            res_diag = res_diag[cols]
        return sps.diags(res_diag)

    def _cross_sandwich(
        self,
        other: MatrixBase,
        d: Union[np.ndarray, List],
        rows: Optional[np.ndarray] = None,
        L_cols: Optional[np.ndarray] = None,
        R_cols: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Perform a sandwich product: X.T @ diag(d) @ Y."""
        from .dense_matrix import DenseMatrix

        if isinstance(other, DenseMatrix):
            return self._cross_dense(other._array, d, rows, L_cols, R_cols)
        if isinstance(other, SparseMatrix):
            return self._cross_sparse(other.array_csc, d, rows, L_cols, R_cols)
        if isinstance(other, CategoricalMatrix):
            return self._cross_categorical(other, d, rows, L_cols, R_cols)
        raise TypeError

    # TODO: best way to return this depends on the use case. See what that is
    # See how csr getcol works
    def getcol(self, i: int) -> SparseMatrix:
        """Return matrix column at specified index."""
        i %= self.shape[1]  # wrap-around indexing

        if self.drop_first:
            i_corr = i + 1
        else:
            i_corr = i

        col_i = sps.csc_matrix((self.indices == i_corr).astype(int)[:, None])
        return SparseMatrix(
            col_i,
            column_names=[self.column_names[i]],
            term_names=[self.term_names[i]],
        )

    def tocsr(self) -> sps.csr_matrix:
        """Return scipy csr representation of matrix."""
        if self.drop_first or self._has_missings:
            nnz, indices, indptr = subset_categorical_complex(
                self.indices, self.shape[1], self.drop_first
            )
            return sps.csr_matrix(
                (np.ones(nnz, dtype=int), indices, indptr), shape=self.shape
            )

        # TODO: data should be uint8
        data = np.ones(self.shape[0], dtype=int)
        return sps.csr_matrix(
            (data, self.indices, np.arange(self.shape[0] + 1, dtype=int)),
            shape=self.shape,
        )

    def to_sparse_matrix(self):
        """Return a tabmat.SparseMatrix representation."""
        from .sparse_matrix import SparseMatrix

        return SparseMatrix(
            self.tocsr(),
            column_names=self.column_names,
            term_names=self.term_names,
        )

    def toarray(self) -> np.ndarray:
        """Return array representation of matrix."""
        return self.tocsr().A

    def unpack(self):
        """Return the underlying pandas.Categorical."""
        return self.cat

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """Return CategoricalMatrix cast to new type."""
        self.dtype = dtype
        return self

    def _get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        """Get standard deviations of columns."""
        # To calculate a variance, we'd normally need to compute E[X^2]:
        # sum_i X_ij^2 w_i
        # but because X_ij is either {0, 1}
        # we don't actually need to square.
        mean = self.transpose_matvec(weights)
        return np.sqrt(mean - col_means**2)

    def __getitem__(self, item):
        row, col = _check_indexer(item)

        if _is_indexer_full_length(self.shape[1], col):
            if isinstance(row, np.ndarray):
                row = row.ravel()
            return CategoricalMatrix(
                self.cat[row],
                drop_first=self.drop_first,
                dtype=self.dtype,
                column_name=self._colname,
                column_name_format=self._colname_format,
                cat_missing_method=self._missing_method,
            )
        else:
            # return a SparseMatrix if we subset columns
            # TODO: this is inefficient. See issue #101.
            return self.to_sparse_matrix()[row, col]

    def _cross_dense(
        self,
        other: np.ndarray,
        d: np.ndarray,
        rows: Optional[np.ndarray],
        L_cols: Optional[np.ndarray],
        R_cols: Optional[np.ndarray],
    ) -> np.ndarray:
        if other.flags["C_CONTIGUOUS"]:
            is_c_contiguous = True
        elif other.flags["F_CONTIGUOUS"]:
            is_c_contiguous = False
        else:
            raise ValueError(
                "Input array needs to be either C-contiguous or F-contiguous."
            )

        rows, R_cols = setup_restrictions((self.shape[0], other.shape[1]), rows, R_cols)

        res = sandwich_cat_dense(
            self.indices,
            self.shape[1],
            d,
            other,
            rows,
            R_cols,
            is_c_contiguous,
            has_missings=self._has_missings,
            drop_first=self.drop_first,
        )

        res = _row_col_indexing(res, L_cols, None)
        return res

    def _cross_categorical(
        self,
        other,
        d: np.ndarray,
        rows: Optional[np.ndarray],
        L_cols: Optional[np.ndarray],
        R_cols: Optional[np.ndarray],
    ) -> np.ndarray:
        if not isinstance(other, CategoricalMatrix):
            raise TypeError

        i_indices = self.indices
        j_indices = other.indices
        rows = set_up_rows_or_cols(rows, self.shape[0])

        res = sandwich_cat_cat(
            i_indices,
            j_indices,
            self.shape[1],
            other.shape[1],
            d,
            rows,
            d.dtype,
            self.drop_first,
            other.drop_first,
            self._has_missings,
            other._has_missings,
        )

        res = _row_col_indexing(res, L_cols, R_cols)
        return res

    def _cross_sparse(
        self,
        other: sps.csc_matrix,
        d: np.ndarray,
        rows: Optional[np.ndarray],
        L_cols: Optional[np.ndarray],
        R_cols: Optional[np.ndarray],
    ) -> np.ndarray:
        term_1 = self.multiply(d)  # multiply will deal with drop_first

        term_1 = _row_col_indexing(term_1, rows, L_cols)

        res = term_1.T.dot(_row_col_indexing(other, rows, R_cols)).A
        return res

    def multiply(self, other) -> SparseMatrix:
        """Element-wise multiplication.

        This assumes that ``other`` is a vector of size ``self.shape[0]``.
        """
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                f"Shapes do not match. Expected length of {self.shape[0]}. Got {len(other)}."
            )

        if self.drop_first or self._has_missings:
            return SparseMatrix(
                sps.csr_matrix(
                    multiply_complex(
                        indices=self.indices,
                        d=np.squeeze(other),
                        ncols=self.shape[1],
                        dtype=other.dtype,
                        drop_first=self.drop_first,
                    ),
                    shape=self.shape,
                )
            )

        return SparseMatrix(
            sps.csr_matrix(
                (
                    np.squeeze(other),
                    self.indices,
                    np.arange(self.shape[0] + 1, dtype=int),
                ),
                shape=self.shape,
            ),
            column_names=self.column_names,
            term_names=self.term_names,
        )

    def __repr__(self):
        return str(self.cat)

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
            name = self._colname
        elif type == "term":
            name = self._term
        else:
            raise ValueError(f"Type must be 'column' or 'term', got {type}")

        if indices is None:
            indices = list(range(len(self.cat.categories) - self.drop_first))
        if name is None and missing_prefix is None:
            return [None] * (len(self.cat.categories) - self.drop_first)
        elif name is None:
            name = f"{missing_prefix}{indices[0]}-{indices[-1]}"

        if type == "column":
            return [
                self._colname_format.format(name=name, category=cat)
                for cat in self.cat.categories[self.drop_first :]
            ]
        else:
            return [name] * (len(self.cat.categories) - self.drop_first)

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

        if len(names) != 1:
            if type == "column":
                # Try finding the column name
                base_names = []
                for name, cat in zip(names, self.cat.categories[self.drop_first :]):
                    partial_name = self._colname_format.format(
                        name="__CAPTURE__", category=cat
                    )
                    pattern = re.escape(partial_name).replace("__CAPTURE__", "(.*)")
                    if name is not None:
                        match = re.search(pattern, name)
                    else:
                        match = None
                    if match is not None:
                        base_names.append(match.group(1))
                    else:
                        base_names.append(name)
                names = base_names

            if len(names) == self.shape[1] and all(name == names[0] for name in names):
                names = [names[0]]

        if len(names) != 1:
            raise ValueError("A categorical matrix has only one name")

        if type == "column":
            self._colname = names[0]
        elif type == "term":
            self._term = names[0]
        else:
            raise ValueError(f"Type must be 'column' or 'term', got {type}")
