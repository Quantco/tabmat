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
    >>> dummies = pd.get_dummies(arr)
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
    >>> dummies = pd.get_dummies(arr)
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

A general sparse CSR matrix-vector products in psuedocode,
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

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from .ext.categorical import matvec, sandwich_categorical, transpose_matvec
from .ext.split import sandwich_cat_cat, sandwich_cat_dense
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .util import (
    check_matvec_out_shape,
    check_transpose_matvec_out_shape,
    set_up_rows_or_cols,
    setup_restrictions,
)


def _is_indexer_full_length(full_length: int, indexer: Any):
    if isinstance(indexer, int):
        return full_length == 1
    elif isinstance(indexer, list):
        if (np.asarray(indexer) > full_length - 1).any():
            raise IndexError("Index out-of-range.")
        return len(set(indexer)) == full_length
    elif isinstance(indexer, np.ndarray):
        if (indexer > full_length - 1).any():
            raise IndexError("Index out-of-range.")
        return len(np.unique(indexer)) == full_length
    elif isinstance(indexer, slice):
        return len(range(*indexer.indices(full_length))) == full_length
    else:
        raise ValueError(f"Indexing with {type(indexer)} is not allowed.")


def _none_to_slice(arr: Optional[np.ndarray], n: int) -> Union[slice, np.ndarray]:
    if arr is None or len(arr) == n:
        return slice(None, None, None)
    return arr


class CategoricalMatrix(MatrixBase):
    """
    A faster, more memory efficient sparse matrix adapted to the specific
    settings of a one-hot encoded categorical variable.

    Parameters
    ----------
    cat_vec:
        array-like vector of categorical data.

    dtype:
    """

    def __init__(
        self,
        cat_vec: Union[List, np.ndarray, pd.Categorical],
        dtype: np.dtype = np.float64,
    ):
        if pd.isnull(cat_vec).any():
            raise ValueError("Categorical data can't have missing values.")

        if isinstance(cat_vec, pd.Categorical):
            self.cat = cat_vec
        else:
            self.cat = pd.Categorical(cat_vec)

        self.shape = (len(self.cat), len(self.cat.categories))
        self.indices = self.cat.codes.astype(np.int32)
        self.x_csc: Optional[Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]] = None
        self.dtype = np.dtype(dtype)

    def recover_orig(self) -> np.ndarray:
        """
        Return 1d numpy array with same data as what was initially fed to __init__.

        Test: matrix/test_categorical_matrix::test_recover_orig
        """
        return self.cat.categories[self.cat.codes]

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
        if other.shape[0] != self.shape[1]:
            raise ValueError(
                f"""Needed other to have first dimension {self.shape[1]},
                but it has shape {other.shape}"""
            )

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

        matvec(self.indices, other_m, self.shape[0], cols, self.shape[1], out)
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

        Perform: self[rows, cols].T @ vec.

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

        transpose_matvec(self.indices, vec, self.shape[1], vec.dtype, rows, cols, out)
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
        res_diag = sandwich_categorical(self.indices, d, rows, d.dtype, self.shape[1])
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
        if isinstance(other, np.ndarray):
            return self._cross_dense(other, d, rows, L_cols, R_cols)
        if isinstance(other, sps.csc_matrix):
            return self._cross_sparse(other, d, rows, L_cols, R_cols)
        if isinstance(other, CategoricalMatrix):
            return self._cross_categorical(other, d, rows, L_cols, R_cols)
        raise TypeError

    # TODO: best way to return this depends on the use case. See what that is
    # See how csr getcol works
    def getcol(self, i: int) -> sps.csc_matrix:
        """Return matrix column at specified index."""
        i %= self.shape[1]  # wrap-around indexing
        col_i = sps.csc_matrix((self.indices == i).astype(int)[:, None])
        return col_i

    def tocsr(self) -> sps.csr_matrix:
        """Return scipy csr representation of matrix."""
        # TODO: data should be uint8
        data = np.ones(self.shape[0], dtype=int)

        return sps.csr_matrix(
            (data, self.indices, np.arange(self.shape[0] + 1, dtype=int)),
            shape=self.shape,
        )

    def toarray(self) -> np.ndarray:
        """Return array representation of matrix."""
        return self.tocsr().A

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
        return np.sqrt(mean - col_means ** 2)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
            if _is_indexer_full_length(self.shape[1], col):
                if isinstance(row, int):
                    row = [row]
                return CategoricalMatrix(self.cat[row])
            else:
                # return a SparseMatrix if we subset columns
                # TODO: this is inefficient. See issue #101.
                return SparseMatrix(self.tocsr()[row, col], dtype=self.dtype)
        else:
            row = item
        if isinstance(row, int):
            row = [row]
        return CategoricalMatrix(self.cat[row])

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
            self.indices, self.shape[1], d, other, rows, R_cols, is_c_contiguous
        )

        res = res[_none_to_slice(L_cols, self.shape[1]), :]
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
            i_indices, j_indices, self.shape[1], other.shape[1], d, rows, d.dtype
        )

        L_cols = _none_to_slice(L_cols, self.shape[1])
        R_cols = _none_to_slice(R_cols, other.shape[1])
        res = res[L_cols, :][:, R_cols]
        return res

    def _cross_sparse(
        self,
        other: sps.csc_matrix,
        d: np.ndarray,
        rows: Optional[np.ndarray],
        L_cols: Optional[np.ndarray],
        R_cols: Optional[np.ndarray],
    ) -> np.ndarray:

        term_1 = self.tocsr()
        term_1.data = term_1.data * d

        rows = _none_to_slice(rows, self.shape[0])
        L_cols = _none_to_slice(L_cols, self.shape[1])
        term_1 = term_1[rows, :][:, L_cols]

        res = term_1.T.dot(other[rows, :][:, _none_to_slice(R_cols, other.shape[1])]).A

        return res

    def multiply(self, other) -> sps.csr_matrix:
        """Element-wise multiplication of each column with other."""
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                f"Shapes do not match. Expected length of {self.shape[0]}. Got {len(other)}."
            )

        return sps.csr_matrix(
            (np.squeeze(other), self.indices, np.arange(self.shape[0] + 1, dtype=int)),
            shape=self.shape,
        )

    __mul__ = multiply
    __rmul__ = multiply

    def __repr__(self):
        return str(self.cat)
