import warnings
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from scipy import sparse as sps

from .dense_matrix import DenseMatrix
from .ext.split import is_sorted, split_col_subsets
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .standardized_mat import StandardizedMatrix
from .util import (
    check_matvec_dimensions,
    check_matvec_out_shape,
    check_sandwich_compatible,
    check_transpose_matvec_out_shape,
    set_up_rows_or_cols,
)


def as_tabmat(a: Union[MatrixBase, StandardizedMatrix, np.ndarray, sps.spmatrix]):
    """Convert an array to a corresponding MatrixBase type.

    If the input is already a MatrixBase, return untouched.
    If the input is sparse, return a SparseMatrix.
    If the input is a numpy array, return a DenseMatrix.
    Raise an error is input is another type.
    """
    if isinstance(a, (MatrixBase, StandardizedMatrix)):
        return a
    elif sps.issparse(a):
        return SparseMatrix(a.tocsc(copy=False))  # type: ignore
    elif isinstance(a, np.ndarray):
        return DenseMatrix(a)
    else:
        raise ValueError(f"Cannot convert type {type(a)} to Matrix.")


def hstack(tup: Sequence[Union[MatrixBase, np.ndarray, sps.spmatrix]]) -> MatrixBase:
    """Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis,
    except for 1-D arrays where it concatenates along the first axis.

    Parameters
    ----------
    tup: sequence of arrays
        The arrays must have the same shape along all but the second axis.
    """
    matrices = [as_tabmat(a) for a in tup]

    if len(matrices) == 0:
        raise ValueError("Need at least one array to concatenate.")

    if all(isinstance(mat, SparseMatrix) for mat in matrices):
        return SparseMatrix(sps.hstack([mat._array for mat in matrices]))
    elif all(isinstance(mat, DenseMatrix) for mat in matrices):
        return DenseMatrix(np.hstack([mat._array for mat in matrices]))
    else:
        return SplitMatrix(matrices)


def _prepare_out_array(out: Optional[np.ndarray], out_shape, out_dtype) -> np.ndarray:
    if out is None:
        out = np.zeros(out_shape, out_dtype)
    else:
        # TODO: make this a re-usable method that all the matrix classes
        # can use to check their out parameter
        if out.dtype != out_dtype:
            raise ValueError(
                f"out array is required to have dtype {out_dtype} but has"
                f"dtype {out.dtype}"
            )
    return out


def _filter_out_empty(matrices, indices):
    keep_idxs = [i for i, m in enumerate(matrices) if m.shape[1] > 0]
    out_mats = [matrices[i] for i in keep_idxs]
    out_idxs = [indices[i] for i in keep_idxs]
    return out_mats, out_idxs


def _combine_matrices(matrices, indices):
    """
    Combine multiple SparseMatrix and DenseMatrix objects into a single object of each
    type.

    ``matrices`` is  and ``indices`` marks which columns they correspond to.
    Categorical matrices remain unmodified by this function since categorical
    matrices cannot be combined (each categorical matrix represents a single category).

    Parameters
    ----------
    matrices:
        The MatrixBase matrices to be combined.

    indices:
        The columns the each matrix corresponds to.
    """
    n_row = matrices[0].shape[0]

    for mat_type_, stack_fn in [
        (
            DenseMatrix,
            lambda matrices: np.hstack([mat._array for mat in matrices]),
        ),
        (
            SparseMatrix,
            lambda matrices: sps.hstack([mat._array for mat in matrices]),
        ),
    ]:
        this_type_matrices = [
            i for i, mat in enumerate(matrices) if isinstance(mat, mat_type_)
        ]
        if len(this_type_matrices) > 1:
            new_matrix = mat_type_(stack_fn([matrices[i] for i in this_type_matrices]))
            new_indices = np.concatenate([indices[i] for i in this_type_matrices])
            new_colnames = np.concatenate(
                [np.array(matrices[i]._colnames) for i in this_type_matrices]
            )
            new_terms = np.concatenate(
                [np.array(matrices[i]._terms) for i in this_type_matrices]
            )
            sorter = np.argsort(new_indices)
            sorted_matrix = new_matrix[:, sorter]
            sorted_matrix._colnames = new_colnames[sorter].tolist()
            sorted_matrix._terms = new_terms[sorter].tolist()
            sorted_indices = new_indices[sorter]

            assert sorted_matrix.shape[0] == n_row
            matrices[this_type_matrices[0]] = sorted_matrix
            indices[this_type_matrices[0]] = sorted_indices
            indices = [
                idx for i, idx in enumerate(indices) if i not in this_type_matrices[1:]
            ]
            matrices = [
                mat for i, mat in enumerate(matrices) if i not in this_type_matrices[1:]
            ]
    return matrices, indices


class SplitMatrix(MatrixBase):
    """
    A class for matrices with sparse, dense and categorical parts.

    For real-world tabular data, it's common for the same dataset to contain a mix
    of columns that are naturally dense, naturally sparse and naturally categorical.
    Representing each of these sets of columns in the format that is most natural
    allows for a significant speedup in matrix multiplications compared to
    representations that are entirely dense or entirely sparse.

    Initialize a SplitMatrix directly with a list of ``matrices`` and a
    list of column ``indices`` for each matrix.
    Most of the time, it will
    be best to use :func:`tabmat.from_pandas` or :func:`tabmat.from_csc`
    to initialize a ``SplitMatrix``.

    Parameters
    ----------
    matrices:
        The sub-matrices composing the columns of this SplitMatrix.

    indices:
        If ``indices`` is not None, then for each matrix passed in
        ``matrices``, ``indices`` must contain the set of columns which that matrix
        covers.
    """

    def __init__(
        self,
        matrices: Sequence[MatrixBase],
        indices: Optional[list[np.ndarray]] = None,
    ):
        flatten_matrices = []
        index_corrections = []
        # First check that all matrices are valid types
        for mat in matrices:
            if not isinstance(mat, MatrixBase):
                raise ValueError(
                    "Expected all elements of matrices to be subclasses of MatrixBase."
                )
            if isinstance(mat, SplitMatrix):
                # Flatten out the SplitMatrix
                current_idx = 0
                for iind, imat in zip(mat.indices, mat.matrices):  # type: ignore
                    flatten_matrices.append(imat)
                    index_corrections.append(
                        iind - np.arange(len(iind), dtype=np.int64) - current_idx
                    )
                    current_idx += len(iind)
            else:
                flatten_matrices.append(mat)
                if len(mat.shape) == 1:
                    index_corrections.append(np.zeros(1, dtype=np.int64))
                else:
                    index_corrections.append(np.zeros(mat.shape[1], dtype=np.int64))

        # Now that we know these are all MatrixBase, we can check consistent
        # shapes and dtypes.
        self.dtype = flatten_matrices[0].dtype
        n_row = flatten_matrices[0].shape[0]
        for i, mat in enumerate(flatten_matrices):
            if mat.dtype != self.dtype:
                warnings.warn(
                    "Matrices do not all have the same dtype. Dtypes are "
                    f"{[elt.dtype for elt in flatten_matrices]}."
                )
            if not mat.shape[0] == n_row:
                raise ValueError(
                    "All matrices should have the same first dimension, "
                    f"but the first matrix has first dimension {n_row} and matrix {i} "
                    f"has first dimension {mat.shape[0]}."
                )
            if mat.ndim == 1:
                flatten_matrices[i] = mat[:, np.newaxis]
            elif mat.ndim > 2:
                raise ValueError("All matrices should be at most two dimensional.")

        if indices is None:
            indices = []
            current_idx = 0
            for mat, ind_corr in zip(flatten_matrices, index_corrections):
                indices.append(
                    np.arange(current_idx, current_idx + mat.shape[1], dtype=np.int64)
                    + ind_corr
                )
                current_idx += mat.shape[1]
            n_col = current_idx
        else:
            all_indices = np.concatenate(indices)
            n_col = len(all_indices)

            if (np.arange(n_col, dtype=np.int64) != np.sort(all_indices)).any():
                raise ValueError(
                    "Indices should contain all integers from 0 to one less than the "
                    "number of columns."
                )

            for i in range(len(indices)):
                indices[i] = np.asarray(indices[i])
                if not is_sorted(indices[i]):
                    raise ValueError(
                        f"Each index block should be sorted, but indices[{i}] was "
                        "not sorted"
                    )

        assert isinstance(indices, list)

        for i, (mat, idx) in enumerate(zip(flatten_matrices, indices)):
            if not mat.shape[1] == len(idx):
                raise ValueError(
                    f"Element {i} of indices should should have length {mat.shape[1]}, "
                    f"but it has shape {idx.shape}"
                )

        filtered_mats, filtered_idxs = _filter_out_empty(flatten_matrices, indices)
        combined_matrices, combined_indices = _combine_matrices(
            filtered_mats, filtered_idxs
        )

        self.matrices = combined_matrices
        self.indices = [np.asarray(elt, dtype=np.int64) for elt in combined_indices]
        self.shape = (n_row, n_col)

        assert self.shape[1] > 0

    def _split_col_subsets(
        self, cols: Optional[np.ndarray]
    ) -> tuple[list[np.ndarray], Union[list[np.ndarray], list[None]], int]:
        """
        Return tuple of things helpful for applying column restrictions to sub-matrices.

        - subset_cols_indices
        - subset_cols
        - n_cols

        Outputs obey
            self.indices[i][subset_cols[i]] == cols[subset_cols_indices[i]]
        for all i when cols is not None, and
            mat.indices[i] == subset_cols_indices[i]
        when cols is None.
        """
        if cols is None:
            subset_cols_indices = self.indices
            subset_cols = [None for _ in range(len(self.indices))]
            return subset_cols_indices, subset_cols, self.shape[1]

        cols = set_up_rows_or_cols(cols, self.shape[1])
        return split_col_subsets(self, cols)

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """Return SplitMatrix cast to new type."""
        if copy:
            new_matrices = [
                mat.astype(dtype=dtype, order=order, casting=casting, copy=True)
                for mat in self.matrices
            ]
            return SplitMatrix(new_matrices, self.indices)
        for i in range(len(self.matrices)):
            self.matrices[i] = self.matrices[i].astype(
                dtype=dtype, order=order, casting=casting, copy=False
            )
        return SplitMatrix(self.matrices, self.indices)

    def toarray(self) -> np.ndarray:
        """Return array representation of matrix."""
        out = np.empty(self.shape)
        for mat, idx in zip(self.matrices, self.indices):
            out[:, idx] = mat.toarray()
        return out

    def getcol(self, i: int) -> Union[np.ndarray, sps.csr_matrix]:
        """Return matrix column at specified index."""
        # wrap-around indexing
        i %= self.shape[1]
        for mat, idx in zip(self.matrices, self.indices):
            if i in idx:
                loc = np.where(idx == i)[0][0]
                return mat.getcol(loc)
        raise RuntimeError(f"Column {i} was not found.")

    def sandwich(
        self,
        d: Union[np.ndarray, list],
        rows: Optional[np.ndarray] = None,
        cols: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Perform a sandwich product: X.T @ diag(d) @ X."""
        d = np.asarray(d)
        check_sandwich_compatible(self, d)

        subset_cols_indices, subset_cols, n_cols = self._split_col_subsets(cols)

        out = np.zeros((n_cols, n_cols))
        for i in range(len(self.indices)):
            idx_i = subset_cols_indices[i]
            mat_i = self.matrices[i]
            res = mat_i.sandwich(d, rows, subset_cols[i])
            if isinstance(res, sps.dia_matrix):
                out[(idx_i, idx_i)] += np.squeeze(res.data)
            else:
                out[np.ix_(idx_i, idx_i)] = res

            for j in range(i + 1, len(self.indices)):
                idx_j = subset_cols_indices[j]
                mat_j = self.matrices[j]
                res = mat_i._cross_sandwich(
                    mat_j, d, rows, subset_cols[i], subset_cols[j]
                )

                out[np.ix_(idx_i, idx_j)] = res
                out[np.ix_(idx_j, idx_i)] = res.T

        return out

    def _get_col_means(self, weights: np.ndarray) -> np.ndarray:
        """Get means of columns."""
        col_means = np.empty(self.shape[1], dtype=self.dtype)
        for idx, mat in zip(self.indices, self.matrices):
            col_means[idx] = mat._get_col_means(weights)
        return col_means

    def _get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        """Get standard deviations of columns."""
        col_stds = np.empty(self.shape[1], dtype=self.dtype)
        for idx, mat in zip(self.indices, self.matrices):
            col_stds[idx] = mat._get_col_stds(weights, col_means[idx])

        return col_stds

    def matvec(
        self,
        v: np.ndarray,
        cols: Optional[np.ndarray] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Perform self[:, cols] @ other[cols]."""
        assert not isinstance(v, sps.spmatrix)
        v = np.asarray(v)
        check_matvec_dimensions(self, v, transpose=False)
        check_matvec_out_shape(self, out)

        if v.shape[0] != self.shape[1]:
            raise ValueError(f"shapes {self.shape} and {v.shape} not aligned")

        _, subset_cols, n_cols = self._split_col_subsets(cols)

        out_shape = [self.shape[0]] + ([] if v.ndim == 1 else list(v.shape[1:]))
        out_dtype = np.result_type(self.dtype, v.dtype)

        # If there is a dense matrix in the list of matrices, we want to
        # multiply that one first for memory use reasons. This is because numpy
        # doesn't provide a blas-like mechanism for specifying that we want to
        # add the result of the matrix-vector product into an existing array.
        # So, we simply use the output of the first dense matrix-vector product
        # as the target for storing the final output. This reduces the number
        # of output arrays allocated from 2 to 1.
        is_matrix_dense = [isinstance(m, DenseMatrix) for m in self.matrices]
        dense_matrix_idx: Union[int, np.intp]
        if np.any(is_matrix_dense):
            dense_matrix_idx = np.argmax(is_matrix_dense)
            sub_cols = subset_cols[dense_matrix_idx]
            idx = self.indices[dense_matrix_idx]
            mat = self.matrices[dense_matrix_idx]
            in_vec = v[idx, ...]
            out = np.asarray(mat.matvec(in_vec, sub_cols, out), dtype=out_dtype)
        else:
            dense_matrix_idx = -1
            out = _prepare_out_array(out, out_shape, out_dtype)

        for i, (sub_cols, idx, mat) in enumerate(
            zip(subset_cols, self.indices, self.matrices)
        ):
            if i == dense_matrix_idx:
                continue
            in_vec = v[idx, ...]
            mat.matvec(in_vec, sub_cols, out=out)
        return out  # type: ignore

    def transpose_matvec(
        self,
        v: Union[np.ndarray, list],
        rows: Optional[np.ndarray] = None,
        cols: Optional[np.ndarray] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Perform: self[rows, cols].T @ vec[rows].

        ::

            self.transpose_matvec(v, rows, cols) = self[rows, cols].T @ v[rows]
            self.transpose_matvec(v, rows, cols)[i]
                = sum_{j in rows} self[j, cols[i]] v[j]
                = sum_{j in rows} sum_{mat in self.matrices} 1(cols[i] in mat)
                                                            self[j, cols[i]] v[j]
        """

        v = np.asarray(v)
        check_matvec_dimensions(self, v, transpose=True)
        check_transpose_matvec_out_shape(self, out)

        subset_cols_indices, subset_cols, n_cols = self._split_col_subsets(cols)

        out_shape = [n_cols] + list(v.shape[1:])
        out_dtype = np.result_type(self.dtype, v.dtype)
        out_is_none = out is None
        out = _prepare_out_array(out, out_shape, out_dtype)
        if cols is not None:
            cols = np.asarray(cols, dtype=np.int32)

        for idx, sub_cols, mat in zip(subset_cols_indices, subset_cols, self.matrices):
            res = mat.transpose_matvec(v, rows=rows, cols=sub_cols)
            if out_is_none or cols is None:
                out[idx, ...] += res
            else:
                out[cols[idx], ...] += res
        return out

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
        else:
            row = key
            col = slice(None, None, None)  # all columns

        if col == slice(None, None, None):
            if isinstance(row, int):
                row = [row]

            return SplitMatrix([mat[row, :] for mat in self.matrices], self.indices)
        else:
            raise NotImplementedError(
                f"Only row indexing is supported. Index passed was {key}."
            )

    def multiply(self, other):
        """Element-wise multiplication.

        This assumes that ``other`` is a vector of size ``self.shape[0]``.
        """
        return SplitMatrix(
            [mat.multiply(other) for mat in self.matrices], indices=self.indices
        )

    def __repr__(self):
        out = "SplitMatrix:"
        for i, mat in enumerate(self.matrices):
            out += (
                f"\n\nComponent {i} with type {mat.__class__.__name__}\n"
                + mat.__repr__()
            )
        return out

    __array_priority__ = 13

    def get_names(
        self,
        type: str = "column",
        missing_prefix: Optional[str] = None,
        indices: Optional[list[int]] = None,
    ) -> list[Optional[str]]:
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
        names = np.empty(self.shape[1], dtype=object)
        for idx, mat in zip(self.indices, self.matrices):
            names[idx] = mat.get_names(type, missing_prefix, idx)
        return names.tolist()

    def set_names(self, names: Union[str, list[Optional[str]]], type: str = "column"):
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
        names_array = np.array(names)

        if len(names) != self.shape[1]:
            raise ValueError(f"Length of names must be {self.shape[1]}")

        for idx, mat in zip(self.indices, self.matrices):
            mat.set_names(names_array[idx].tolist(), type)
