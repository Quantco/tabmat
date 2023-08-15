import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sps

import tabmat as tm
from tabmat.matrix_base import MatrixBase


def base_array(order="F") -> np.ndarray:
    return np.array([[0, 0], [0, -1.0], [0, 2.0]], order=order)


def dense_matrix_F() -> tm.DenseMatrix:
    return tm.DenseMatrix(base_array())


def dense_matrix_C() -> tm.DenseMatrix:
    return tm.DenseMatrix(base_array(order="C"))


def dense_matrix_not_writeable() -> tm.DenseMatrix:
    mat = dense_matrix_F()
    mat._array.setflags(write=False)
    return mat


def sparse_matrix() -> tm.SparseMatrix:
    return tm.SparseMatrix(sps.csc_matrix(base_array()))


def sparse_matrix_64() -> tm.SparseMatrix:
    csc = sps.csc_matrix(base_array())
    mat = tm.SparseMatrix(
        (csc.data, csc.indices.astype(np.int64), csc.indptr.astype(np.int64))
    )
    return mat


def categorical_matrix():
    vec = [1, 0, 1]
    return tm.CategoricalMatrix(vec)


def categorical_matrix_drop_first():
    vec = [0, 1, 2]
    return tm.CategoricalMatrix(vec, drop_first=True)


def get_unscaled_matrices() -> (
    List[Union[tm.DenseMatrix, tm.SparseMatrix, tm.CategoricalMatrix]]
):
    return [
        dense_matrix_F(),
        dense_matrix_C(),
        dense_matrix_not_writeable(),
        sparse_matrix(),
        sparse_matrix_64(),
        categorical_matrix(),
        categorical_matrix_drop_first(),
    ]


def complex_split_matrix():
    return tm.SplitMatrix(get_unscaled_matrices())


def shift_complex_split_matrix():
    mat = complex_split_matrix()
    np.random.seed(0)
    return tm.StandardizedMatrix(mat, np.random.random(mat.shape[1]))


def shift_scale_complex_split_matrix():
    mat = complex_split_matrix()
    np.random.seed(0)
    return tm.StandardizedMatrix(
        mat, np.random.random(mat.shape[1]), np.random.random(mat.shape[1])
    )


def get_all_matrix_base_subclass_mats():
    return get_unscaled_matrices() + [complex_split_matrix()]


def get_standardized_shifted_matrices():
    return [tm.StandardizedMatrix(elt, [0.3, 2]) for elt in get_unscaled_matrices()] + [
        shift_complex_split_matrix()
    ]


def get_standardized_shifted_scaled_matrices():
    return [
        tm.StandardizedMatrix(elt, [0.3, 0.2], [0.6, 1.67])
        for elt in get_unscaled_matrices()
    ] + [shift_scale_complex_split_matrix()]


def get_matrices():
    return (
        get_all_matrix_base_subclass_mats()
        + get_standardized_shifted_matrices()
        + get_standardized_shifted_scaled_matrices()
    )


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("cols", [None, [], [1], np.array([1])])
def test_matvec_out_parameter_wrong_shape(mat, cols):
    out = np.zeros(mat.shape[0] + 1)
    v = np.zeros(mat.shape[1])
    with pytest.raises(ValueError, match="first dimension of 'out' must be"):
        mat.matvec(v, cols, out)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("cols", [None, [], [1], np.array([1])])
@pytest.mark.parametrize("rows", [None, [], [1], np.array([1])])
def test_transpose_matvec_out_parameter_wrong_shape(mat, cols, rows):
    out = np.zeros(mat.shape[1] + 1)
    v = np.zeros(mat.shape[0])
    with pytest.raises(ValueError, match="dimension of 'out' must be"):
        mat.transpose_matvec(v, rows, cols, out)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("cols", [None, [], [1], np.array([1])])
def test_matvec_out_parameter(mat, cols):
    out = np.random.rand(mat.shape[0])
    out_copy = out.copy()
    v = np.random.rand(mat.shape[1])

    # This should modify out in place.
    out2 = mat.matvec(v, cols=cols, out=out)
    assert out.__array_interface__["data"][0] == out2.__array_interface__["data"][0]
    assert out.shape == out_copy.shape

    correct = out_copy + mat.matvec(v, cols=cols)
    np.testing.assert_almost_equal(out, out2)
    np.testing.assert_almost_equal(out, correct)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("cols", [None, [], [1], np.array([0, 1])])
@pytest.mark.parametrize("rows", [None, [], [1], np.array([0, 2])])
def test_transpose_matvec_out_parameter(mat, cols, rows):
    out = np.random.rand(mat.shape[1])
    out_copy = out.copy()
    v = np.random.rand(mat.shape[0])

    # This should modify out in place.
    out2 = mat.transpose_matvec(v, rows=rows, cols=cols, out=out)
    # Check that modification has been in-place
    assert out.__array_interface__["data"][0] == out2.__array_interface__["data"][0]
    assert out.shape == out_copy.shape

    col_idx = np.arange(mat.shape[1], dtype=int) if cols is None else cols
    row_idx = np.arange(mat.shape[0], dtype=int) if rows is None else rows
    matvec_part = mat.A[row_idx, :][:, col_idx].T.dot(v[row_idx])

    if cols is None:
        correct = out_copy + matvec_part
    else:
        correct = out_copy
        correct[cols] += matvec_part

    np.testing.assert_almost_equal(out, out2)
    np.testing.assert_almost_equal(out, correct)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("cols", [None, [], [1], np.array([0, 1])])
@pytest.mark.parametrize("rows", [None, [], [1], np.array([0, 2])])
def test_matvec_dimension_mismatch_raises(mat, rows, cols):
    too_short = np.ones(mat.shape[1] - 1, dtype=mat.dtype)
    just_right = np.ones(mat.shape[1], dtype=mat.dtype)
    too_long = np.ones(mat.shape[1] + 1, dtype=mat.dtype)
    mat.matvec(just_right, cols=cols)
    with pytest.raises(ValueError):
        mat.matvec(too_short, cols=cols)
    with pytest.raises(ValueError):
        mat.matvec(too_long, cols=cols)

    too_short_transpose = np.ones(mat.shape[0] - 1, dtype=mat.dtype)
    just_right_transpose = np.ones(mat.shape[0], dtype=mat.dtype)
    too_long_transpose = np.ones(mat.shape[0] + 1, dtype=mat.dtype)
    mat.transpose_matvec(just_right_transpose, rows=rows, cols=cols)
    with pytest.raises(ValueError):
        mat.transpose_matvec(too_short_transpose, rows=rows, cols=cols)
    with pytest.raises(ValueError):
        mat.transpose_matvec(too_long_transpose, rows=rows, cols=cols)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("i", [1, -2])
def test_getcol(mat: Union[tm.MatrixBase, tm.StandardizedMatrix], i):
    col = mat.getcol(i)

    if not isinstance(col, np.ndarray):
        col = col.A
    np.testing.assert_almost_equal(col, mat.A[:, [i]])


@pytest.mark.parametrize("mat", get_all_matrix_base_subclass_mats())
def test_to_array_matrix_base(mat: tm.MatrixBase):
    assert isinstance(mat.A, np.ndarray)
    if isinstance(mat, tm.CategoricalMatrix) and not mat.drop_first:
        expected = np.array([[0, 1], [1, 0], [0, 1]])
    elif isinstance(mat, tm.CategoricalMatrix) and mat.drop_first:
        expected = np.array([[0, 0], [1, 0], [0, 1]])
    elif isinstance(mat, tm.SplitMatrix):
        expected = np.hstack([elt.A for elt in mat.matrices])
    else:
        expected = base_array()
    np.testing.assert_allclose(mat.A, expected)


@pytest.mark.parametrize(
    "mat",
    get_standardized_shifted_matrices() + get_standardized_shifted_scaled_matrices(),
)
def test_to_array_standardized_mat(mat: tm.StandardizedMatrix):
    assert isinstance(mat.A, np.ndarray)
    true_mat_part = mat.mat.A
    if mat.mult is not None:
        true_mat_part = mat.mult[None, :] * mat.mat.A
    np.testing.assert_allclose(mat.A, true_mat_part + mat.shift)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize(
    "other_type",
    [lambda x: x, np.asarray],
)
@pytest.mark.parametrize("cols", [None, [], [1], np.array([1])])
@pytest.mark.parametrize("other_shape", [[], [1], [2]])
def test_matvec(
    mat: Union[tm.MatrixBase, tm.StandardizedMatrix], other_type, cols, other_shape
):
    """
    Mat.

    t: Function transforming list to list, array, or DenseMatrix
    cols: Argument 1 to matvec, specifying which columns of the matrix (and
        which elements of 'other') to use
    other_shape: Second dimension of 'other.shape', if any. If other_shape is [], then
        other is 1d.
    """
    n_row = mat.shape[1]
    shape = [n_row] + other_shape
    other_as_list = np.random.random(shape).tolist()
    other = other_type(other_as_list)

    def is_split_with_cat_part(x):
        return isinstance(x, tm.SplitMatrix) and any(
            isinstance(elt, tm.CategoricalMatrix) for elt in x.matrices
        )

    has_categorical_component = (
        isinstance(mat, tm.CategoricalMatrix)
        or is_split_with_cat_part(mat)
        or (
            isinstance(mat, tm.StandardizedMatrix)
            and (
                isinstance(mat.mat, tm.CategoricalMatrix)
                or is_split_with_cat_part(mat.mat)
            )
        )
    )

    if has_categorical_component and len(shape) > 1:
        with pytest.raises(NotImplementedError, match="only implemented for 1d"):
            mat.matvec(other, cols)
    else:
        res = mat.matvec(other, cols)

        mat_subset, vec_subset = process_mat_vec_subsets(mat, other, None, cols, cols)
        expected = mat_subset.dot(vec_subset)

        np.testing.assert_allclose(res, expected)
        assert isinstance(res, np.ndarray)

        if cols is None:
            res2 = mat @ other
            np.testing.assert_allclose(res2, expected)


def process_mat_vec_subsets(mat, vec, mat_rows, mat_cols, vec_idxs):
    mat_subset = mat.A
    vec_subset = vec
    if mat_rows is not None:
        mat_subset = mat_subset[mat_rows, :]
    if mat_cols is not None:
        mat_subset = mat_subset[:, mat_cols]
    if vec_idxs is not None:
        vec_subset = np.array(vec_subset)[vec_idxs]
    return mat_subset, vec_subset


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize(
    "other_type",
    [lambda x: x, np.array],
)
@pytest.mark.parametrize("rows", [None, [], [2], np.arange(2)])
@pytest.mark.parametrize("cols", [None, [], [1], np.arange(1)])
def test_transpose_matvec(
    mat: Union[tm.MatrixBase, tm.StandardizedMatrix], other_type, rows, cols
):
    other_as_list = [3.0, -0.1, 0]
    other = other_type(other_as_list)
    assert np.shape(other)[0] == mat.shape[0]
    res = mat.transpose_matvec(other, rows, cols)

    mat_subset, vec_subset = process_mat_vec_subsets(
        mat, other_as_list, rows, cols, rows
    )
    expected = mat_subset.T.dot(vec_subset)
    np.testing.assert_allclose(res, expected)
    assert isinstance(res, np.ndarray)


@pytest.mark.parametrize(
    "mat_i, mat_j",
    [
        (dense_matrix_C(), sparse_matrix()),
        (dense_matrix_C(), sparse_matrix_64()),
        (dense_matrix_C(), categorical_matrix()),
        (dense_matrix_F(), sparse_matrix()),
        (dense_matrix_F(), sparse_matrix_64()),
        (dense_matrix_F(), categorical_matrix()),
        (dense_matrix_not_writeable(), sparse_matrix()),
        (dense_matrix_not_writeable(), sparse_matrix_64()),
        (dense_matrix_not_writeable(), categorical_matrix()),
        (sparse_matrix(), dense_matrix_C()),
        (sparse_matrix(), dense_matrix_F()),
        (sparse_matrix(), dense_matrix_not_writeable()),
        (sparse_matrix(), categorical_matrix()),
        (sparse_matrix_64(), dense_matrix_C()),
        (sparse_matrix_64(), dense_matrix_F()),
        (sparse_matrix_64(), dense_matrix_not_writeable()),
        (sparse_matrix_64(), categorical_matrix()),
        (categorical_matrix(), dense_matrix_C()),
        (categorical_matrix(), dense_matrix_F()),
        (categorical_matrix(), dense_matrix_not_writeable()),
        (categorical_matrix(), sparse_matrix()),
        (categorical_matrix(), sparse_matrix_64()),
        (categorical_matrix(), categorical_matrix()),
    ],
)
@pytest.mark.parametrize("rows", [None, [2], np.arange(2)])
@pytest.mark.parametrize("L_cols", [None, [1], np.arange(1)])
@pytest.mark.parametrize("R_cols", [None, [1], np.arange(1)])
def test_cross_sandwich(
    mat_i: Union[tm.DenseMatrix, tm.SparseMatrix, tm.CategoricalMatrix],
    mat_j: Union[tm.DenseMatrix, tm.SparseMatrix, tm.CategoricalMatrix],
    rows: Optional[np.ndarray],
    L_cols: Optional[np.ndarray],
    R_cols: Optional[np.ndarray],
):
    assert mat_i.shape[0] == mat_j.shape[0]
    d = np.random.random(mat_i.shape[0])
    mat_i_, _ = process_mat_vec_subsets(mat_i, None, rows, L_cols, None)
    mat_j_, d_ = process_mat_vec_subsets(mat_j, d, rows, R_cols, rows)
    expected = mat_i_.T @ np.diag(d_) @ mat_j_
    res = mat_i._cross_sandwich(mat_j, d, rows, L_cols, R_cols)
    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize(
    "vec_type",
    [lambda x: x, np.array],
)
@pytest.mark.parametrize("rows", [None, [], [1], np.arange(2)])
@pytest.mark.parametrize("cols", [None, [], [0], np.arange(1)])
def test_self_sandwich(
    mat: Union[tm.MatrixBase, tm.StandardizedMatrix], vec_type, rows, cols
):
    vec_as_list = [3, 0.1, 1]
    vec = vec_type(vec_as_list)
    res = mat.sandwich(vec, rows, cols)

    mat_subset, vec_subset = process_mat_vec_subsets(mat, vec_as_list, rows, cols, rows)
    expected = mat_subset.T @ np.diag(vec_subset) @ mat_subset
    if sps.issparse(res):
        res = res.A
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("rows", [None, [], [0], np.arange(2)])
@pytest.mark.parametrize("cols", [None, [], [0], np.arange(1)])
def test_split_sandwich(rows: Optional[np.ndarray], cols: Optional[np.ndarray]):
    mat = complex_split_matrix()
    d = np.random.random(mat.shape[0])
    result = mat.sandwich(d, rows=rows, cols=cols)

    mat_as_dense = mat.A
    d_rows = d
    if rows is not None:
        mat_as_dense = mat_as_dense[rows, :]
        d_rows = d[rows]
    if cols is not None:
        mat_as_dense = mat_as_dense[:, cols]

    expected = mat_as_dense.T @ np.diag(d_rows) @ mat_as_dense
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "mat",
    [
        dense_matrix_F(),
        dense_matrix_C(),
        dense_matrix_not_writeable(),
        sparse_matrix(),
        sparse_matrix_64(),
    ],
)
def test_transpose(mat):
    res = mat.T.A
    expected = mat.A.T
    assert res.shape == (mat.shape[1], mat.shape[0])
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize(
    "vec_type",
    [lambda x: x, np.array],
)
def test_rmatmul(mat: Union[tm.MatrixBase, tm.StandardizedMatrix], vec_type):
    vec_as_list = [3.0, -0.1, 0]
    vec = vec_type(vec_as_list)
    res = mat.__rmatmul__(vec)
    res2 = vec @ mat
    expected = vec_as_list @ mat.A
    np.testing.assert_allclose(res, expected)
    np.testing.assert_allclose(res2, expected)
    assert isinstance(res, (np.ndarray, tm.DenseMatrix))


@pytest.mark.parametrize("mat", get_matrices())
def test_matvec_raises(mat: Union[tm.MatrixBase, tm.StandardizedMatrix]):
    with pytest.raises(ValueError):
        mat.matvec(np.ones(11))


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_astype(mat: Union[tm.MatrixBase, tm.StandardizedMatrix], dtype):
    new_mat = mat.astype(dtype)
    assert np.issubdtype(new_mat.dtype, dtype)
    vec = np.zeros(mat.shape[1], dtype=dtype)
    res = new_mat.matvec(vec)
    assert res.dtype == new_mat.dtype


@pytest.mark.parametrize("mat", get_all_matrix_base_subclass_mats())
def test_get_col_means(mat: tm.MatrixBase):
    weights = np.random.random(mat.shape[0])
    # TODO: make weights sum to 1 within functions
    weights /= weights.sum()
    means = mat._get_col_means(weights)
    expected = mat.A.T.dot(weights)
    np.testing.assert_allclose(means, expected)


@pytest.mark.parametrize("mat", get_all_matrix_base_subclass_mats())
def test_get_col_means_unweighted(mat: tm.MatrixBase):
    weights = np.ones(mat.shape[0])
    # TODO: make weights sum to 1 within functions
    weights /= weights.sum()
    means = mat._get_col_means(weights)
    expected = mat.A.mean(0)
    np.testing.assert_allclose(means, expected)


@pytest.mark.parametrize("mat", get_all_matrix_base_subclass_mats())
def test_get_col_stds(mat: tm.MatrixBase):
    weights = np.random.random(mat.shape[0])
    # TODO: make weights sum to 1
    weights /= weights.sum()
    means = mat._get_col_means(weights)
    expected = np.sqrt((mat.A**2).T.dot(weights) - means**2)
    stds = mat._get_col_stds(weights, means)
    np.testing.assert_allclose(stds, expected)


@pytest.mark.parametrize("mat", get_unscaled_matrices())
def test_get_col_stds_unweighted(mat: tm.MatrixBase):
    weights = np.ones(mat.shape[0])
    # TODO: make weights sum to 1
    weights /= weights.sum()
    means = mat._get_col_means(weights)
    expected = mat.A.std(0)
    stds = mat._get_col_stds(weights, means)
    np.testing.assert_allclose(stds, expected)


@pytest.mark.parametrize("mat", get_unscaled_matrices())
@pytest.mark.parametrize("center_predictors", [False, True])
@pytest.mark.parametrize("scale_predictors", [False, True])
def test_standardize(
    mat: tm.MatrixBase, center_predictors: bool, scale_predictors: bool
):
    asarray = mat.A.copy()
    weights = np.random.rand(mat.shape[0])
    weights /= weights.sum()

    true_means = asarray.T.dot(weights)
    true_sds = np.sqrt((asarray**2).T.dot(weights) - true_means**2)

    standardized, means, stds = mat.standardize(
        weights, center_predictors, scale_predictors
    )
    assert isinstance(standardized, tm.StandardizedMatrix)
    assert isinstance(standardized.mat, type(mat))
    if center_predictors:
        np.testing.assert_allclose(
            standardized.transpose_matvec(weights), 0, atol=1e-11
        )
        np.testing.assert_allclose(means, asarray.T.dot(weights))
    else:
        np.testing.assert_almost_equal(means, 0)

    if scale_predictors:
        np.testing.assert_allclose(stds, true_sds)
    else:
        assert stds is None

    expected_sds = true_sds if scale_predictors else np.ones_like(true_sds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        one_over_sds = np.nan_to_num(1 / expected_sds)

    expected_mat = asarray * one_over_sds
    if center_predictors:
        expected_mat -= true_means * one_over_sds
    np.testing.assert_allclose(standardized.A, expected_mat)

    unstandardized = standardized.unstandardize()
    assert isinstance(unstandardized, type(mat))
    np.testing.assert_allclose(unstandardized.A, asarray)


@pytest.mark.parametrize("mat", get_matrices())
def test_indexing_int_row(mat: Union[tm.MatrixBase, tm.StandardizedMatrix]):
    res = mat[0, :]
    if not isinstance(res, np.ndarray):
        res = res.A
    expected = mat.A[[0], :]
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", get_matrices())
def test_indexing_range_row(mat: Union[tm.MatrixBase, tm.StandardizedMatrix]):
    res = mat[0:2, :]
    assert res.ndim == 2
    if not isinstance(res, np.ndarray):
        res = res.A
    expected = mat.A[0:2, :]
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("mat", get_unscaled_matrices())
def test_indexing_int_col(mat):
    res = mat[:, 0]
    if not isinstance(res, np.ndarray):
        res = res.A
    assert res.shape == (mat.shape[0], 1)
    expected = mat.A[:, [0]]
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("mat", get_unscaled_matrices())
def test_indexing_range_col(mat):
    res = mat[:, 0:2]
    if not isinstance(res, np.ndarray):
        res = res.A
    assert res.shape == (mat.shape[0], 2)
    expected = mat.A[:, 0:2]
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("mat", get_unscaled_matrices())
def test_indexing_int_both(mat):
    res = mat[0, 0]
    if not isinstance(res, np.ndarray):
        res = res.A
    assert res.shape == (1, 1)
    expected = mat.A[0, 0]
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("mat", get_unscaled_matrices())
def test_indexing_seq_both(mat):
    res = mat[[0, 1], [0, 1]]
    if not isinstance(res, np.ndarray):
        res = res.A
    assert res.shape == (2, 2)
    expected = mat.A[np.ix_([0, 1], [0, 1])]
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("mat", get_unscaled_matrices())
def test_indexing_ix_both(mat):
    indexer = np.ix_([0, 1], [0, 1])
    res = mat[indexer]
    if not isinstance(res, np.ndarray):
        res = res.A
    assert res.shape == (2, 2)
    expected = mat.A[indexer]
    np.testing.assert_array_equal(res, expected)


def test_pandas_to_matrix():
    n_rows = 50
    dense_column = np.linspace(-10, 10, num=n_rows, dtype=np.float64)
    dense_column_with_lots_of_zeros = dense_column.copy()
    dense_column_with_lots_of_zeros[:44] = 0.0
    sparse_column = np.zeros(n_rows, dtype=np.float64)
    sparse_column[0] = 1.0
    cat_column_lowdim = np.tile(["a", "b"], n_rows // 2)
    cat_column_highdim = np.arange(n_rows)

    dense_ser = pd.Series(dense_column)
    lowdense_ser = pd.Series(dense_column_with_lots_of_zeros)
    sparse_ser = pd.Series(sparse_column, dtype=pd.SparseDtype("float", 0.0))
    cat_ser_lowdim = pd.Categorical(cat_column_lowdim)
    cat_ser_highdim = pd.Categorical(cat_column_highdim)

    df = pd.DataFrame(
        data={
            "d": dense_ser,
            "ds": lowdense_ser,
            "s": sparse_ser,
            "cl_obj": cat_ser_lowdim.astype(object),
            "ch": cat_ser_highdim,
        }
    )

    mat = tm.from_pandas(
        df, dtype=np.float64, sparse_threshold=0.3, cat_threshold=4, object_as_cat=True
    )

    assert mat.shape == (n_rows, n_rows + 5)
    assert len(mat.matrices) == 3
    assert isinstance(mat, tm.SplitMatrix)

    nb_col_by_type = {
        tm.DenseMatrix: 3,  # includes low-dimension categorical
        tm.SparseMatrix: 2,  # sparse column
        tm.CategoricalMatrix: n_rows,
    }
    for submat in mat.matrices:
        assert submat.shape[1] == nb_col_by_type[type(submat)]

    # Prevent a regression where the column type of sparsified dense columns
    # was being changed in place.
    assert df["cl_obj"].dtype == object
    assert df["ds"].dtype == np.float64


@pytest.mark.parametrize("mat", get_all_matrix_base_subclass_mats())
def test_split_matrix_creation(mat):
    sm = tm.SplitMatrix(matrices=[mat, mat])
    assert sm.shape[0] == mat.shape[0]
    assert sm.shape[1] == 2 * mat.shape[1]


@pytest.mark.parametrize("mat", get_matrices())
def test_multiply(mat):
    other = np.arange(mat.shape[0])
    expected = mat.A * other[:, np.newaxis]
    actual = []
    actual.append(mat.multiply(other))
    actual.append(mat.multiply(other[:, np.newaxis]))

    for act in actual:
        assert isinstance(act, MatrixBase)
        np.testing.assert_allclose(act.A, expected)


@pytest.mark.parametrize(
    "mat_1",
    get_all_matrix_base_subclass_mats()
    + [base_array()]
    + [sps.csc_matrix(base_array())],
)
@pytest.mark.parametrize(
    "mat_2",
    get_all_matrix_base_subclass_mats()
    + [base_array()]
    + [sps.csc_matrix(base_array())],
)
def test_hstack(mat_1, mat_2):
    mats = [mat_1, mat_2]
    stacked = tm.hstack(mats)

    if all(isinstance(mat, (np.ndarray, tm.DenseMatrix)) for mat in mats):
        assert isinstance(stacked, tm.DenseMatrix)
    elif all(isinstance(mat, (sps.csc_matrix, tm.SparseMatrix)) for mat in mats):
        assert isinstance(stacked, tm.SparseMatrix)
    else:
        assert isinstance(stacked, tm.SplitMatrix)

    np.testing.assert_array_equal(
        stacked.A,
        np.hstack([mat.A if not isinstance(mat, np.ndarray) else mat for mat in mats]),
    )


def test_names_against_expectation():
    X = tm.DenseMatrix(
        np.ones((5, 2)), column_names=["a", None], term_names=["a", None]
    )
    Xc = tm.CategoricalMatrix(
        pd.Categorical(["a", "b", "c", "b", "a"]), column_name="c", term_name="c"
    )
    Xc2 = tm.CategoricalMatrix(pd.Categorical(["a", "b", "c", "b", "a"]))
    Xs = tm.SparseMatrix(
        sps.csc_matrix(np.ones((5, 2))),
        column_names=["s1", "s2"],
        term_names=["s", "s"],
    )

    mat = tm.SplitMatrix(matrices=[X, Xc, Xc2, Xs])

    assert mat.get_names(type="column") == [
        "a",
        None,
        "c[a]",
        "c[b]",
        "c[c]",
        None,
        None,
        None,
        "s1",
        "s2",
    ]

    assert mat.get_names(type="term") == [
        "a",
        None,
        "c",
        "c",
        "c",
        None,
        None,
        None,
        "s",
        "s",
    ]

    assert mat.get_names(type="column", missing_prefix="_col_") == [
        "a",
        "_col_1",
        "c[a]",
        "c[b]",
        "c[c]",
        "_col_5-7[a]",
        "_col_5-7[b]",
        "_col_5-7[c]",
        "s1",
        "s2",
    ]

    assert mat.get_names(type="term", missing_prefix="_col_") == [
        "a",
        "_col_1",
        "c",
        "c",
        "c",
        "_col_5-7",
        "_col_5-7",
        "_col_5-7",
        "s",
        "s",
    ]


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("missing_prefix", ["_col_", "X"])
def test_names_getter_setter(mat, missing_prefix):
    names = mat.get_names(missing_prefix=missing_prefix, type="column")
    mat.column_names = names
    assert mat.column_names == names


@pytest.mark.parametrize("mat", get_matrices())
@pytest.mark.parametrize("missing_prefix", ["_col_", "X"])
def test_terms_getter_setter(mat, missing_prefix):
    names = mat.get_names(missing_prefix=missing_prefix, type="term")
    mat.term_names = names
    assert mat.term_names == names


@pytest.mark.parametrize("indexer_1", [slice(None, None), 0, slice(2, 8)])
@pytest.mark.parametrize("indexer_2", [[0], slice(1, 4), [0, 2, 3], [4, 3, 2, 1, 0]])
@pytest.mark.parametrize("sparse", [True, False])
def test_names_indexing(indexer_1, indexer_2, sparse):
    X = np.ones((10, 5), dtype=np.float64)
    colnames = ["a", "b", None, "d", "e"]
    termnames = ["t1", "t1", None, "t4", "t5"]

    colnames_array = np.array(colnames)
    termnames_array = np.array(termnames)

    if sparse:
        X = tm.SparseMatrix(
            sps.csc_matrix(X), column_names=colnames, term_names=termnames
        )
    else:
        X = tm.DenseMatrix(X, column_names=colnames, term_names=termnames)

    X_indexed = X[indexer_1, indexer_2]
    if not isinstance(X_indexed, tm.MatrixBase):
        pytest.skip("Does not return MatrixBase")
    assert X_indexed.column_names == list(colnames_array[indexer_2])
    assert X_indexed.term_names == list(termnames_array[indexer_2])


@pytest.mark.parametrize("mat_1", get_all_matrix_base_subclass_mats())
@pytest.mark.parametrize("mat_2", get_all_matrix_base_subclass_mats())
def test_combine_names(mat_1, mat_2):
    mat_1.column_names = mat_1.get_names(missing_prefix="m1_", type="column")
    mat_2.column_names = mat_2.get_names(missing_prefix="m2_", type="column")

    mat_1.term_names = mat_1.get_names(missing_prefix="m1_", type="term")
    mat_2.term_names = mat_2.get_names(missing_prefix="m2_", type="term")

    combined = tm.SplitMatrix(matrices=[mat_1, mat_2])

    assert combined.column_names == mat_1.column_names + mat_2.column_names
    assert combined.term_names == mat_1.term_names + mat_2.term_names


@pytest.mark.parametrize("prefix_sep", ["_", ": "])
@pytest.mark.parametrize("drop_first", [True, False])
def test_names_pandas(prefix_sep, drop_first):
    n_rows = 50
    dense_column = np.linspace(-10, 10, num=n_rows, dtype=np.float64)
    dense_column_with_lots_of_zeros = dense_column.copy()
    dense_column_with_lots_of_zeros[:44] = 0.0
    sparse_column = np.zeros(n_rows, dtype=np.float64)
    sparse_column[0] = 1.0
    cat_column_lowdim = np.tile(["a", "b"], n_rows // 2)
    cat_column_highdim = np.arange(n_rows)

    dense_ser = pd.Series(dense_column)
    lowdense_ser = pd.Series(dense_column_with_lots_of_zeros)
    sparse_ser = pd.Series(sparse_column, dtype=pd.SparseDtype("float", 0.0))
    cat_ser_lowdim = pd.Categorical(cat_column_lowdim)
    cat_ser_highdim = pd.Categorical(cat_column_highdim)

    df = pd.DataFrame(
        data={
            "d": dense_ser,
            "cl_obj": cat_ser_lowdim.astype(object),
            "ch": cat_ser_highdim,
            "ds": lowdense_ser,
            "s": sparse_ser,
        }
    )

    categorical_format = "{name}" + prefix_sep + "{category}"
    mat_end = tm.from_pandas(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        object_as_cat=True,
        cat_position="end",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    expanded_df = pd.get_dummies(df, prefix_sep=prefix_sep, drop_first=drop_first)
    assert mat_end.column_names == expanded_df.columns.tolist()

    mat_expand = tm.from_pandas(
        df,
        dtype=np.float64,
        sparse_threshold=0.3,
        cat_threshold=4,
        object_as_cat=True,
        cat_position="expand",
        categorical_format=categorical_format,
        drop_first=drop_first,
    )

    unique_terms = list(dict.fromkeys(mat_expand.term_names))
    assert unique_terms == df.columns.tolist()
