from typing import Optional, Tuple

import numpy as np


def set_up_rows_or_cols(
    arr: Optional[np.ndarray], length: int, dtype=np.int32
) -> np.ndarray:
    """Set up rows or columns using input array and input length."""
    if arr is None:
        return np.arange(length, dtype=dtype)
    return np.asarray(arr).astype(dtype)


def setup_restrictions(
    shape: Tuple[int, int],
    rows: Optional[np.ndarray],
    cols: Optional[np.ndarray],
    dtype=np.int32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set up row and column restrictions."""
    rows = set_up_rows_or_cols(rows, shape[0], dtype)
    cols = set_up_rows_or_cols(cols, shape[1], dtype)
    return rows, cols


def _check_out_shape(out: Optional[np.ndarray], expected_first_dim: int) -> None:
    if out is not None and out.shape[0] != expected_first_dim:
        raise ValueError(
            f"""The first dimension of 'out' must be {expected_first_dim}, but it is
            {out.shape[0]}."""
        )


def check_transpose_matvec_out_shape(mat, out: Optional[np.ndarray]) -> None:
    """Assert that the first dimension of the transpose_matvec output is correct."""
    _check_out_shape(out, mat.shape[1])


def check_matvec_out_shape(mat, out: Optional[np.ndarray]) -> None:
    """Assert that the first dimension of the matvec output is correct."""
    _check_out_shape(out, mat.shape[0])


def _get_expected_axis_length(orig_length, indexer):
    if isinstance(indexer, int):
        return 1
    if isinstance(indexer, list):
        return len(indexer)
    elif isinstance(indexer, np.ndarray):
        assert indexer.ndim < 2
        return len(indexer)
    elif isinstance(indexer, slice):
        return len(range(*indexer.indices(orig_length)))
    else:
        raise ValueError(f"Indexing with {type(indexer)} is not allowed.")


def _get_expected_shape(orig_shape, indexer):
    if isinstance(indexer, tuple):
        row, col = indexer
        new_row_shape = _get_expected_axis_length(orig_shape[0], row)
        new_col_shape = _get_expected_axis_length(orig_shape[1], col)
        return (new_row_shape, new_col_shape)
    else:
        return (_get_expected_axis_length(orig_shape[0], indexer),) + orig_shape[1:]
