from typing import Optional

import numpy as np


def set_up_rows_or_cols(
    arr: Optional[np.ndarray], length: int, dtype=np.int32
) -> np.ndarray:
    """Set up rows or columns using input array and input length."""
    if arr is None:
        return np.arange(length, dtype=dtype)
    return np.asarray(arr).astype(dtype)


def setup_restrictions(
    shape: tuple[int, int],
    rows: Optional[np.ndarray],
    cols: Optional[np.ndarray],
    dtype=np.int32,
) -> tuple[np.ndarray, np.ndarray]:
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


def check_matvec_dimensions(mat, vec: np.ndarray, transpose: bool) -> None:
    """Assert that the dimensions for the matvec operation are compatible."""
    match_dim = 0 if transpose else 1
    if mat.shape[match_dim] != vec.shape[0]:
        raise ValueError(
            f"shapes {mat.shape} and {vec.shape} not aligned: "
            f"{mat.shape[match_dim]} (dim {match_dim}) != {vec.shape[0]} (dim 0)"
        )
