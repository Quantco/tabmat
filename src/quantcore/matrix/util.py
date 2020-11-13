from typing import List, Optional, Tuple, Union

import numpy as np


def set_up_rows_or_cols(
    arr: Optional[Union[List[int], np.ndarray]], length: int, dtype=np.int32
) -> np.ndarray:
    if arr is None:
        return np.arange(length, dtype=dtype)
    return np.asarray(arr).astype(dtype)


def setup_restrictions(
    shape: Tuple[int, int],
    rows: Optional[np.ndarray],
    cols: Optional[np.ndarray],
    dtype=np.int32,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = set_up_rows_or_cols(rows, shape[0], dtype)
    cols = set_up_rows_or_cols(cols, shape[1], dtype)
    return rows, cols
