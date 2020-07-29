from typing import Optional, Tuple

import numpy as np


def setup_restrictions(
    shape: Tuple[int, int],
    rows: Optional[np.ndarray],
    cols: Optional[np.ndarray],
    dtype=np.int32,
) -> Tuple[np.ndarray, np.ndarray]:
    if rows is None:
        rows = np.arange(shape[0], dtype=dtype)
    elif rows.dtype != dtype:
        rows = rows.astype(dtype)
    if cols is None:
        cols = np.arange(shape[1], dtype=dtype)
    elif cols.dtype != dtype:
        cols = cols.astype(dtype)
    return rows, cols
