"""Kernels for :class:`tabmat.DenseMatrix`.

The implementation lives in the Rust extension module ``tabmat.ext._rust``
(see ``src/tabmat/ext/rust/dense.rs``); this module only re-exports its
functions under their historical import path.
"""

import numpy as np

from ._rust.dense import (
    dense_matvec,
    dense_rmatvec,
    dense_sandwich,
    transpose_square_dot_weights,
)

__all__ = [
    "dense_matvec",
    "dense_rmatvec",
    "dense_sandwich",
    "transpose_square_dot_weights",
]


def _check_center(center, m, dtype, name="center", zeros=False):
    """Return ``center`` as a contiguous array of length ``m``.

    ``None`` stays ``None``, or becomes zeros if ``zeros``. Used by the
    remaining Cython modules (``sparse.pyx`` and ``split.pyx``); the Rust
    kernels validate their ``center`` argument themselves.
    """
    if center is None:
        return np.zeros(m, dtype=dtype) if zeros else None
    center = np.ascontiguousarray(center, dtype=dtype)
    if center.shape[0] != m:
        raise ValueError(f"{name} must have length {m}, got {center.shape[0]}.")
    return center
