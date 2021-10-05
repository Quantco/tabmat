.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

Unreleased
----------

**Breaking changes**:

- The :func:`one_over_var_inf_to_val` function has been made private.
- The :func:`csc_to_split` function has been re-named to :func:`quantcore.matrix.from_csc` to match the :func:`quantcore.matrix.from_pandas` function.

**Bug fix**

- :func:`quantcore.matrix.StandardizedMatrix.transpose_matvec` was giving the wrong answer when the ``out`` parameter was provided. This is now fixed.

**Other changes**

- Implemented :func:`CategoricalMatrix.__rmul__`
- Reorganizing the documentation and updating the text to match the current API.

2.0.3 - 2021-07-15
------------------

**Bug fix**

- In :func:`SplitMatrix.sandwich`, when a col subset was specified, incorrect output was produced if the components of the indices array were not sorted. :func:`SplitMatrix.__init__` now checks for sorted indices and maintains sorted index lists when combining matrices.

**Other changes**

- :func:`SplitMatrix.__init__` now filters out any empty matrices.
- :func:`StandardizedMatrix.sandwich` passes ``rows=None`` and ``cols=None`` onwards to the underlying matrix instead of replacing them with full arrays of indices. This should improve performance slightly.
- :func:`SplitMatrix.__repr__` now includes the type of the underlying matrix objects in the string output.

2.0.2 - 2021-06-24
------------------

**Bug fix**

Sparse matrices now accept 64-bit indices on Windows.


2.0.1 - 2021-06-20
------------------

**Bug fix**:

Split matrices now also work on Windows.


2.0.0 - 2021-06-17
------------------

**Breaking changes**:

We renamed several public functions to make them private. These include functions in :mod:`quantcore.matrix.benchmark` that are unlikely to be used outside of this package as well as

   - :func:`quantcore.matrix.dense_matrix._matvec_helper`
   - :func:`quantcore.matrix.sparse_matrix._matvec_helper`.
   - :func:`quantcore.matrix.split_matrix._prepare_out_array`.


**Other changes**:

- We removed the dependency on ``sparse_dot_mkl``. We now use :func:`scipy.sparse.csr_matvec` instead of :func:`sparse_dot_mkl.dot_product_mkl` on all platforms, because the former suffered from poor performance, especially on narrow problems. This also means that we removed the function :func:`quantcore.matrix.sparse_matrix._dot_product_maybe_mkl`.
- We updated the pre-commit hooks and made sure the code is line with the new hooks.


1.0.6 - 2020-04-26
------------------

**Other changes**:

We are now also making releases for Windows.

1.0.5 - 2020-04-26
------------------

**Other changes**:

Still trying.

1.0.4 - 2020-04-26
------------------

**Other changes**:

We are trying to make releases for Windows.


1.0.3 - 2020-04-21
------------------

**Bug fixes:**

- Added a check that matrices are two-dimensional in the ``SplitMatrix.__init__``
- Replace ``np.int`` with ``np.int64`` where appropriate due to NumPy deprecation of ``np.int``.


1.0.2 - 2020-04-20
------------------

**Other changes:**

- Added Python 3.9 support.
- Use ``scipy.sparse`` dot product when MKL isn't available.

1.0.1 - 2020-11-25
------------------

**Bug fixes:**

- Handling for nulls when setting up a ``CategoricalMatrix``
- Fixes to make several functions work with both row and col restrictions and out

**Other changes:**

- Added various tests and documentation improvements

1.0.0 - 2020-11-11
------------------

**Breaking change:**

- Rename `dot` to `matvec`. Our `dot` function supports matrix-vector multiplication for every subclass, but only supports matrix-matrix multiplication for some. We therefore rename it to `matvec` in line with other libraries.

**Bug fix:**

- Fix a bug in `matvec` for categorical components when the number of categories exceeds the number of rows.


0.0.6 - 2020-08-03 
------------------

See git history.
