.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

2.0.1 - 2021-06-15
------------------

**Other changes**:

We removed the dependency on ``sparse_dot_mkl``. We now use ``scipy.sparse.csr_matvec`` instead of ``sparse_dot_mkl.dot_product_mkl`` on all platforms, because the former suffered from poor performance, especially on narrow problems.

2.0.0 - 2021-06-10
------------------

**Breaking change**:

Renaming several public functions to make them private.

**Other changes**:

Updating linter


Changelog
=========

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
