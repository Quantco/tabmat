.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

3.0.3 - 2021-10-15
------------------

**Bug fix**

- We now support ``xsimd>=8`` and support alternative jemalloc installations.


3.0.2 - 2021-10-14
------------------

**Bug fix**

- Allow to link to alternatively suffixed jemalloc installation to work around `#113 <https://github.com/Quantco/tabmat/issues/113>`_ .

3.0.1 - 2021-10-07
------------------

**Bug fix**

- The license was mistakenly left as proprietary. Corrected to BSD-3-Clause.

**Other changes**

- ReadTheDocs integration.
- CONTRIBUTING.md
- Correct pyproject.toml to work with PEP-517

3.0.0 - 2021-10-07
------------------

**Breaking changes**:

- The package has been renamed to ``tabmat``. CELEBRATE!
- The :func:`one_over_var_inf_to_val` function has been made private.
- The :func:`csc_to_split` function has been re-named to :func:`tabmat.from_csc` to match the :func:`tabmat.from_pandas` function.
- The :meth:`tabmat.MatrixBase.get_col_means` and :meth:`tabmat.MatrixBase.get_col_stds` methods have been made private.
- The :meth:`cross_sandwich` method has also been made private.

**Bug fix**

- :func:`StandardizedMatrix.transpose_matvec` was giving the wrong answer when the `out` parameter was provided. This is now fixed.
- :func:`SplitMatrix.__repr__` now calls the `__repr__` method of component matrices instead of `__str__`.

**Other changes**

- Optimized the :meth:`tabmat.SparseMatrix.matvec` and :meth:`tabmat.SparseMatrix.tranpose_matvec` for when ``rows`` and ``cols`` are None.
- Implemented :func:`CategoricalMatrix.__rmul__`
- Reorganizing the documentation and updating the text to match the current API.
- Enable indexing the rows of a ``CategoricalMatrix``. Previously :func:`CategoricalMatrix.__getitem__` only supported column indexing.
- Allow creating a ``SplitMatrix`` from a list of any ``MatrixBase`` objects including another ``SplitMatrix``.
- Reduced memory usage in :meth:`tabmat.SplitMatrix.matvec`.

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

We renamed several public functions to make them private. These include functions in :mod:`tabmat.benchmark` that are unlikely to be used outside of this package as well as

   - :func:`tabmat.dense_matrix._matvec_helper`
   - :func:`tabmat.sparse_matrix._matvec_helper`.
   - :func:`tabmat.split_matrix._prepare_out_array`.


**Other changes**:

- We removed the dependency on ``sparse_dot_mkl``. We now use :func:`scipy.sparse.csr_matvec` instead of :func:`sparse_dot_mkl.dot_product_mkl` on all platforms, because the former suffered from poor performance, especially on narrow problems. This also means that we removed the function :func:`tabmat.sparse_matrix._dot_product_maybe_mkl`.
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
