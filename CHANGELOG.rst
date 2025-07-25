.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

4.1.2 - 2025-07-17
------------------

**Bug fix:**

- Fixed a bug which caused issues when constructing tabmat matrices from existing ``ModelSpec``\s when they contained categorical columns with all levels dropped.
- We can now treat dedicated pandas string series - which are the defaults for strings since pandas 2.3 - as categoricals.


4.1.1 - 2025-01-30
------------------

**Bug fix:**

- A more robust :meth:`DenseMatrix._get_col_stds` results in more accurate :meth:`StandardizedMatrix.sandwich` results.

**Other changes:**

- Build wheel for pypi on python 3.13.
- Build and test with python 3.13 in CI.


4.1.0 - 2024-11-07
------------------

**New feature:**

- Added a new function, :func:`tabmat.from_df`, to convert any dataframe supported by narwhals into a :class:`tabmat.SplitMatrix`.

**Other changes:**

- Allow :class:`CategoricalMatrix` to be initialized directly with indices and categories.
- Added checks for dimension and ``dtype`` mismatch in :meth:`MatrixBasesandwich.sandwich`.

**Bug fix:**

- Fixed a bug in :meth:`tabmat.CategoricalMatrix.standardize` that sometimes returned ``nan`` values for the standard deviation due to numerical instability if using ``np.float32`` precision.


4.0.1 - 2024-06-25
------------------

**Other changes:**

- Removed reference to the ``.A`` attribute and replaced it with ``.toarray()``.
- Add support between formulaic and pandas 3.0.
- Support pypi release for numpy 2.0

4.0.0 - 2024-04-23
------------------

**Breaking changes**:

- To unify the API, :class:`DenseMatrix` does not inherit from :class:`np.ndarray` anymore. To convert a :class:`DenseMatrix` to a :class:`np.ndarray`, use :meth:`DenseMatrix.unpack`.
- Similarly, :class:`SparseMatrix` does not inherit from :class:`sps.csc_matrix` anymore. To convert a :class:`SparseMatrix` to a :class:`sps.csc_matrix`, use :meth:`SparseMatrix.unpack`.

**New features:**

- Added column name and term name metadata to :class:`MatrixBase` objects. These are automatically populated when initializing a :class:`MatrixBase` from a :class:`pandas.DataFrame`. In addition, they can be accessed and modified via the :attr:`MatrixBase.column_names` and :attr:`MatrixBase.term_names` properties.
- Added a formula interface for creating tabmat matrices from pandas data frames. See :func:`tabmat.from_formula` for details.
- Added support for missing values in :class:`CategoricalMatrix` by either creating a separate category for them or treating them as all-zero rows.
- Added support for handling missing categorical values in pandas data frames.

**Bug fix:**

- Added cython compiler directive ``legacy_implicit_noexcept = True`` to fix performance regression with cython 3.

**Other changes:**

- Refactored the pre-commit hooks to use ruff.
- Refactored :meth:`CategoricalMatrix.transpose_matvec` to be deterministic when using OpenMP.
- Adjusted transformation to sparse format in :func:`tabmat.from_pandas` to future changes in pandas.

3.1.13 - 2023-10-17
-------------------

**Other changes:**

- Pypi release is now done using trusted publisher.
- Fix build and upload of ``x86_64`` wheels on Linux.

3.1.12 - 2023-10-16
-------------------

**Other changes:**

- Fixed macos arm64 wheels with proper linkage.

3.1.11 - 2023-10-13
-------------------

**Other changes:**

- Improve the performance of ``from_pandas`` in the case of low-cardinality categorical variables.
- Require Python>=3.9 in line with `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html#support-table>`_
- Build and test with Python 3.12 in CI.
- Fixed macos arm64 wheels with proper linkage.

3.1.10 - 2023-06-23
-------------------

**Bug fixes:**

- We fixed a bug in the dense sandwich product, which would previously segfault for very large matrices.
- Fixed the column order when initializing a ``SplitMatrix`` from a list containing other ``SplitMatrix`` objects.
- Fixed ``getcol`` not respecting the ``drop_first`` attribute of a ``CategoricalMatrix``.

3.1.9 - 2023-06-16
------------------

**Other changes:**

- Support building on architectures that are unsupported by xsimd.

3.1.8 - 2023-06-13
------------------

**Other changes:**

- The C++ types have been refactored. Loop indices are now using the ``Py_ssize_t`` type. Integers now have a templated type as well.
- The documentation for ``matvec`` and ``matvec_transpose`` has been updated to reflect actual behavior.
- Checks for dimension mismatch in ``matvec`` and ``matvec_transpose`` arguments have been added.
- Remove upper pin on xsimd.

3.1.7 - 2022-03-28
------------------

**Bug fix:**

- We fixed a bug in the cross sandwich product, which would previously segfault for very large matrices.

3.1.6 - 2022-03-27
------------------

**Bug fix:**

- We fixed a bug in the dense sandwich product, which would previously segfault for very large F-contiguous matrices.

3.1.5 - 2022-03-20
------------------

**Bug fix:**

- We fixed a bug in the dense matrix-vector and sandwich products, which would previously segfault for very large matrices.


3.1.4 - 2022-02-07
------------------

**Bug fix:**

- Fixed the loading of jemalloc in Apple Silicon wheels.


3.1.3 - 2022-01-26
------------------

**Other changes:**

- Build and upload wheels for Apple Silicon.


3.1.2 - 2022-07-01
------------------

**Other changes:**

- Next attempt to build wheel for PyPI without ``march=native``.


3.1.1 - 2022-07-01
------------------

**Other changes:**

- Add Python 3.10 support to CI (remove Python 3.6).
- Build wheel for PyPI without ``march=native``.


3.1.0 - 2022-03-07
------------------

**New feature**

- :class:`tabmat.CategoricalMatrix` now accepts a `drop_first` argurment. This allows the user to drop the first column of a CategoricalMatrix to avoid multicollinearity problems in unregularized models.
- :class:`tabmat.StandardizedMatrix` and :class:`tabmat.MatrixBase` now support the `multiply` method.


3.0.8 - 2022-01-03
------------------

**Bug fix**

- Always use 64bit integers for indexing in :meth:`tabmat.ext.sparse.sparse_sandwich` to avoid segmentation faults on very wide problems.


3.0.7 - 2021-11-23
------------------

**Bug fix**

- Disable the use of static TLS in the Linux wheels to avoid issues with too small TLS on some distributions.

3.0.6 - 2021-11-11
------------------

**Bug fix**

- We fixed a bug in :meth:`tabmat.SplitMatrix.matvec`, where incorrect matrix vector products were computed when a ``SplitMatrix`` did not contain any dense components.


3.0.5 - 2021-11-05
------------------

**Other changes**

- We are now specifying the run time dependencies in ``setup.py``, so that missing dependencies are automatically installed from PyPI when installing ``tabmat`` via pip.

3.0.4 - 2021-11-03
------------------

**Other changes**

- tabmat is now available on PyPI and will be automatically updated when a new release is published.

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

- Optimized the :meth:`tabmat.SparseMatrix.matvec` and :meth:`tabmat.SparseMatrix.transpose_matvec` for when ``rows`` and ``cols`` are None.
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
