.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========


1.0.0 - 2020-11-10
-------------------

**Breaking change:**

- Rename `dot` to `matvec`. Our `dot` function supports matrix-vector multiplication for every subclass, but only supports matrix-matrix multiplication for some. We therefore rename it to `matvec` in line with other libraries.

**Bug fix:**

- Fix a bug in matvec for categorical components when the number of categories exceeds the number of rows.


0.0.6 - 2020-08-03 
------------------

See git history.
