# Contributing and Development

Hello! And thanks for exploring tabmat more deeply. Please see the issue tracker and pull requests tabs on Github for information about what is currently happening. Feel free to post an issue if you'd like to get involved in development and don't really know where to start -- we can give some advice. 

We welcome contributions of any kind!

- New features
- Feature requests
- Bug reports
- Documentation
- Tests
- Questions

Pull request process
--------------------------------------------------

- Before working on a non-trivial PR, please first discuss the change you wish to make via issue, Slack, email or any other method with the owners of this repository. This is meant to prevent spending time on a feature that will not be merged.
- Please make sure that a new feature comes with adequate tests. If these require data, please check if any of our existing test data sets fits the bill.
- Please make sure that all functions come with proper docstrings. If you do extensive work on docstrings, please check if the Sphinx documentation renders them correctly. ReadTheDocs builds on every commit to an open pull request. You can see whether the documentation has successfully built in the "checks" section of the PR. Once the build finishes, your documentation should be accessible by clicking the "details" link next to the check in the GitHub interface and will appear at a URL like: ``https://tabmat--###.org.readthedocs.build/en/###/`` where ``###`` is the number of your PR.
- Please make sure you have our pre-commit hooks installed.
- If you fix a bug, please consider first contributing a test that _fails_ because of the bug and then adding the fix as a separate commit, so that the CI system picks it up.
- Please add an entry to the change log and increment the version number according to the type of change. We use semantic versioning. Update the major if you break the public API. Update the minor if you add new functionality. Update the patch if you fixed a bug. All changes that have not been released are collected under the date ``UNRELEASED``.

Releases
--------------------------------------------------

- We make package releases infrequently, but usually any time a new non-trivial feature is contributed or a bug is fixed. To make a release, just open a PR that updates the change log with the current date. Once that PR is approved and merged, you can create a new release on [GitHub](https://github.com/Quantco/tabmat/releases/new). Use the version from the change log as tag and copy the change log entry into the release description. New releases on GitHub are automatically deployed to the QuantCo conda channel.

Development installation
------------------------
We use [pixi](https://prefix.dev/) for setting up the project dependencies. [Install it](https://pixi.sh/latest/#installation) first if you do not have it already. The following commands will set up the project for development:

```bash
git clone git@github.com:Quantco/tabmat.git
cd tabmat

# Set up our pre-commit hooks for ruff, mypy, and cython-lint.
pixi run pre-commit-install

# Set up a pixi environment with the dependencies and install the package in editable mode.
pixi run postinstall

# If you want to install the dependencies necessary for benchmarking against other GLM packages:
pixi run -e benchmark postinstall

# If you want to work on the documentation:
pixi run -e docs postinstall

# You can run any command in the pixi environment with `pixi run <command>`. For example:
pixi run [-e ENVIRONMENT] ipython

# Alternatively, you can create a shell with the pixi environment activated:
pixi shell

# A number of pixi tasks are available for commonly used commands.
# You can run them with `pixi run <task>`.
# To get a list of available tasks, run:
pixi task list
```

Testing and continuous integration
--------------------------------------------------
The test suite is in ``tests/`` and can be run using ``pixi run test``.

Developing the documentation
----------------------------------------

The documentation is built with Sphinx. To develop the documentation: ``pixi run serve-docs``. Then, navigate to `<http://localhost:8000>`_ to view the documentation.

Alternatively, if you install `entr <http://eradman.com/entrproject/>`_, then you can auto-rebuild the documentation any time a file changes with:

```bash
   cd docs
   ./dev
```

Conda-forge packaging
---------------------

See [the feedstock on GitHub here](https://github.com/conda-forge/tabmat-feedstock).

See the packaging and maintenance advice [here](https://conda-forge.org/docs/maintainer/adding_pkgs.html#the-staging-process) and [here](https://conda-forge.org/docs/maintainer/updating_pkgs.html).
