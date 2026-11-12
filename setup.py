import os
from os import path

from setuptools import find_packages, setup
from setuptools_rust import RustExtension

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

# check if debug build
debug_build = os.getenv("TABMAT_DEBUG", "0").lower() in ("true", "1")
print(f"Debug Build: {debug_build}")

setup(
    name="tabmat",
    use_scm_version={"version_scheme": "post-release"},
    description="Efficient matrix representations for working with tabular data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quantco/tabmat",
    author="QuantCo, Inc.",
    author_email="noreply@quantco.com",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["formulaic>=1.2", "narwhals", "numpy", "scipy"],
    python_requires=">=3.11",
    rust_extensions=[
        # All kernels are one extension module (see Cargo.toml) so that they
        # share one rayon thread pool. `debug` is passed explicitly because
        # setuptools-rust would otherwise make in-place (editable) builds
        # debug builds.
        RustExtension("tabmat.ext.crabmat", path="Cargo.toml", debug=debug_build),
    ],
    zip_safe=False,
)
