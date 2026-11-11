import os
import shutil
import sys
from os import path
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools_rust import RustExtension

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

# add numpy headers
include_dirs = [np.get_include()]

# check if debug build
debug_build = os.getenv("TABMAT_DEBUG", "0").lower() in ("true", "1")
print(f"Debug Build: {debug_build}")

if sys.platform == "win32":
    allocator_libs = []  # type: ignore
    extra_compile_args = ["/openmp", "/O2", "/std:c++17"]
    extra_link_args = ["/openmp"]
    # make sure we can find xsimd headers
    include_dirs.append(os.path.join(sys.prefix, "Library", "include"))
elif sys.platform == "darwin":
    jemalloc_config = shutil.which("jemalloc-config")
    if "JE_INSTALL_SUFFIX" in os.environ:
        je_install_suffix = os.environ["JE_INSTALL_SUFFIX"]
    elif jemalloc_config is None:
        je_install_suffix = ""
    else:
        pkg_info = (
            Path(jemalloc_config).parent.parent / "lib" / "pkgconfig" / "jemalloc.pc"
        ).read_text()
        je_install_suffix = [
            i.split("=")[1]
            for i in pkg_info.split("\n")
            if i.startswith("install_suffix=")
        ].pop()
    allocator_libs = [f"jemalloc{je_install_suffix}"]
    extra_compile_args = [
        "-Xpreprocessor",
        "-fopenmp",
        "-O3",
        "--std=c++17",
        f"-DJEMALLOC_INSTALL_SUFFIX={je_install_suffix}",
    ]
    extra_link_args = ["-lomp"]
else:
    jemalloc_config = shutil.which("jemalloc-config")
    if jemalloc_config is None:
        je_install_suffix = ""
    else:
        pkg_info = (
            Path(jemalloc_config).parent.parent / "lib" / "pkgconfig" / "jemalloc.pc"
        ).read_text()
        je_install_suffix = [
            i.split("=")[1]
            for i in pkg_info.split("\n")
            if i.startswith("install_suffix=")
        ].pop()
    allocator_libs = [f"jemalloc{je_install_suffix}"]
    extra_compile_args = [
        "-fopenmp",
        "-O3",
        "--std=c++17",
        f"-DJEMALLOC_INSTALL_SUFFIX={je_install_suffix}",
    ]
    extra_link_args = ["-fopenmp"]

extension_args = dict(
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)
ext_modules = [
    Extension(
        name="tabmat.ext.sparse",
        sources=["src/tabmat/ext/sparse.pyx"],
        libraries=allocator_libs,
        **extension_args,
    ),
    Extension(
        name="tabmat.ext.dense",
        sources=["src/tabmat/ext/dense.pyx"],
        libraries=allocator_libs,
        **extension_args,
    ),
    Extension(
        name="tabmat.ext.split",
        sources=["src/tabmat/ext/split.pyx"],
        **extension_args,
    ),
]

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
    ext_modules=cythonize(
        ext_modules,
        annotate=False,
        compiler_directives={
            "language_level": "3",
            "freethreading_compatible": True,
            "boundscheck": debug_build,
            "wraparound": debug_build,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "cpow": True,
            "legacy_implicit_noexcept": True,
        },
    ),
    rust_extensions=[
        # Built from the Cargo.toml at the repository root. `debug` is passed
        # explicitly because setuptools-rust would otherwise make in-place
        # (editable) builds debug builds.
        RustExtension("tabmat.ext.categorical", debug=debug_build),
    ],
    zip_safe=False,
)
