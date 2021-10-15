import io
import os
import platform
import shutil
import sys
from os import path
from pathlib import Path

import mako.template
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

# TODO: this should be moved inside the compilation of the extension
print("templating C source")
templates = [
    "src/tabmat/ext/dense_helpers-tmpl.cpp",
    "src/tabmat/ext/sparse_helpers-tmpl.cpp",
    "src/tabmat/ext/cat_split_helpers-tmpl.cpp",
]

for fn in templates:
    tmpl = mako.template.Template(filename=fn)

    buf = io.StringIO()
    ctx = mako.runtime.Context(buf)
    tmpl.render_context(ctx)
    rendered_src = buf.getvalue()

    out_fn = fn.split("-tmpl")[0] + ".cpp"

    # When the templated source code hasn't changed, we don't want to write the
    # file again because that'll touch the file and result in a rebuild
    write = True
    if path.exists(out_fn):
        with open(out_fn, "r") as f:
            out_fn_src = f.read()
            if out_fn_src == rendered_src:
                write = False

    if write:
        with open(out_fn, "w") as f:
            f.write(rendered_src)

# add numpy headers
include_dirs = [np.get_include()]

if sys.platform == "win32":
    allocator_libs = []
    extra_compile_args = ["/openmp", "/O2"]
    extra_link_args = ["/openmp"]
    # make sure we can find xsimd headers
    include_dirs.append(os.path.join(sys.prefix, "Library", "include"))
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
        "-ffast-math",
        "--std=c++17",
        f"-DJEMALLOC_INSTALL_SUFFIX={je_install_suffix}",
    ]
    extra_link_args = ["-fopenmp"]


architecture = os.environ.get("GLM_ARCHITECTURE", "native")
if architecture != "default":
    # Don't set "-march=native" on macOS arm64 as this doesn't exist there.
    # Note that "arm64" already implies macOS. On Linux this is called "aarch64".
    if not (platform.machine() == "arm64" and architecture == "native"):
        extra_compile_args.append("-march=" + architecture)

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
        name="tabmat.ext.categorical",
        sources=["src/tabmat/ext/categorical.pyx"],
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
    setup_requires=["setuptools_scm"],
    description="Efficient matrix representations for working with tabular data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quantco/tabmat",
    author="QuantCo, Inc.",
    author_email="noreply@quantco.com",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    ext_modules=cythonize(ext_modules, annotate=False),
    zip_safe=False,
)
