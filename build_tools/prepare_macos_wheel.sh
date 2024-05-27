#!/bin/bash

set -exuo pipefail

if [[ "${ARCHFLAGS:-}" == *arm64 ]]; then
    export CONDA_SUBDIR="osx-arm64"
else
    export CONDA_SUBDIR="osx-64"
    # libcxx>=17 needs osx 10.13, cibuildwheel wants 10.9
    # export CONDA_ARGS=" libcxx<17"
fi

/Users/runner/micromamba-bin/micromamba create -y -p $CONDA/envs/build -c conda-forge jemalloc-local "xsimd<11|>12.1" llvm-openmp ${CONDA_ARGS:-}
