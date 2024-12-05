#!/bin/bash

set -exuo pipefail

if [[ "${ARCHFLAGS:-}" == *arm64 ]]; then
    export CONDA_SUBDIR="osx-arm64"
else
    export CONDA_SUBDIR="osx-64"
fi

/Users/runner/micromamba-bin/micromamba create -y -p $MAMBA_ROOT_PREFIX/envs/build -c conda-forge jemalloc-local "xsimd<11|>12.1" llvm-openmp
