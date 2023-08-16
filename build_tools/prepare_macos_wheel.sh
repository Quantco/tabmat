#!/bin/bash

set -exuo pipefail

if [[ "${ARCHFLAGS:-}" == *arm64 ]]; then
    CONDA_CHANNEL="conda-forge/osx-arm64"
else
    CONDA_CHANNEL="conda-forge/osx-64"
fi

/Users/runner/micromamba-bin/micromamba create -y -p $CONDA/envs/build -c $CONDA_CHANNEL jemalloc-local xsimd llvm-openmp
