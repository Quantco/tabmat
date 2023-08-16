#!/bin/bash

set -exuo pipefail

if [[ "${ARCHFLAGS:-}" == *arm64 ]]; then
    CONDA_CHANNEL="conda-forge/osx-arm64"
else
    CONDA_CHANNEL="conda-forge/osx-64"
fi

conda create -y -n build -c $CONDA_CHANNEL jemalloc-local xsimd llvm-openmp
