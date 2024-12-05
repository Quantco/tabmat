#!/bin/bash

set -exuo pipefail

/Users/runner/micromamba-bin/micromamba create -y -p $MAMBA_ROOT_PREFIX/envs/build -c conda-forge jemalloc-local "xsimd<11|>12.1" llvm-openmp
