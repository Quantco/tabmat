#!/bin/bash

set -exo pipefail

export CONDA_BUILD_SYSROOT=$(xcrun --sdk macosx --show-sdk-path)
mamba install -y conda-build
if grep -q "osx-arm64" .ci_support/${CONDA_BUILD_YML}.yaml; then
  CONDA_BUILD_ARGS="--no-test"
fi
conda build -m .ci_support/${CONDA_BUILD_YML}.yaml conda.recipe ${CONDA_BUILD_ARGS:-}
