#!/bin/bash

set -exo pipefail

mamba install -y conda-build
if grep -q "osx-arm64" .ci_support/${CONDA_BUILD_YML}; then
  CONDA_BUILD_ARGS="--no-test"
fi
conda build -m .ci_support/${CONDA_BUILD_YML} conda.recipe ${CONDA_BUILD_ARGS:-}
