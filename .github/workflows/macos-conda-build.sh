#!/bin/bash

set -exo pipefail

mamba install -y conda-build
conda build -m ${CONDA_BUILD_YML} conda.recipe
