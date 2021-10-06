#!/bin/bash

set -exo pipefail

source ~/.profile
mamba install -y yq jq

mamba install -y yq
yq -Y ". + {dependencies: [.dependencies[], \"python=${PYTHON_VERSION}\"] }" environment.yml > /tmp/environment.yml
mamba env create -f /tmp/environment.yml
conda activate $(yq -r .name environment.yml)
pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .
pytest tests --doctest-modules src/
