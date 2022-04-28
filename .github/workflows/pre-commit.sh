#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh $*

git config --global --add safe.directory /github/workspace

mamba install -y pre-commit
pre-commit install
pre-commit run -a
