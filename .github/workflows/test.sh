#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh "$PYTHON_VERSION"

mamba install -y yq

cat environment.yml > /tmp/environment.yml

# pin version of some libraries, if specified
LIBRARIES=("python" "pandas" "numpy" "scipy")
for library in "${LIBRARIES[@]}"; do
    varname="${library^^}_VERSION"
    version=${!varname}
    if [[ -n "$version" && "$version" != "nightly" ]]; then
        yq -Y --in-place ". + {dependencies: [.dependencies[], \"${library}=${version}\"]}" /tmp/environment.yml
    fi
done

cat /tmp/environment.yml

mamba env create -f /tmp/environment.yml
conda activate $(yq -r .name environment.yml)

PRE_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
if [[ "$NUMPY_VERSION" == "nightly" ]]; then
    echo "Installing Numpy nightly"
    conda uninstall -y --force numpy
    pip install --pre --no-deps --only-binary :all: --upgrade --timeout=60 -i $PRE_WHEELS numpy
fi
if [[ "$PANDAS_VERSION" == "nightly" ]]; then
    echo "Installing Pandas nightly"
    conda uninstall -y --force pandas
    pip install --pre --no-deps --only-binary :all: --upgrade --timeout=60 -i $PRE_WHEELS pandas
fi
if [[ "$SCIPY_VERSION" == "nightly" ]]; then
    echo "Installing Scipy nightly"
    conda uninstall -y --force scipy
    pip install --pre --no-deps --only-binary :all: --upgrade --timeout=60 -i $PRE_WHEELS scipy
fi

git config --global --add safe.directory /github/workspace

pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .
pytest -nauto tests -m "not high_memory" --doctest-modules src/
