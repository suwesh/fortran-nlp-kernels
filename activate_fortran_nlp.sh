#!/usr/bin/env bash

source "$HOME/env3.12/bin/activate"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "fortran-nlp-kernels environment activated"
echo "Python: $(command -v python)"
echo "Project root: $PROJECT_ROOT"