#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'Erro: %s não foi encontrado. Instale Python 3 e tente novamente.\n' "${PYTHON_BIN}" >&2
  exit 1
fi

printf 'Criando/atualizando ambiente virtual em %s\n' "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  numpy \
  pandas \
  scikit-learn \
  scipy \
  matplotlib \
  seaborn \
  jupyterlab

printf '\nAmbiente pronto. Ative-o com:\n  source %s/bin/activate\n' "${VENV_DIR}"
