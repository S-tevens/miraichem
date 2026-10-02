#!/usr/bin/env bash
# Creates a Python 3.11 venv in ~/mc-venv (WSL/Linux) with all pinned-candidate dependencies.
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:/usr/bin:/bin"
uv --version
cd ~
uv venv --python 3.11 mc-venv
uv pip install --python mc-venv/bin/python qiskit qiskit-aer qiskit-ibm-runtime pyscf mitiq scipy numpy pandas pyarrow matplotlib plotly streamlit pydantic pyyaml typer rich python-dotenv pytest pytest-cov ruff mypy pre-commit
mc-venv/bin/python -c "import pyscf,qiskit;print('pyscf',pyscf.__version__,'qiskit',qiskit.__version__)"
