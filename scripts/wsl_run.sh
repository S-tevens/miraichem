#!/usr/bin/env bash
# Usage: scripts/wsl_run.sh <command...>   (runs inside the WSL venv with src/ on the path)
export PATH="$HOME/.local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec "$HOME/mc-venv/bin/$@"