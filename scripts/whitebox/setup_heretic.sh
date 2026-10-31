#!/usr/bin/env bash
# Install Heretic at the commit the `heretic` attack expects, in its own environment.
#
# Heretic pins transformers 5.x, which conflicts with tamperbench's environment, so it
# lives in a separate uv project environment and runs as a subprocess. PyPI
# `heretic-llm` cannot run headlessly, so this installs from git.
#
#   HERETIC_HOME=/scratch/me/heretic scripts/whitebox/setup_heretic.sh
#   export HERETIC_BIN="$HERETIC_HOME/venv/bin/heretic"
#
# Run it where PyPI is reachable (on clusters whose compute nodes have no PyPI egress,
# the login node). Re-running is idempotent.
set -euo pipefail

HERETIC_COMMIT=3521f8648a0dccf6e12a92666862632235fac7e6
HERETIC_HOME="${HERETIC_HOME:-$HOME/.cache/tamperbench/heretic-install}"
SRC="$HERETIC_HOME/src"
VENV="$HERETIC_HOME/venv"

if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/p-e-w/heretic.git "$SRC"
fi
git -C "$SRC" fetch -q origin
git -C "$SRC" checkout -q "$HERETIC_COMMIT"

# Heretic's own lockfile, into a dedicated environment (never tamperbench's).
(cd "$SRC" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen)
# The Jev scorer plugin (`objective: jev`) calls the judge from inside Heretic's process.
# `uv sync` removes it again, so it is reinstalled on every run.
uv pip install --quiet --python "$VENV/bin/python" "typesafe-sdk>=0.7.1,<0.8"

"$VENV/bin/python" -c "import heretic, torch, transformers, typesafe_sdk; print('heretic ok; torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
echo "export HERETIC_BIN=$VENV/bin/heretic"
