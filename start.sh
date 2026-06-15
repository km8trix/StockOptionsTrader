#!/usr/bin/env bash
#
# Launch the trading GUI with the operator's .env loaded into the process.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# run_gui.py deliberately passes load_dotenv=False to Flask: the APP must
# never bulk-load .env (and the real E*TRADE credentials in it) on its own —
# credentials enter the process ONLY by explicit operator action (see the
# TestLauncherEnvHygiene regression test). Running THIS script is that
# explicit action: you, the operator, choose to source .env and launch.
# run_gui.py's control is untouched; the loading happens in your shell here.
#
# USAGE
# -----
#   ./start.sh
#
# Note: .env is sourced as a shell script, so it must contain plain
# KEY=value lines (quote any value with spaces or '#'). For production you
# additionally need ETRADE_ENV=production, ETRADE_PROD_CONSUMER_KEY/SECRET,
# ETRADE_PRODUCTION_ACK=I_UNDERSTAND_LIVE_TRADING and ETRADE_ALLOW_NETWORK=1.
set -euo pipefail

# Resolve the repo root (this script's directory) so it works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
    # set -a: every variable assigned while sourcing is exported, so .env
    # values land in the environment run_gui.py reads. set +a restores normal.
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "Loaded .env from $ROOT/.env"
else
    echo "No .env found in $ROOT — launching with the existing environment."
fi

# Prefer the project venv (Python 3.13); fall back to python3 if it is absent.
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    PY="python3"
fi

# exec so the Python process replaces this shell (clean Ctrl-C / signals).
exec "$PY" run_gui.py
