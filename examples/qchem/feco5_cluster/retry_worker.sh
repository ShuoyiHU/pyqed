#!/usr/bin/env bash
set -euo pipefail
: "${FECO_ROOT:?}"
source "$FECO_ROOT/environment.sh"
exec "$LETTA_PYTHON" "$FECO_ROOT/measure_worker.py" bash "$FECO_ROOT/submit.sh" worker "$@"
