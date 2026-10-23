#!/usr/bin/env bash
set -euo pipefail
: "${FECO_ROOT:?}" "${OUTPUT_ROOT:?}"
source "$FECO_ROOT/environment.sh"
exec "$LETTA_PYTHON" "$FECO_ROOT/measure_worker.py" "$LETTA_PYTHON" -u "$FECO_ROOT/resume_letta.py" "$@"
