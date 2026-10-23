#!/usr/bin/env bash
set -euo pipefail
FECO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FECO_ROOT
source "$FECO_ROOT/environment.sh"
exec "$LETTA_PYTHON" "$FECO_ROOT/retry_cases.py" "$@"
