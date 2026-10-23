#!/usr/bin/env bash
set -euo pipefail
: "${FECO_ROOT:?}"
exec /usr/bin/time -v bash "$FECO_ROOT/submit.sh" worker "$@"
