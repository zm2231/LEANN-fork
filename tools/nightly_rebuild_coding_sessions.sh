#!/usr/bin/env bash
# Nightly rebuild of the coding-sessions LEANN index.
# Invoke from cron at 04:30 local time:
#   30 4 * * * /Volumes/4/GitHub/LEANN-fork/tools/nightly_rebuild_coding_sessions.sh
#
# Environment overrides:
#   LEANN_LOCAL_USER         Author tag for user-authored chunks (default: local_user)
#   LEANN_PI_AGENT_ROOTS     Extra pi-agent session roots (comma-separated)
#   CODING_SESSIONS_INDEX    Target index name (default: coding-sessions)
#   LEANN_HOME               Repo root (default: /Volumes/4/GitHub/LEANN-fork)
set -euo pipefail

LEANN_HOME="${LEANN_HOME:-/Volumes/4/GitHub/LEANN-fork}"
LOG_DIR="${LOG_DIR:-$HOME/.leann/logs}"
INDEX_NAME="${CODING_SESSIONS_INDEX:-coding-sessions}"
LEANN_LOCAL_USER="${LEANN_LOCAL_USER:-zain}"
LEANN_PI_AGENT_ROOTS="${LEANN_PI_AGENT_ROOTS:-/Volumes/4/GitHub/pi-ult/sessions}"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/coding-sessions-${TS}.log"

cd "$LEANN_HOME"
exec >"$LOG_FILE" 2>&1

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting coding-sessions rebuild ==="
echo "index: $INDEX_NAME"
echo "user: $LEANN_LOCAL_USER"
echo "pi roots: $LEANN_PI_AGENT_ROOTS"

LEANN_LOCAL_USER="$LEANN_LOCAL_USER" \
LEANN_PI_AGENT_ROOTS="$LEANN_PI_AGENT_ROOTS" \
"$LEANN_HOME/.venv/bin/python" "$LEANN_HOME/tools/build_coding_sessions.py" \
  --index "$INDEX_NAME" \
  --force

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished ==="
