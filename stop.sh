#!/usr/bin/env bash
# stop.sh — Stop the Flint backend gracefully.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/data/flint.pid"

RESET=$'\033[0m'; BOLD=$'\033[1m'; DIM=$'\033[2m'
GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'

ok()   { echo "${GREEN}✓${RESET}  $*"; }
warn() { echo "${YELLOW}⚠${RESET}  $*"; }
err()  { echo "${RED}✗${RESET}  $*" >&2; }

echo ""

if [[ ! -f "$PID_FILE" ]]; then
    warn "No PID file found — Flint does not appear to be running."
    echo ""
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    warn "Process $PID is not running (stale PID file removed)."
    rm -f "$PID_FILE"
    echo ""
    exit 0
fi

echo "  Stopping Flint (PID $PID)…"

# SIGTERM first — lets uvicorn finish the current request and run shutdown hooks
kill -TERM "$PID" 2>/dev/null

# Wait up to 8 seconds for clean exit
for i in $(seq 1 8); do
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

# Force-kill if still alive
if kill -0 "$PID" 2>/dev/null; then
    warn "Process did not exit cleanly — sending SIGKILL."
    kill -KILL "$PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
ok "Flint stopped."
echo ""
