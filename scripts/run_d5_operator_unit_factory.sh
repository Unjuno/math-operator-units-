#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-detach}"
RUN_ROOT="runs/d5_operator_unit_factory"
PID_FILE="$RUN_ROOT/factory.pid"
LOCK_FILE="$RUN_ROOT/factory.lock"
mkdir -p "$RUN_ROOT" logs

status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "process: running (PID $(cat "$PID_FILE"))"
  else
    echo "process: not running"
  fi
  [[ -f "$RUN_ROOT/state.json" ]] && cat "$RUN_ROOT/state.json"
  [[ -f "evaluations/d5_operator_unit_factory/summary.json" ]] && echo "summary: evaluations/d5_operator_unit_factory/summary.json"
}

case "$MODE" in
  status) status; exit 0 ;;
  detach)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "D5 operator-unit factory already running: PID $(cat "$PID_FILE")"
      exit 0
    fi
    log="logs/d5_operator_unit_factory_$(date -u +%Y%m%dT%H%M%SZ).log"
    if command -v systemd-inhibit >/dev/null 2>&1; then
      nohup setsid systemd-inhibit --what=sleep:idle --why="D5 operator-unit factory" bash "$0" watch >"$log" 2>&1 < /dev/null &
    else
      nohup setsid bash "$0" watch >"$log" 2>&1 < /dev/null &
    fi
    echo $! > "$PID_FILE"
    echo "D5 operator-unit factory started: PID $!"
    echo "log: $log"
    exit 0
    ;;
  watch)
    exec 9>"$LOCK_FILE"
    flock -n 9 || { echo "another D5 factory owns the lock" >&2; exit 73; }
    trap 'rm -f "$PID_FILE"' EXIT
    max_attempts="${D5_MAX_ATTEMPTS:-5}"
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
      echo "D5 attempt $attempt/$max_attempts"
      set +e
      .venv/bin/python scripts/d5_operator_unit_factory.py
      code=$?
      set -e
      [[ $code -eq 0 ]] && exit 0
      [[ $code -eq 64 || $code -eq 65 || $code -eq 66 || $code -eq 73 ]] && exit "$code"
      sleep "${D5_RETRY_DELAY_SECONDS:-60}"
    done
    exit 1
    ;;
  foreground)
    exec .venv/bin/python scripts/d5_operator_unit_factory.py
    ;;
  *) echo "usage: $0 [detach|foreground|watch|status]" >&2; exit 64 ;;
esac
