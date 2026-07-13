#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
WORKER="${WORKER:-$ROOT/scripts/run_model_design_pilot.sh}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-60}"
LOCK_FILE="${LOCK_FILE:-$ROOT/runs/model_design_pilot/pilot.lock}"
STATE_FILE="${STATE_FILE:-$ROOT/runs/model_design_pilot/pilot_state.json}"
PID_FILE="${PID_FILE:-$ROOT/runs/model_design_pilot/pilot.pid}"
PILOT_LOG="${PILOT_LOG:-}"

mkdir -p runs/model_design_pilot logs
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another model-design pilot holds $LOCK_FILE" >&2
  exit 73
fi

write_watch_state() {
  local status="$1"
  local attempt="$2"
  local detail="$3"
  if [[ ! -x "$PYTHON" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${status}: ${detail}" >&2
    return
  fi
  "$PYTHON" - "$STATE_FILE" "$status" "$attempt" "$detail" "$$" "$PILOT_LOG" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
existing = {}
if path.exists():
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
existing.update(
    {
        "status": sys.argv[2],
        "watchdog_attempt": int(sys.argv[3]),
        "watchdog_detail": sys.argv[4],
        "watchdog_pid": int(sys.argv[5]),
        "log": sys.argv[6] or existing.get("log"),
        "updated_unix": time.time(),
    }
)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

cleanup_pid() {
  if [[ -f "$PID_FILE" ]]; then
    recorded="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$recorded" == "$$" || "$recorded" == "$PPID" ]]; then
      rm -f "$PID_FILE"
    fi
  fi
}

child_pid=""
on_signal() {
  local signal="$1"
  echo "received ${signal}; stopping pilot worker" >&2
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  write_watch_state interrupted "$attempt" "watchdog received ${signal}"
  cleanup_pid
  exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap cleanup_pid EXIT

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] model-design pilot attempt ${attempt}/${MAX_RESTARTS}"
  write_watch_state running "$attempt" "starting or resuming the four-condition pilot"

  env OPFUSION_PILOT_CHILD=1 OPFUSION_PILOT_WATCHED=1 \
    bash "$WORKER" foreground &
  child_pid=$!
  wait "$child_pid"
  status=$?
  child_pid=""

  if [[ $status -eq 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] model-design pilot completed"
    write_watch_state completed "$attempt" "all conditions and evaluations completed"
    exit 0
  fi

  case "$status" in
    64|65|66|73)
      echo "pilot stopped with permanent preflight status=${status}; not retrying" >&2
      write_watch_state failed "$attempt" "permanent preflight failure status=${status}"
      exit "$status"
      ;;
  esac

  if [[ $attempt -ge $MAX_RESTARTS ]]; then
    echo "giving up after ${attempt} attempts; last status=${status}" >&2
    write_watch_state failed "$attempt" "retry limit reached; last status=${status}"
    exit "$status"
  fi

  echo "pilot worker failed with status=${status}; resuming in ${RESTART_DELAY_SECONDS}s" >&2
  write_watch_state restarting "$attempt" "worker status=${status}; sleeping ${RESTART_DELAY_SECONDS}s"
  sleep "$RESTART_DELAY_SECONDS"
done
