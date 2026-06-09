#!/usr/bin/env bash
# Restart-loop wrapper for the H406b weekend extreme-scale run.
#
# Runs hypotheses/h406b_weekend_maxscale.py in a loop. If the python process
# exits non-zero (crash / OOM-kill / power blip) AND the .DONE flag is absent,
# it sleeps RESTART_SLEEP seconds and relaunches -- the script resumes from the
# single overwriting checkpoint. The loop stops as soon as the .DONE flag
# exists (clean completion) or the restart cap is hit.
#
# Designed to be launched detached so it survives terminal close, e.g.:
#   cd /run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension
#   setsid nohup bash scripts/h406_weekend_run.sh >/dev/null 2>&1 &
#
# Stdout/stderr of each attempt is appended to the log (small text, fine).

set -u

PROJ="/run/media/mohit/NewVolume1/adversarial-robustness-toolbox/dissertation_extension"
PY="$PROJ/.venv/bin/python"
SCRIPT="$PROJ/hypotheses/h406b_weekend_maxscale.py"
RESULTS="$PROJ/results/fashion_mnist"
DONE_FLAG="$RESULTS/h406b_weekend.DONE"
LOG="$RESULTS/h406b_weekend.log"

MAX_RESTARTS="${MAX_RESTARTS:-200}"
RESTART_SLEEP="${RESTART_SLEEP:-30}"

cd "$PROJ" || { echo "cannot cd to $PROJ" >&2; exit 1; }

mkdir -p "$RESULTS"

attempt=0
{
  echo "=================================================================="
  echo "[wrapper] H406b weekend run starting $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[wrapper] python=$PY"
  echo "[wrapper] script=$SCRIPT"
  echo "[wrapper] DONE flag=$DONE_FLAG"
  echo "[wrapper] MAX_RESTARTS=$MAX_RESTARTS RESTART_SLEEP=${RESTART_SLEEP}s"
  echo "=================================================================="
} >> "$LOG" 2>&1

while :; do
  if [ -f "$DONE_FLAG" ]; then
    echo "[wrapper] .DONE present -> nothing to do, exiting." >> "$LOG" 2>&1
    break
  fi

  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_RESTARTS" ]; then
    echo "[wrapper] hit MAX_RESTARTS=$MAX_RESTARTS without .DONE; giving up at $(date '+%H:%M:%S')." >> "$LOG" 2>&1
    exit 1
  fi

  echo "[wrapper] attempt #$attempt launch $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG" 2>&1
  "$PY" "$SCRIPT" >> "$LOG" 2>&1
  rc=$?
  echo "[wrapper] attempt #$attempt exited rc=$rc $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG" 2>&1

  if [ -f "$DONE_FLAG" ]; then
    echo "[wrapper] .DONE present after attempt #$attempt -> COMPLETE." >> "$LOG" 2>&1
    break
  fi

  if [ "$rc" -eq 0 ]; then
    # Clean exit without .DONE means a deliberate stop (e.g. MAX_SECONDS); do
    # not hammer-relaunch. Still relaunch after the normal sleep so resume works.
    echo "[wrapper] rc=0 but no .DONE (deliberate stop?); sleeping ${RESTART_SLEEP}s then resuming." >> "$LOG" 2>&1
  else
    echo "[wrapper] non-zero exit; sleeping ${RESTART_SLEEP}s then resuming from checkpoint." >> "$LOG" 2>&1
  fi
  sleep "$RESTART_SLEEP"
done

echo "[wrapper] loop finished $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG" 2>&1
