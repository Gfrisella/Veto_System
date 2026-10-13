#!/bin/bash
# Run this project on the physik cluster (through our own proxy on port 2226).
#
#   bin/remote.sh sync                       copy veto/ (code only) to the cluster
#   bin/remote.sh run <cmd...>               run in the container on $GPU_HOST, foreground
#                                            e.g. bin/remote.sh run bin/optimize_sbt.py
#   bin/remote.sh launch <id> <cmd...>       long job in tmux on $GPU_HOST (see physik_deployment)
#   bin/remote.sh status | peek <id> | kill <id>
#   bin/remote.sh fetch-results [subdir]     copy results/ (or one sub-folder) back
#
# On the cluster data is read straight from Guglielmo's directory (see lib/veto/paths.py).
set -euo pipefail

export PROXY_PORT="${PROXY_PORT:-2226}"
export REMOTE_ROOT="${REMOTE_ROOT:-/disk/users/sqasim/workspace/uzh/Veto_System/veto}"
export EXEC="${EXEC:-bash /disk/lhcb_data/sqasim/images/shared/exec_cudabear_november_2025.sh}"
export GPU_HOST="${GPU_HOST:-farm-gpu3}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CTL="$HERE/../physik_deployment/remote_ctl.sh"
SSH=(ssh -p "$PROXY_PORT" 127.0.0.1)

cmd="${1:-}"; shift || true
case "$cmd" in
  sync)
    "${SSH[@]}" "mkdir -p $REMOTE_ROOT" < /dev/null
    # tar stream: macOS openrsync hangs through the proxy
    tar cf - -C "$HERE" --exclude ./data --exclude ./results --exclude __pycache__ --exclude .DS_Store . \
      | "${SSH[@]}" "cd $REMOTE_ROOT && rm -rf lib bin && tar xf -"
    echo "synced to $REMOTE_ROOT"
    ;;
  run)
    # not on farm-ui: it has ~15 GB of RAM shared by everyone
    ssh -J "127.0.0.1:$PROXY_PORT" "$GPU_HOST" "cd $REMOTE_ROOT && $EXEC python3 $*" < /dev/null
    ;;
  launch)
    run_id="$1"; shift
    "$CTL" launch "$run_id" python3 "$@"
    ;;
  status|peek|attach|kill)
    "$CTL" "$cmd" "$@"
    ;;
  fetch-results)
    mkdir -p "$HERE/results"
    "${SSH[@]}" "cd $REMOTE_ROOT/results && tar cf - ${1:-.}" < /dev/null | tar xf - -C "$HERE/results"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
