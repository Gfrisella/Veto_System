#!/bin/bash
# Run the design evolution entirely on a cluster node (Claude Code CLI and evaluation there), in tmux.
# Log in to claude once on that node first (`claude`, then /login). Extra arguments go to evolve_designs.py.
#
#   ssh farm-gpu3
#   cd /disk/users/sqasim/workspace/uzh/Veto_System/veto && evolve/run_on_cluster.sh --generations 25 --children 8
#   tmux attach -t veto_evolve_<time>               # the script prints the name; detach with Ctrl-b d
set -euo pipefail
cd "$(dirname "$0")/.."
command -v claude > /dev/null || { echo "claude not on PATH"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/evolve/run_$STAMP.log"
SESSION="veto_evolve_$STAMP"
mkdir -p results/evolve
tmux new-session -d -s "$SESSION" \
  "python3 evolve/evolve_designs.py --host local $* 2>&1 | tee $LOG; echo; echo done, press enter; read"
echo "started: tmux attach -t $SESSION   (log $LOG)"
