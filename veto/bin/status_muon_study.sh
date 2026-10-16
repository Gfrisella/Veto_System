#!/bin/bash
# Progress of the muon production: seed files, rate, workers, last lines of each GPU log.
#   bin/status_muon_study.sh [OUT_DIR]
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR=${1:-$HERE/data/sim/veto_muons_v1}
n=$(ls "$OUT_DIR"/seed_*.npz 2>/dev/null | wc -l)
echo "$(date '+%F %T')  $n seed files in $OUT_DIR ($(du -sh "$OUT_DIR" 2>/dev/null | cut -f1))"
recent=$(find "$OUT_DIR" -maxdepth 1 -name 'seed_*.npz' -mmin -60 | wc -l)
echo "last hour: $recent seeds -> $(( recent > 0 ? (20000 - n) / recent : -1 )) h to go for 20 000 (at this rate)"
echo "workers: $(pgrep -u "$USER" -fc generate_muon_study.py) generate_muon_study.py processes"
for log in "$OUT_DIR"/logs/*.log; do
    [ -f "$log" ] || continue
    echo "== $(basename "$log"): $(grep -c '^seed ' "$log") seeds, $(grep -ci 'traceback\|error' "$log") errors"
    grep '^seed ' "$log" | tail -1 | cut -c1-160
done
