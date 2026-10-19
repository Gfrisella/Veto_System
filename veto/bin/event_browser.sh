#!/bin/bash
# Veto event browser: an interactive HTML page that steps through the SST1 muons one by one, by
# category (wall bounce, floor bounce, selected, other), in the full 3D cavern geometry, for a few
# designs side by side, with per-category rates, confusion matrices and the fake-signal rate.
#
#   bin/event_browser.sh rebuild  [TAG] [DESIGN...]   page from an existing validation (seconds, farm-ui is fine)
#   bin/event_browser.sh validate TAG DESIGN... [-- validate_design.py options]
#                                                     full-statistics validation on Slurm, then rebuild (~15 min / design)
#   bin/event_browser.sh serve    [PORT]              serve results/validate/ on localhost:PORT (default 8765)
#   bin/event_browser.sh list                         validations available, with their designs
#
# TAG: name of the validation (results/validate/<TAG>.json); default bottom_main.
# DESIGN: for rebuild, design-name prefixes inside that validation (default D6_straw OPT1_tile2_flames OPT2_rank1);
#         for validate, design names (veto/designs/) or JSON files.
# Output: results/validate/event_browser_<TAG>.html (data inlined; plotly and fonts come from CDNs, so it needs internet).
#
# Defaults of validate (see bin/validate_design.py): no timing detector, track time 5 ns, +-5 sigma,
# pairs = two muons of the same 1 s spill, 3 M muons per type, seed 2.
set -euo pipefail
cd "$(dirname "$0")/.."
VETO_DIR="$PWD"
export VETO_STORE="${VETO_STORE:-/disk/users/sqasim/workspace/uzh/Veto_System/veto/data/store}"
export OMP_NUM_THREADS=1

cmd="${1:-}"; shift || true
case "$cmd" in
  rebuild)
    tag="${1:-bottom_main}"; shift || true
    if [ ! -f "results/validate/$tag.json" ]; then
      echo "no results/validate/$tag.json; run: bin/event_browser.sh list" >&2; exit 1
    fi
    if [ $# -gt 0 ]; then python3 bin/make_event_browser.py --tag "$tag" --designs "$@"
    else python3 bin/make_event_browser.py --tag "$tag"; fi
    ;;
  validate)
    tag="${1:?give a TAG}"; shift
    designs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do designs+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    [ ${#designs[@]} -gt 0 ] || { echo "give at least one design" >&2; exit 1; }
    # 12 workers need ~55 GB (4.5 GB each); farm-wn44 (hep) is usually free, else the standard partition
    part="${PARTITION:-hep}"; node=(); [ "$part" = hep ] && node=(-w farm-wn44)
    j1=$(sbatch --parsable -p "$part" "${node[@]}" -c 12 --mem=55G -t 12:00:00 -J "val_$tag" \
         -o "results/logs/val_${tag}_%j.out" \
         --wrap "cd $VETO_DIR && python3 bin/validate_design.py ${designs[*]} --tag $tag --workers 12 --chunk 10000 $*")
    names=(); for d in "${designs[@]}"; do b=$(basename "$d" .json); names+=("$b"); done
    j2=$(sbatch --parsable -p express -c 1 --mem=8G -t 0:30:00 -J "browser_$tag" --dependency=afterok:$j1 \
         -o "results/logs/browser_${tag}_%j.out" \
         --wrap "cd $VETO_DIR && python3 bin/make_event_browser.py --tag $tag --designs ${names[*]}")
    echo "validation job $j1, browser job $j2 (after it): results/validate/event_browser_$tag.html"
    echo "progress: squeue -u $USER; tail -f results/logs/val_${tag}_$j1.out"
    echo "note: the browser matches designs by the 'name' inside each JSON; if a name differs from the file name,"
    echo "      rerun: bin/event_browser.sh rebuild $tag <name prefixes>"
    ;;
  serve)
    port="${1:-8765}"
    echo "serving $VETO_DIR/results/validate on http://localhost:$port/ (VS Code forwards the port; Ctrl-C to stop)"
    ls results/validate/event_browser*.html 2>/dev/null | sed "s#results/validate/#  http://localhost:$port/#"
    exec python3 -m http.server "$port" --bind 127.0.0.1 --directory results/validate
    ;;
  list)
    for f in results/validate/*.json; do
      case "$f" in *_paths.json|*_browser.json) continue ;; esac
      python3 - "$f" <<'EOF'
import json, sys
p = sys.argv[1]
try:
    r = json.load(open(p))["results"]
except Exception:
    sys.exit(0)
s = r[0].get("settings", {})
print(f"{p.split('/')[-1][:-5]:<22} {', '.join(x['design'] for x in r)}   "
      f"(track {s.get('sigma_t_track_ns')} ns, +-{s.get('n_sigma_t')} sigma, coincidence {s.get('coincidence_ps') or 'none'})")
EOF
    done
    ;;
  *)
    sed -n '2,19p' "$0"; exit 1
    ;;
esac
