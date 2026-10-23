#!/bin/bash
# Copy files from CERN (lxplus) to the physik cluster with live progress. Run it on farm-ui / farm-gpu*.
#
#   bin/fetch_from_lxplus.sh                        # default: Guglielmo's withCharmandBeauty *_mu.root files
#   bin/fetch_from_lxplus.sh '/afs/cern.ch/.../*_nu.root' [DEST]
#
# One login (password / 2FA asked once, the connection is reused for the listing and the copy).
# Shows the file list and total size first, then one live line: bytes done, %, speed, ETA.
# Resumable: rerun after an interruption, finished files are skipped and partial ones continued.
set -euo pipefail

CERN_USER=${CERN_USER:-sqasim}
PATTERN=${1:-/afs/cern.ch/work/g/gfrisell/background/pythia8_Geant4_10.0_withCharmandBeauty*_mu.root}
DEST=${2:-$(cd "$(dirname "$0")/.." && pwd)/data/source/flux}
HOST=lxplus.cern.ch

SOCK=$(mktemp -u /tmp/lxplus_ctl_XXXXXX)
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$SOCK" -o ControlPersist=10m)
cleanup() { ssh "${SSH_OPTS[@]}" -O exit "$CERN_USER@$HOST" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "$DEST"
echo "== logging in to $CERN_USER@$HOST (once)"
ssh "${SSH_OPTS[@]}" -fN "$CERN_USER@$HOST"

echo "== files matching $PATTERN"
# shellcheck disable=SC2029  # the pattern is meant to expand on lxplus
LISTING=$(ssh "${SSH_OPTS[@]}" "$CERN_USER@$HOST" "ls -l $PATTERN" 2>&1) || { echo "$LISTING"; exit 1; }
echo "$LISTING" | awk '{printf "  %8.2f GB  %s\n", $5/1e9, $NF; n++; s+=$5} END {printf "  %d files, %.2f GB total\n", n, s/1e9}'
echo "== free space at $DEST: $(df -h "$DEST" | awk 'NR==2 {print $4}')"
echo "== copying to $DEST (Ctrl-C is safe, rerun to resume)"

rsync -a --partial --append-verify --info=progress2,name1 --human-readable \
    -e "ssh ${SSH_OPTS[*]}" \
    "$CERN_USER@$HOST:$PATTERN" "$DEST/"

echo "== done"
ls -lh "$DEST"
