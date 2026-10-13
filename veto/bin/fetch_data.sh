#!/bin/bash
# Copy a subset of Guglielmo's simulation outputs (and optionally the derived
# datasets) from the cluster into veto/data, mirroring the remote layout.
#
#   bin/fetch_data.sh [n_files] [--derived]
#
# Goes through this project's OTP proxy (physik_deployment/custom_ssh, port 2226).
set -euo pipefail

N_FILES="${1:-100}"
PROXY_PORT="${PROXY_PORT:-2226}"
REMOTE_DATA="${REMOTE_DATA:-/disk/users/gfrise/Optimization_Veto_System}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${VETO_DATA_ROOT:-$HERE/data}"
SSH=(ssh -p "$PROXY_PORT" 127.0.0.1)

mkdir -p "$DEST/outputs" "$DEST/data"

# tar stream instead of rsync: macOS openrsync --files-from hangs through the proxy
files=$(seq 0 $((N_FILES - 1)) | sed 's/.*/TRY6_UBT_T1_seed_&.pkl/' | tr '\n' ' ')
"${SSH[@]}" "cd $REMOTE_DATA/outputs && tar cf - $files" < /dev/null | tar xf - -C "$DEST/outputs"

if [[ " $* " == *" --derived "* ]]; then
  "${SSH[@]}" "cd $REMOTE_DATA/data && tar cf - ." < /dev/null | tar xf - -C "$DEST/data"
fi
echo "Data in $DEST: $(ls "$DEST/outputs" | wc -l) output files"
