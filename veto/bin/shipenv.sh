#!/bin/bash
# Run a command in the SHiP CVMFS software (FairShip / ROOT 6.30 / GenFit, release 25.09, slc9 build) on this host:
#   bin/shipenv.sh python3 bin/genfit_straws.py hits.npz fit.npz
# The host (openSUSE Leap 16) has no libcrypt.so.2: a link to libxcrypt's libcrypt.so.1 is put in a private directory.
set -eo pipefail
COMPAT="${TMPDIR:-/tmp}/shipenv-compat-$(id -u)"
mkdir -p "$COMPAT"; [ -e "$COMPAT/libcrypt.so.2" ] || ln -sf /usr/lib64/libcrypt.so.1 "$COMPAT/libcrypt.so.2"
export WORK_DIR=/cvmfs/ship.cern.ch/25.09/sw
source /cvmfs/ship.cern.ch/25.09/sw/slc9_x86-64/FairShip/latest/etc/profile.d/init.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$COMPAT:$LD_LIBRARY_PATH"
exec "$@"
