#!/bin/bash
# Run a command in the SHiP CVMFS software (FairShip / ROOT 6.30, release 25.09, slc9 build) on this host:
#   bin/shipenv.sh python3 <script>            SHIP_RELEASE=26.10 bin/shipenv.sh python3 <script>   (ACTS)
# The host (openSUSE Leap 16) has no libcrypt.so.2: a link to libxcrypt's libcrypt.so.1 is put in a private directory.
set -eo pipefail
COMPAT="${TMPDIR:-/tmp}/shipenv-compat-$(id -u)"
mkdir -p "$COMPAT"; [ -e "$COMPAT/libcrypt.so.2" ] || ln -s /usr/lib64/libcrypt.so.1 "$COMPAT/libcrypt.so.2" 2>/dev/null || true   # race-safe (parallel workers)
# SHIP_RELEASE (default 25.09): 26.10 has ACTS (FairShip python/ACTSReco.py). Its FreeType 2.10.1 links the HOST
# libharfbuzz, which on Leap 16 needs FT_Get_Transform (FreeType >= 2.11) -> "undefined symbol: FT_Get_Transform" on
# import acts / ROOT graphics: the host FreeType (2.13) is preloaded for 26.x releases.
REL="${SHIP_RELEASE:-25.09}"
export WORK_DIR=/cvmfs/ship.cern.ch/$REL/sw
source /cvmfs/ship.cern.ch/$REL/sw/slc9_x86-64/FairShip/latest/etc/profile.d/init.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$COMPAT:$LD_LIBRARY_PATH"
case "$REL" in 26.*) export LD_PRELOAD="/usr/lib64/libfreetype.so.6${LD_PRELOAD:+:$LD_PRELOAD}" ;; esac
exec "$@"
