#!/bin/bash
# Run a command in the cudabear container (Geant4 11.2.2) with what the Geant4 python modules (g4dis, tailtransport) need:
#   bin/g4.sh python3 bin/g4dis_smoke.py
# GLIBC_TUNABLES: Geant4's thread-local storage does not fit the default static TLS block when it is
# loaded as a Python extension ("cannot allocate memory in static TLS block").
# external/pylib: python packages the container lacks (numpy, h5py, scipy), installed with
#   bin/g4.sh python3 -m pip install --target external/pylib numpy==2.1.3 h5py==3.12.1 scipy==1.14.1
PYLIB="$(cd "$(dirname "$0")/.." && pwd)/external/pylib"
[ -d "$PYLIB" ] && export PYTHONPATH="$PYLIB${PYTHONPATH:+:$PYTHONPATH}"
exec bash /disk/lhcb_data/sqasim/images/shared/exec_cudabear_november_2025.sh \
    env GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16777216 ${PYTHONPATH:+PYTHONPATH=$PYTHONPATH} "$@"
