#!/bin/bash
# Run a command in the cudabear container (Geant4 11.2.2) with what the Geant4 python modules (g4dis, tailtransport) need:
#   bin/g4.sh python3 bin/g4dis_smoke.py
# GLIBC_TUNABLES: Geant4's thread-local storage does not fit the default static TLS block when it is
# loaded as a Python extension ("cannot allocate memory in static TLS block").
exec bash /disk/lhcb_data/sqasim/images/shared/exec_cudabear_november_2025.sh \
    env GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16777216 "$@"
