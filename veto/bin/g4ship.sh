#!/bin/bash
# Run a command with the SHiP CVMFS Geant4 10.7.3 (FairShip's version) on the host, no container needed:
#   bin/g4ship.sh build                         # compile cpp/g4vessel.cc -> cpp/build_ship (host g++ / python3)
#   bin/g4ship.sh python3 bin/g4_vessel_transport.py --interactions ... --workers 32
# Uses the host compiler and Python (the CVMFS GCC 11 headers do not work with this host's glibc), linked against the
# CVMFS Geant4 libraries (rpath), and sources geant4.sh for the data sets.
set -euo pipefail
G4=/cvmfs/ship.cern.ch/25.09/sw/slc9_x86-64/GEANT4/v10.7.3-local1
XERCES=/cvmfs/ship.cern.ch/25.09/sw/slc9_x86-64/XercesC/v3.3.0-local1
HERE="$(cd "$(dirname "$0")/.." && pwd)"
set +eu; source "$G4/bin/geant4.sh" >/dev/null 2>&1; set -eu   # geant4.sh can return non-zero in non-interactive shells
if [ "${1:-}" = "build" ]; then
    PY=/usr/bin/python3
    EXT=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
    INC=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
    mkdir -p "$HERE/cpp/build_ship"
    for mod in g4vessel g4shield g4dis; do                # vessel transport, MS7 shield-iron transport, DIS library
    /usr/bin/g++ -O2 -shared -fPIC -std=c++17 -w -I"$HERE/external/pybind11_pkg/pybind11/include" -I"$INC" \
        -I"$G4/include/Geant4" -I"$XERCES/include" "$HERE/cpp/$mod.cc" -o "$HERE/cpp/build_ship/$mod$EXT" \
        -L"$G4/lib64" -Wl,-rpath,"$G4/lib64" -lG4physicslists -lG4run -lG4event -lG4tracking -lG4parmodels \
        -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps \
        -lG4intercoms -lG4global -lG4clhep -lG4expat -lG4zlib
    done
    ls -la "$HERE/cpp/build_ship/"
    exit 0
fi
exec "$@"
