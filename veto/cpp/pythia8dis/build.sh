#!/bin/bash
# Build cpp/pythia8dis/pythia8_dis against the SHiP CVMFS Pythia 8.230 with the host g++.
set -euo pipefail
P8=/cvmfs/ship.cern.ch/25.09/sw/slc9_x86-64/pythia/v8230-ship-local1
cd "$(dirname "$0")"
/usr/bin/g++ -O2 -std=c++11 -w -I$P8/include pythia8_dis.cc -o pythia8_dis -L$P8/lib -Wl,-rpath,$P8/lib -lpythia8 -ldl
ls -la pythia8_dis
