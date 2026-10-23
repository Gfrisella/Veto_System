#!/bin/bash
# Build the Pythia-6 driver (cpp/tailgen/pythia6_mesons.f) against PYTHIA 6.4.28 (ShipSoft fork, v6.4.28-ship1),
# cloned into external/pythia6, inside the cudabear container (gfortran 13):
#   bin/g4.sh bash cpp/tailgen/build_pythia6.sh
set -euo pipefail
cd "$(dirname "$0")"
P6=../../external/pythia6
# git https crashes inside the container (as curl does): if the clone fails, run it once on the host
[ -f "$P6/pythia-6.4.28.f" ] || git clone -q --depth 1 --branch v6.4.28-ship1 https://github.com/ShipSoft/pythia6 "$P6" || {
  echo "clone failed; on the host: git clone --depth 1 --branch v6.4.28-ship1 https://github.com/ShipSoft/pythia6 external/pythia6"; exit 1; }
mkdir -p build
gfortran -O2 -std=legacy -fno-automatic -c "$P6/pythia-6.4.28.f" -o build/pythia-6.4.28.o 2> build/pythia6_compile.log || { tail -20 build/pythia6_compile.log; exit 1; }
gfortran -O2 -std=legacy -fno-automatic -ffixed-line-length-none pythia6_mesons.f build/pythia-6.4.28.o -o build/pythia6_mesons
echo "built build/pythia6_mesons"
gfortran -O2 -std=legacy -fno-automatic -ffixed-line-length-none pythia6_charm.f build/pythia-6.4.28.o -o build/pythia6_charm
echo "built build/pythia6_charm"
gfortran -O2 -std=legacy -fno-automatic -ffixed-line-length-none pythia6_dis.f build/pythia-6.4.28.o -o build/pythia6_dis
echo "built build/pythia6_dis"
