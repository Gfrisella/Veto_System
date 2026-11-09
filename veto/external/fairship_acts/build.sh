#!/bin/bash
# Rebuild libShipActsTracking.so of the FairShip pixi environment 26.10 with ActsStrawGeometry.cpp from PR #1545
# (straw material from the TGeo straw volumes); every other source of ActsTracking/ is identical to the PR head.
# Compiled with the environment's own compiler (conda gcc 15.3), ACTS 48.2 and ROOT 6.40; output lib/, loaded in
# front of the installed library (bin/fairship_env.sh).  Run: bash external/fairship_acts/build.sh
set -e
H=$(cd "$(dirname "$0")" && pwd)
ENV=${FAIRSHIP_PIXI:-/cvmfs/ship.cern.ch/pixi-envs/fairship/26.10}
source $ENV/activate >/dev/null 2>&1
B=$H/build; mkdir -p $B $H/lib
INC="-I$H/src -I$ENV/shipdata -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/eigen3 -I$(root-config --incdir)"
FLAGS="-std=c++20 -O2 -fPIC -DNDEBUG -w -pthread"
cd $B
rootcling -f G__ShipActsTracking.cxx -s libShipActsTracking.so -rml libShipActsTracking.so \
  -rmf libShipActsTracking.rootmap -I$H/src -I$ENV/shipdata $H/src/ActsStrawTracker.h $H/src/LinkDef.h
objs=""
for f in $H/src/*.cxx $H/src/*.cpp G__ShipActsTracking.cxx; do
  o=$(basename ${f%.*}).o; c++ $FLAGS $INC -c $f -o $o & objs="$objs $o"
done; wait
c++ -shared -o $H/lib/libShipActsTracking.so -Wl,-soname,libShipActsTracking.so.26.10 $objs \
  -L$ENV/build/lib -lShipData -L$CONDA_PREFIX/lib -lActsCore $(root-config --libs) -lGeom
cp libShipActsTracking.rootmap libShipActsTracking_rdict.pcm $H/lib/
for v in .26 .26.10; do ln -sf libShipActsTracking.so $H/lib/libShipActsTracking.so$v; done
echo "built $H/lib/libShipActsTracking.so"
