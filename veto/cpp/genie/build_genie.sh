#!/bin/bash
# Our own GENIE 3.06.02 (reference configuration: Pythia 6 hadronisation / decays, GENIE's default configs), built
# inside the cudabear container (Ubuntu 24.04, gcc 13.3, gfortran, cmake, ROOT 6.36). LCG_106 cannot be used: its
# gcc 13 fails on any <mutex> on the el9 farm nodes (include-fixed/pthread.h from an older glibc).
# Everything goes into external/genie:
#   pythia6/      libPythia6.so: pristine PYTHIA 6.4.28 + ROOT's C wrappers (as ROOT's old pythia6.tar.gz build)
#   tbb/, gsl/, libxml2/, lhapdf/  not in the container
#   log4cpp/      log4cpp 1.1.4 (upstream release)
#   egpythia6/    libEGPythia6 (TPythia6): removed from ROOT in 6.30, standalone copy (karuboniru/egpythia6-standalone)
#   src/, install/  GENIE Generator R-3_06_02
#   bash cpp/genie/build_genie.sh fetch                    (host: downloads)
#   bin/g4.sh bash cpp/genie/build_genie.sh            (container: build; then bin/genie.sh <command>)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
X=$HERE/external/genie
mkdir -p $X && cd $X
J=${JOBS:-32}
D=$X/downloads
if [ "${1:-}" = fetch ]; then          # on the host (curl segfaults inside the container)
  mkdir -p $D && cd $D
  get() { [ -s "$1" ] || curl -sfL -o "$1" "$2"; }
  # root.cern now sends this one to a CERN SSO login (2026-10-21): if it arrives as a web page, take the copy
  # kept next to the build (PYTHIA6_TGZ=<file>, e.g. an earlier build's downloads/pythia6.tar.gz)
  get pythia6.tar.gz https://root.cern/download/pythia6.tar.gz
  gzip -t pythia6.tar.gz 2> /dev/null || { rm -f pythia6.tar.gz; [ -n "${PYTHIA6_TGZ:-}" ] && cp "$PYTHIA6_TGZ" pythia6.tar.gz; }
  gzip -t pythia6.tar.gz 2> /dev/null || { echo "pythia6.tar.gz: no valid archive (root.cern needs a CERN login); set PYTHIA6_TGZ"; exit 1; }
  get pythia6428.f https://pythia.org/download/pythia6/pythia6428.f
  get gsl.tar https://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz
  get libxml2.tar https://download.gnome.org/sources/libxml2/2.12/libxml2-2.12.9.tar.xz
  get tbb.tar https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2021.13.0.tar.gz
  get lhapdf.tar https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.5.4.tar.gz
  get log4cpp.tar "https://sourceforge.net/projects/log4cpp/files/log4cpp-1.1.x%20%28new%29/log4cpp-1.1/log4cpp-1.1.4.tar.gz/download"
  [ -d egpythia6 ] || git clone -q --depth 1 https://github.com/karuboniru/egpythia6-standalone egpythia6
  [ -d genie ] || git clone -q --depth 1 --branch R-3_06_02 https://github.com/GENIE-MC/Generator genie
  ls -la $D; exit 0
fi
command -v root-config > /dev/null || { echo "run inside the container: bin/g4.sh bash $0 (after: bash $0 fetch)"; exit 1; }

# ---------------------------------------------------------------- oneTBB: the container's ROOT needs libtbb.so.12
# (libImt, used by Tree, Hist, MathCore, ...) but the container has none
if [ ! -f tbb/install/.done ]; then
  rm -rf tbb && mkdir -p tbb/src && tar xf $D/tbb.tar -C tbb/src --strip-components 1
  cmake -S tbb/src -B tbb/build -DCMAKE_INSTALL_PREFIX=$X/tbb/install -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF \
        -DTBB_STRICT=OFF > tbb_cmake.log 2>&1 || { tail -20 tbb_cmake.log; exit 1; }
  cmake --build tbb/build -j $J > tbb_build.log 2>&1 || { tail -30 tbb_build.log; exit 1; }
  cmake --install tbb/build > /dev/null && touch tbb/install/.done
  echo "built oneTBB"
fi
TBBLIB=$(dirname $(ls $X/tbb/install/lib*/libtbb.so.12 | head -1))
export LD_LIBRARY_PATH=$TBBLIB:${LD_LIBRARY_PATH:-}

# ---------------------------------------------------------------- GSL and libxml2 (autotools)
build_auto() {   # name tarball configure-args...
  local name=$1 tarball=$2; shift 2
  [ -f $name/install/.done ] && return
  rm -rf $name && mkdir -p $name/src && cd $name
  tar xf $D/$tarball -C src --strip-components 1 && cd src
  [ -x configure ] || cd */           # tarballs with ./name/ paths
  ./configure --prefix=$X/$name/install "$@" > ../configure.log 2>&1 || { tail -20 ../configure.log; exit 1; }
  make -j $J > ../build.log 2>&1 || { tail -30 ../build.log; exit 1; }
  make install > ../install.log 2>&1 && touch $X/$name/install/.done
  cd $X
  echo "built $name"
}
build_auto gsl gsl.tar
build_auto libxml2 libxml2.tar --without-python --without-lzma --without-zlib --without-icu
build_auto log4cpp log4cpp.tar
# LHAPDF 6: GENIE 3.06 compiles its HEDIS module (high-energy tunes) against it even when disabled; G18_02a itself
# uses GENIE's built-in GRV98 (no PDF sets needed)
build_auto lhapdf lhapdf.tar --disable-python

# ---------------------------------------------------------------- PYTHIA 6.4.28 shared library
if [ ! -f pythia6/lib/libPythia6.so ]; then
  mkdir -p pythia6/build pythia6/lib && cd pythia6/build
  tar xzf $D/pythia6.tar.gz
  rm -f pythia6/pythia6416.f
  cp $D/pythia6428.f pythia6/
  cd pythia6
  rm -f *.o
  gfortran -O2 -fPIC -fno-automatic -std=legacy -c pythia6428.f
  gfortran -O2 -fPIC -fno-second-underscore -c tpythia6_called_from_cc.F
  gcc -O2 -fPIC -fcommon -c pythia6_common_address.c         # common blocks defined in C: -fcommon for gcc >= 10
  gfortran -shared -o ../../lib/libPythia6.so *.o
  cd $X
  echo "built pythia6/lib/libPythia6.so"
fi

L4LIB=$(dirname $(ls $X/log4cpp/install/lib*/liblog4cpp.so | head -1))

# ---------------------------------------------------------------- EGPythia6 (TPythia6)
if [ ! -f egpythia6/install/lib64/libEGPythia6.so ] && [ ! -f egpythia6/install/lib/libEGPythia6.so ]; then
  rm -rf egpythia6 && mkdir -p egpythia6 && cp -r $D/egpythia6 egpythia6/src
  # the container's ROOT was configured with an external VDT that is not installed; ROOT::Core/EG/Graf/Physics
  # (all EGPythia6 links) do not use it: a stub satisfies ROOTConfig's find_dependency(Vdt)
  mkdir -p $X/vdt_stub/include/vdt && printf "#define VDT_VERSION_MAJOR 0\n#define VDT_VERSION_MINOR 4\n#define VDT_VERSION_PATCH 4\n" > $X/vdt_stub/include/vdt/vdtMath.h
  cmake -S egpythia6/src -B egpythia6/build -DCMAKE_INSTALL_PREFIX=$X/egpythia6/install -DPYTHIA_LIB_PATH=$X/pythia6/lib \
        -DVDT_INCLUDE_DIR=$X/vdt_stub/include -DVDT_LIBRARY=$(ls /usr/lib/x86_64-linux-gnu/libm.so.6) \
        -DCMAKE_BUILD_TYPE=Release > egpythia6_cmake.log 2>&1 || { tail -20 egpythia6_cmake.log; exit 1; }
  cmake --build egpythia6/build -j $J > egpythia6_build.log 2>&1 || { tail -30 egpythia6_build.log; exit 1; }
  cmake --install egpythia6/build > /dev/null
  echo "built EGPythia6"
fi
EGLIB=$(dirname $(ls $X/egpythia6/install/lib*/libEGPythia6.so | head -1))

# ---------------------------------------------------------------- GENIE
if [ ! -x install/bin/gevgen ]; then
  [ -d src ] || cp -r $D/genie src
  export GENIE=$X/src
  export LD_LIBRARY_PATH=$X/pythia6/lib:$L4LIB:$EGLIB:$X/gsl/install/lib:$X/libxml2/install/lib:$X/lhapdf/install/lib:${LD_LIBRARY_PATH:-}
  export PATH=$X/gsl/install/bin:$PATH          # gsl-config
  # TPythia6 headers / library outside ROOTSYS: give them to the compiler and linker
  export CPATH=$X/egpythia6/install/include:${CPATH:-}
  export LIBRARY_PATH=$EGLIB:$X/pythia6/lib:${LIBRARY_PATH:-}
  cd src
  ./configure --prefix=$X/install --disable-lhapdf5 --enable-lhapdf6 \
      --with-lhapdf6-inc=$X/lhapdf/install/include --with-lhapdf6-lib=$X/lhapdf/install/lib \
      --with-pythia6-lib=$X/pythia6/lib \
      --with-log4cpp-inc=$X/log4cpp/install/include --with-log4cpp-lib=$L4LIB \
      --with-libxml2-inc=$X/libxml2/install/include/libxml2 --with-libxml2-lib=$X/libxml2/install/lib > ../genie_configure.log 2>&1
  make -j $J > ../genie_build.log 2>&1 || { tail -40 ../genie_build.log; exit 1; }
  make install > ../genie_install.log 2>&1
  cd $X
  echo "built GENIE $(cat src/VERSION)"
fi
ls -la install/bin | head
