#!/bin/bash
# Run a command with our own GENIE 3.06.02 (cpp/genie/build_genie.sh: reference configuration, Pythia 6, default
# configs; cpp/genie/config overrides: charm and tau decayed) inside the cudabear container. Splines: data/interactions/genie
# (bin/genie_splines.py, official G18_02a set).
#   bin/genie.sh gevgen -n 1000 -p 14 -t 1000020040 -e 20 --tune G18_02a_00_000 --cross-sections data/interactions/genie/gxspl-fvs.xml
# GENIE lives in external/genie of this project (bin/.. = veto/), next to cpp/.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
X=$HERE/external/genie
L=$X/install/lib:$X/pythia6/lib:$X/log4cpp/install/lib:$X/egpythia6/install/lib:$X/gsl/install/lib:$X/libxml2/install/lib
L=$L:$X/lhapdf/install/lib:$(dirname $(ls $X/tbb/install/lib*/libtbb.so.12 | head -1)):$(dirname $(ls $X/egpythia6/install/lib*/libEGPythia6.so | head -1))
exec bash /disk/lhcb_data/sqasim/images/shared/exec_cudabear_november_2025.sh \
    env GENIE=$X/src PATH=$X/install/bin:/hepsw/root/bin:/usr/bin:/bin LD_LIBRARY_PATH=$L:/hepsw/root/lib \
        ROOT_INCLUDE_PATH=$X/src/src:$X/egpythia6/install/include GXMLPATH=$HERE/cpp/genie/config "$@"
