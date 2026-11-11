#!/bin/bash
# ISSUES B4 (decided 11 Nov 2026): regenerate the signal benchmark set with EventCalc dcdcfc1 (decay volume = the
# FairShip vessel, z 32.4768 - 82.4768 m from the target centre, 6.0 m high at the exit; the LLP starts at the target)
# and import it for the veto chain. ASK GUGLIELMO BEFORE RUNNING (generation; EventCalc_signal_rejection working rules).
#
# Only the 4 benchmarks of signal_eventcalc.BENCHMARK_SET (equal weights): HNL-mu 1 GeV, dark photon 0.9 GeV,
# scalar 2 GeV, ALP-fermion 1 GeV, all c tau = 100 m; 50k sampled points each (as the 4 Nov set), Pythia 8 (LCG_106),
# no exHad.
#   1. keeps the 4 Nov samples: EventCalc_run/outputs -> outputs_vessel6p2 (file names do not change with geometry)
#   2. runs EventCalc_run/run_batch.py for the 4 benchmarks (EventCalc_run is synced with dcdcfc1, NEXT_STEPS.md)
#   3. imports them into data/signal/eventcalc_dcdcfc1 with --vessel-z0-eventcalc 32.4768 (frame: z_FS = z_EC + 0.7932,
#      z_veto = z_EC - 1.3468; production point = the EventCalc origin = target centre)
# Then: VETO_SIGNAL_STORE=<veto>/data/signal/eventcalc_dcdcfc1 python3 bin/run_reco.py <design> --signal eventcalc
set -euo pipefail
R=${EVENTCALC_STUDY:-$HOME/work_dir/EventCalc_signal_rejection}
VETO="$(cd "$(dirname "$0")/.." && pwd)"
N=${N:-50000}
cd "$R/EventCalc_run"
grep -q "32.4768" -r . --include=*.py || { echo "EventCalc_run does not have the dcdcfc1 decay volume (32.4768 m): sync it first"; exit 1; }
if [ -d outputs ] && [ ! -d outputs_vessel6p2 ]; then mv outputs outputs_vessel6p2; echo "4 Nov samples kept in outputs_vessel6p2"; fi
mkdir -p "$R/logs"
(
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
  python3 run_batch.py --llp HNL --mixing 0 1 0 --masses 1.0 --ctaus 100 --nevents $N --exhad off > "$R/logs/b4_HNL_mu.log" 2>&1 &
  python3 run_batch.py --llp Dark-photons --uncertainty central --dp-production combined --masses 0.9 --ctaus 100 --nevents $N --exhad off > "$R/logs/b4_DP.log" 2>&1 &
  python3 run_batch.py --llp Scalar-mixing --masses 2.0 --ctaus 100 --nevents $N --exhad off > "$R/logs/b4_S.log" 2>&1 &
  python3 run_batch.py --llp ALP-fermion --masses 1.0 --ctaus 100 --nevents $N --exhad off > "$R/logs/b4_ALPf.log" 2>&1 &
  wait
)
echo "$(date '+%F %T') B4: 4 benchmarks regenerated with dcdcfc1 (bin/eventcalc_b4.sh, Veto_System)" >> "$R/README.md"
cd "$VETO"
python3 bin/import_eventcalc.py --src "$R/EventCalc_run/outputs" --out "$VETO/data/signal/eventcalc_dcdcfc1" --vessel-z0-eventcalc 32.4768
echo "imported -> $VETO/data/signal/eventcalc_dcdcfc1 (use VETO_SIGNAL_STORE=$VETO/data/signal/eventcalc_dcdcfc1)"
