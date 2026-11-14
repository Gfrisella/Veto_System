#!/bin/bash
# Re-run batch (ISSUES action list step 5; prepared 12 Nov 2026). Submit from a farm node with Slurm (farm-gpu3/4/5):
#     bash bin/batch/submit_rerun.sh            # writes results_rerun/batch/*.cmd and submits 7 stages
#     DRY=1 bash bin/batch/submit_rerun.sh      # only writes the command files and prints the sbatch lines
# Everything goes to $VETO_RESULTS (default <veto>/results_rerun): the current results stay untouched until checked.
# Stages (array jobs; each line of a .cmd file is one task; dependencies afterok):
#   S1 maps        rate maps + peak maps + tile map from the 3220-spill store, vessel-DIS parent store
#   S3 ms7         MS7 muon chain on the V0-stable library (10 x 2e6, 5 V0 copies) + MS7 neutrinos (2026 flux)
#   S4 inputs      pair inputs from the full store (after S1), then cross-run pairs (S4b, A1)
#   S2 em          EM debris on the Nov-4 MS7 parents, 2e6 muons, incl. the tile plane (after S4: map binning)
#   S5 interact    per design: muon DIS (helium, wall, air, air upstream with targeted draws, planes) + neutrino DIS
#   S6 stores      event stores (MS7, vessel per design) + forced K_L decays (C7) (after S3, S5)
#   S7 evaluate    per design: run_reco full / partial IP with EM debris and the B4 signal set, event-store evaluation,
#                  DIS x mu (D9), self-veto (D2); closure (A1); rock x mu (D7) for D6_straw_mod (after S2, S4b, S6)
set -euo pipefail
export VETO_CODE=${VETO_CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
source "$VETO_CODE/bin/batch/env.sh"
B=$VETO_RESULTS/batch; mkdir -p "$B/logs"
R=$VETO_RESULTS
name() { basename "$1" .json; }

# ---------------- command files
cat > $B/S1_maps.cmd <<EOT
python3 bin/make_design_explorer.py --dataset $VETO_MUON_DATASET --out $VETO_RATE_MAPS
python3 bin/make_peak_maps.py --store $VETO_MUON_DATASET --workers 16
python3 bin/build_dis_parent_store.py --store $VETO_MUON_DATASET
python3 -c "import sys; sys.path.insert(0, 'lib'); from veto import reco, geometry as g; reco.tile_rate_map(g.Z_UBT_TILES, g.UBT_TILES_HALF_X, g.UBT_TILES_HALF_Y)"
EOT
: > $B/S3_ms7.cmd
for s in $(seq 501 510); do
  echo "g4:python3 bin/shield_dis.py generate --kind muon --n 2000000 --seed $s --mu-lib $MU_LIB_V0STABLE --v0-copies 5 --workers 16 --tag _v0s_s$s" >> $B/S3_ms7.cmd
done
echo "g4:python3 bin/shield_dis.py generate --kind nu --n 500000 --seed 601 --v0-copies 100 --workers 16 --tag _nu2026" >> $B/S3_ms7.cmd
cat > $B/S4_inputs.cmd <<EOT
python3 bin/make_reco_inputs.py --dataset $VETO_MUON_DATASET --maps $VETO_RATE_MAPS --out $VETO_RECO_INPUTS --device cpu --pair-block 20000
EOT
cat > $B/S4b_xrun.cmd <<EOT
python3 bin/make_crossrun_pairs.py
EOT
cat > $B/S2_em.cmd <<EOT
g4:python3 bin/em_debris.py --n 2000000 --workers 16
EOT
: > $B/S5_interact.cmd
for d in $DESIGNS; do n=$(name $d)
  echo "python3 bin/run_interactions.py --design $d --kind muon --dataset $DIS_PARENTS --air-upstream-draws 2e6 --workers 16 --tag _rerun --out $R/interactions/${n}_muon_rerun.json" >> $B/S5_interact.cmd
  echo "python3 bin/nu_dis_apply.py --design $d --workers 16 --out $R/interactions/apply2026_${n}.npz" >> $B/S5_interact.cmd
done
: > $B/S6_stores.cmd
echo "python3 bin/evstore_build.py shield \"$R/shield_dis/shield_muon_v0s_s5*.npz\" --out $R/evstore/ms7_muon_v0s" >> $B/S6_stores.cmd
echo "python3 bin/evstore_build.py shield \"$R/shield_dis/shield_nu_nu2026.npz\" --out $R/evstore/ms7_nu2026" >> $B/S6_stores.cmd
for d in $DESIGNS; do n=$(name $d)
  echo "python3 bin/evstore_build.py apply $R/interactions/${n}_muon_interactions_rerun.npz --out $R/evstore/vessel_mu_${n} && python3 bin/evstore_redecay.py $R/evstore/vessel_mu_${n} --natural --n 10 --out $R/evstore/vessel_mu_${n}_kl10" >> $B/S6_stores.cmd
  echo "python3 bin/evstore_build.py apply $R/interactions/apply2026_${n}.npz --out $R/evstore/vessel_nu2026_${n}" >> $B/S6_stores.cmd
done
: > $B/S7_eval.cmd
for d in $DESIGNS; do n=$(name $d)
  for ip in full partial; do
    echo "python3 bin/run_reco.py $d --ip-cut $ip --em-debris --signal eventcalc --multiple-scattering --out reco_${n}_${ip}.json" >> $B/S7_eval.cmd
  done
  echo "python3 bin/evstore_eval.py --design $d --store $R/evstore/vessel_mu_${n} $R/evstore/vessel_mu_${n}_kl10 $R/evstore/vessel_nu2026_${n} $R/evstore/ms7_muon_v0s $R/evstore/ms7_nu2026 --ip-tail --out $R/evstore/eval_${n}.json" >> $B/S7_eval.cmd
  echo "python3 bin/dis_combinatorial.py --design $d --stores $R/evstore/ms7_muon_v0s $R/evstore/vessel_mu_${n} --mu-sample 300000 --out $R/reco/dis_comb_${n}.json" >> $B/S7_eval.cmd
  echo "python3 bin/d2_selfveto.py --design $d --stores $R/evstore/vessel_nu2026_${n} --out $R/interactions/d2_selfveto_${n}.json" >> $B/S7_eval.cmd
done
echo "python3 bin/closure_check.py $DESIGNS --out closure_rerun.json" >> $B/S7_eval.cmd
echo "python3 bin/rock_combinatorial.py --design designs/modules/D6_straw_mod.json --rock-mu $ROCK_MU --out $R/reco/rock_comb_D6_straw_mod.json" >> $B/S7_eval.cmd

# ---------------- submission
sub() {   # sub <stage> <dependency job ids, space separated or empty> [extra sbatch args]
  local st=$1 dep=$2; shift 2
  local n; n=$(grep -c . "$B/$st.cmd")
  local args=(--parsable --job-name="rr_$st" --array=0-$((n - 1)) --output="$B/logs/${st}_%a.log"
              --export=ALL,CMDFILE=$B/$st.cmd,VETO_CODE=$VETO_CODE "$@")
  dep=$(echo $dep)                                   # drop empty entries
  [ -n "$dep" ] && args+=(--dependency="afterok:$(echo $dep | tr ' ' ':')")
  if [ -n "${DRY:-}" ]; then echo "sbatch ${args[*]} bin/batch/task.sbatch" >&2; echo "DRY_$st"; else sbatch "${args[@]}" bin/batch/task.sbatch; fi
}
cd "$VETO_CODE"
# STAGES: which stages to submit (default all). A stage not submitted here is assumed done (or run elsewhere, e.g. S4 / S4b
# on the farm-gpu3 GPUs with bin/batch/run_local_gpu.sh); J_<stage>=<job id> makes later stages wait for an earlier job.
STAGES=${STAGES:-"S1 S3 S4 S4b S2 S5 S6 S7"}
want() { [[ " $STAGES " == *" $1 "* ]]; }
j1=${J_S1:-}; j3=${J_S3:-}; j4=${J_S4:-}; j4b=${J_S4b:-}; j2=${J_S2:-}; j5=${J_S5:-}; j6=${J_S6:-}
want S1 && j1=$(sub S1_maps "")
want S3 && j3=$(sub S3_ms7 "" --mem=128G)          # 16 Geant4 workers peak at 103 GB (12 Nov)
want S4 && j4=$(sub S4_inputs "$j1" --mem=200G)
want S4b && j4b=$(sub S4b_xrun "$j4" --mem=128G)
want S2 && j2=$(sub S2_em "$j4" --cpus-per-task=32 --mem=160G)
want S5 && j5=$(sub S5_interact "$j4" --mem=128G)
want S6 && j6=$(sub S6_stores "$j3 $j5" --mem=96G)
want S7 && j7=$(sub S7_eval "$j2 $j4b $j6" --mem=128G)
echo "submitted ($STAGES): S1 ${j1:--} S3 ${j3:--} S4 ${j4:--} S4b ${j4b:--} S2 ${j2:--} S5 ${j5:--} S6 ${j6:--} S7 ${j7:--}; logs in $B/logs; outputs in $R"
