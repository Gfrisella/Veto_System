# veto — muon background for the SHiP veto system

A cleaned-up version of the analysis in the repository root (`analysis.ipynb`,
`analysis_multifiles.ipynb`, `Optimization_SBT.ipynb`, `Analysis_*.py`,
`Muon_Flames_rate.py`, `temp1.py`). Shared code is in `lib/veto`; every notebook
or script became a command-line program in `bin/`. The original files in the
root are left as they were.

```
veto/
  lib/veto/
    geometry.py    z planes, decay vessel, UBT/SBT, SST1, timing plane, spectrometer, TRY6 shield
    config.py      default cuts (p > 1 GeV, timing acceptance, DOCA < 2 cm, IP < 2.5 m, ...)
    paths.py       where inputs are read from / results written to
    io.py          output pickles, vessel dataset, 3D distribution, field map
    tracks.py      straight-line extrapolation, back-tracking to the vessel walls, categories
    weights.py     repeated-evt_n weight corrections
    pairs.py       all pairs / time-window pairs / KD-tree candidates, DOCA + vertex
    selection.py   single-muon and pair cut flows
    plotting.py    matplotlib style, geometry overlays, diagnostic plots
    maps.py        chunk-fillable histograms: SBT wall maps, side views, x-y maps
    viz3d.py       interactive plotly 3D view
  bin/
    run_selection.py          pair background: --pairing per-file | time-window | kdtree
    make_vessel_dataset.py    -> muon_vessel_dataset.npz (input to the SBT optimisation)
    make_muon_distribution.py -> muon_distribution_3d.pkl (was Muon_Flames_rate.py)
    plot_muon_distribution.py plots of the 3D distribution
    optimize_sbt.py           SBT optimisation entry point (summary only so far)
    visualize_detector.py     3D html + side views + UBT plane + unfolded SBT walls + field map
                              (--source sst1 | ubt)
    convert_ubt_data.py       UBT_data_TRY6.pkl -> memory-mapped column store + 2 M subsample
    analyse_ubt_plane.py      chunked pass over the 150 M UBT-plane muons
    convert_sst1_outputs.py   all 20 001 output pickles -> one column store (78 s on 32 cores)
    benchmark.py              timing of the building blocks, CPU vs GPU
    benchmark_designs.py      per-design cost for arbitrary UBT/SBT geometries (truth crossings)
    benchmark_digitization.py per-design cost with digitization + hit-to-track matching
    fetch_data.sh             copy N output files (+ derived datasets) from the cluster
    remote.sh                 sync code / run / launch tmux jobs on the cluster
  data/      local copy of the inputs (git-ignored)
  results/   everything we produce (git-ignored)
```

## Where the originals went

| original | now |
|---|---|
| `Analysis_singlefiles.py`, `analysis_multifiles.ipynb` | `run_selection.py --pairing per-file` |
| `analysis.ipynb` (selection cells), `Analysis_multifiles.py` | `run_selection.py --pairing time-window` (`--opposite-charge false` reproduces the notebook's DOCA cell) |
| `temp1.py` | `run_selection.py --pairing kdtree` |
| `analysis.ipynb` cell 21 | `make_vessel_dataset.py` |
| `analysis.ipynb` cell 32 | `plot_muon_distribution.py` |
| `Muon_Flames_rate.py` | `make_muon_distribution.py` |
| `Optimization_SBT.ipynb` | `optimize_sbt.py` |
| `analysis.ipynb` "check" plots | `--plots` on `run_selection.py` |

## Data

Inputs are Guglielmo's, on the physik cluster, in
`/disk/users/gfrise/Optimization_Veto_System` (same directory as
`/home/hep/gfrise/work_dir/Optimization_Veto_System`):

- `outputs/TRY6_UBT_T1_seed_*.pkl`: 20 001 files, one per seed, about 1 300 muons each.
  `planes[0]` is the muon start (z ≈ 0, no z stored), `planes[1]` is at the UBT (z = 30.6 m)
  and `planes[2]` is at SST1 (z = 82 m). Each file also has `weight`, `evt_n` and `pdg`.
- `data/muon_vessel_dataset.npz`, `data/muon_distribution_3d.pkl`, `data/MainSpectrometerField.root`.
- `MainSpectrometerField.h5`: field map of the spectrometer, z = 77–97 m.

On the cluster the code reads that directory directly; locally it reads
`veto/data` (`bin/fetch_data.sh 200 --derived`). Set `VETO_DATA_ROOT` to point
somewhere else and `VETO_RESULTS` to change the output directory. Nothing is
ever written into Guglielmo's directory.

### The UBT-plane dataset

`/disk/users/gfrise/MuonsAndMatter/data/outputs/UBT_data_TRY6.pkl` holds 150 M muons
(4.8 GB) crossing a 6 x 6 m plane at z = 30.6 m: x, y, z, px, py, pz, pdg and weight.
There is no evt_n, so the repeated-event weight correction cannot be applied.
`convert_ubt_data.py` turns it into a column store (`data/store/ubt_TRY6/*.npy`, one
memory-mappable file per column) and a 2 M random subsample (`data/store/ubt_TRY6_sample.npz`,
weights scaled up so totals are kept). On the cluster these live in
`/disk/users/sqasim/workspace/uzh/Veto_System/veto/data/store`.

**Missing data:** muons that never cross that 6 x 6 m plane are not stored, for example
those that bounce off the cavern walls and only reach the SBT or SST further downstream.
The SST1 outputs contain some of them (category `bouncing`).

`analyse_ubt_plane.py` propagates every muon forward in a straight line and clips it
against the vessel. That gives the entry face (UBT window, an SBT wall, or never inside),
the exit face, and whether the line reaches the SST1 acceptance. It keeps only histograms
(`results/ubt/hists.npz`, replot with `--replot`) and every vessel-entering muon
(`results/ubt/entering_vessel.npz`).

## Running

```bash
bin/fetch_data.sh 200 --derived                     # local subset
bin/visualize_detector.py --max-files 100           # results/visualization/
bin/run_selection.py --pairing per-file --max-files 200 --plots
bin/optimize_sbt.py

# full statistics on the cluster (bin/remote.sh run/launch use farm-gpu3; farm-ui is too small)
bin/remote.sh sync
bin/remote.sh run bin/convert_ubt_data.py           # once
bin/remote.sh launch ubt_full bin/analyse_ubt_plane.py
bin/remote.sh fetch-results ubt
bin/visualize_detector.py --source ubt
bin/remote.sh run bin/optimize_sbt.py
bin/remote.sh launch sel_tw bin/run_selection.py --pairing time-window --plots
bin/remote.sh peek sel_tw
bin/remote.sh fetch-results
```

Needs numpy, scipy, matplotlib, h5py and plotly. The cluster container
(`exec_cudabear_november_2025.sh`) has these.

## Things to know

- All tracks are straight lines built from the SST1 state. Inside the vessel there is no
  field, so back-projecting to the vessel walls is fine. The start → UBT → SST1 segments
  drawn in the 3D view only join the true crossing points. The real path bends in the
  shield field and can scatter off the walls.
- `bouncing` uses `|x| > 3 or y < -3.36` in the vessel dataset but `|x| > 3 or |y| > 3`
  in `load_simulation_file` of `analysis.ipynb`. `tracks.categories` follows the dataset.
- The original `backtrack_to_lateral` intersected the *infinite* planes of the walls without
  checking that the point is on the wall. For about 3 % of the SST1 tracks it returned entry
  points off the vessel surface, and most of the "SBT ±y" entries came from this. It is
  replaced by `tracks.vessel_crossings` (exact clipping of the line against the frustum).
  The existing `muon_vessel_dataset.npz` was made with the old version.
- Straight lines ignore multiple scattering in the vessel walls and in air.
- `analysis_multifiles.ipynb` unpacked `backtrack_to_lateral` as `(x, y, z)` although it
  returns `(z, x, y)`, so its `x_decay_vessel`/`z_decay_vessel` columns were swapped.
  Those columns were never used downstream, and the vessel dataset is not affected.
- The time-window pairing uses random times uniform in [0, 1) s, as in the original.
