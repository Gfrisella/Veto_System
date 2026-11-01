# Changelog

Changes to the veto optimisation code, newest first. Each entry gives the branch, the base commit and
what changed. Open problems are in `ISSUES.txt`. Earlier entries (2026-10-19/20) are in this file on branch
`veto-all-sst1`.

## 2026-11-01 — branch `veto-shield-dis` (commit 700eb10, with the 2026-10-30 entries)

### Bug fix: own UBT hits of tracks born downstream (`lib/veto/reco.py` `own_hits`)
- With multiple scattering on, `own_hits` recomputed `ok = crossed & inside` and dropped `true_crossing`'s condition
  that a track must start upstream of the UBT (`az < Z_UBT + A_TOLERANCE`). Every track then got a UBT hit at the
  (x, y) of its start, so DIS tracks born inside the vessel could veto themselves (ν DIS in helium with the minimal
  UBT: 9.5 % fake vetoes). Muons from the transport start at the UBT and are not affected; fins were not affected
  (`path_at_z` returns NaN for tracks that do not cross them).
- Fix: `crossed &= (apz > 0) & (az < Z_UBT + A_TOLERANCE)` for the UBT.
- Veto step rerun on the stored D6_straw files (`results/dis_per_track/rerun_ownhits_fix.sh`, old outputs
  `*.before_ownhits_fix.json`), per 15 y after veto b (old selection, measured IP < 10 cm, <= 2 tracks):
  ν vessel 0.70 -> 0.93 (of 1.03 selected), MS7 μ K_L 2.1e4 -> 3.4e4 (of 3.4e4: no veto), MS7 ν 0.77 -> 0.83
  (of 1.68), μ He 32 -> 37 ± 12 (of 1 913; the random stream changes with the number of hits, within errors).

## 2026-10-30 — branch `veto-shield-dis` (on top of `e35533a`, not committed yet)

### DIS sampling: one draw per crossing / per track (Guglielmo, 2026-10-30)
- `lib/fastveto/dis.py` `run`: a draw count M = −k gives k draws on EVERY crossing of the group, weight w·P / k each
  (default M > 0 unchanged: M draws picked with probability w·P / S, weight S / M). Both unbiased; per crossing
  every track contributes once instead of the heaviest tracks being redrawn many times.
- `bin/shield_dis.py generate --per-track`: one interaction per parent line, weight w·P.
- Rerun: `results/dis_per_track/run.sh` (μ vessel He / PVC / planes per crossing, air 2e5 draws as before; ν every
  target per crossing; shield μ / ν per track). The w = 1 alternative (flattened weights) was rejected: the
  w = 768.75 minimum-bias class is 40.5 % of the true shield DIS rate and would become 0.8 %.

### DIS in the last 50 cm of the muon shield (`docs/shield_iron_dis.md`)
- `lib/veto/shield_iron.py` (new): MS7 iron box, field map loader / lookup, chord, Geant4 set-up; input muons
  `Muons_2950_50cm_before_end_MS.pkl` at true z = 29.20 m (stored z − 0.30 m).
- `cpp/g4shield.cc` (new) + `cpp/CMakeLists.txt`, `cpp/build.sh`: Geant4 transport of final states through the iron
  with the field map, recorded at z = 29.90 m. `build.sh` finds pybind11 in `external/pybind11_pkg` if the container
  has none.
- `bin/shield_dis.py` (new): prepare / generate (forced interactions, transport, forced V0 decays in the vessel);
  `bin/shield_dis_smoke.py` (new) checks.
- `bin/dis_library.py`: iron (G4_Fe) material; `--only-g4 MATERIAL --base LIB` adds Geant4 tables to an existing
  library. `lib/fastveto/materials.py`: iron.
- `bin/genie_splines.py`, `bin/nu_library.py`: Fe56; `--nuclei`, `--nus` options. `lib/fastveto/nudis.py`: iron.
- `lib/fastveto/decay.py`: one decay round factored out (`_decay_round`, `decay_all` unchanged), `force_decay`
  (decay exactly at the given point) with K_L channels.
- `lib/veto/interactions.py`: target codes 3 `shield_iron`, 4 `shield_iron_v0`.
- `bin/g4.sh`: puts `external/pylib` (numpy, h5py, scipy for the container's python) on PYTHONPATH if present.
