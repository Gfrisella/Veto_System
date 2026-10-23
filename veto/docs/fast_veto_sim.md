# fastveto: source muons, interaction libraries, scattering tails

Ported on 2026-10-21 from the fast-veto-sim repository (`/Users/shahrukhqasim/workspace/uzh/fast-veto-sim`, cluster
`/disk/users/sqasim/workspace/uzh/fast-veto-sim`), which can be archived. Nothing in Veto_System reads it anymore.
Physics and validation: `docs/fastveto_physics.md` (fast-veto-sim's downstream_sim.md); Geant4 neutrino notes:
`docs/geant4_neutrino_notes.md`.

```
R  = /disk/users/sqasim/workspace/uzh/Veto_System/veto      (this project on the cluster)
MM = /disk/users/gfrise/MuonsAndMatter                       (cuda-muons, read only)
```

| part | where |
|---|---|
| per-design interaction code (dis, nudis, targets, materials, vessel, kokoulin, decay, neutrinos) | `lib/fastveto/` |
| per-design drivers | `bin/muon_dis_apply.py`, `bin/nu_dis_apply.py`; `bin/run_interactions.py`, `bin/end_to_end.py` call `fastveto` directly |
| neutrino flux preparation | `bin/extract_neutrinos.py` (LCG: `bin/lcg.sh`), `bin/nu_prepare.py`, `bin/fetch_from_lxplus.sh` |
| library builders | `bin/dis_library.py`, `bin/genie_splines.py`, `bin/nu_library.py`; checks `bin/dis_closure.py`, `bin/g4dis_smoke.py`, `bin/plot_dis_library.py`, `bin/dis_check.py` |
| source production at the target | `bin/tail_produce.py`, `bin/build_enriched.py`, `bin/run_slices_any.sh`; validation of the old file `bin/charm_*.py`, `bin/tail_combine.py`, `bin/tail_bins.py`, `bin/plot_reproduction.py`, `bin/old_convention.py`, `bin/ratio_at_creation.py` |
| C++ / Fortran | `cpp/g4dis.cc` (-> `cpp/build/g4dis*.so`), `cpp/tailgen/` (-> `cpp/tailgen/build`), `cpp/genie/` (-> `external/genie`) |
| containers | `bin/g4.sh` (cudabear + Geant4 TLS fix), `bin/lcg.sh` (LCG_106), `bin/genie.sh` (our GENIE) |
| data roots | `paths.interactions_data()` = `data/interactions` (`dis/`, `genie/`, `neutrinos/`), `paths.source_data()` = `data/source` (enriched h5, `tail/`, `flux/`), productions in `data/sim/` |

Builds live in `cpp/build`, `cpp/tailgen/build` and `external/`: `bin/remote.sh sync` replaces `lib/` and `bin/`
only. Build once on the cluster:
```
bin/g4.sh bash cpp/build.sh                      # g4dis
bin/g4.sh bash cpp/tailgen/build_pythia6.sh      # pythia6_dis / _mesons / _charm (clones ShipSoft pythia6 into external/)
bin/g4.sh bash cpp/tailgen/build.sh              # tailtransport (+ pythia_mesons, Pythia 8)
bash cpp/genie/build_genie.sh fetch              # on the host (curl crashes in the container)
bin/g4.sh bash cpp/genie/build_genie.sh          # GENIE 3.06.02 -> external/genie (~10 min)
```

## 1. Source muons: the enriched input file

### The problem
The input of every cuda-muons run is MuonsAndMatter's `$MM/data/muons/full_sample_after_target.h5`
(506,374,000 muons; columns px, py, pz, x, y, z, pdg, weight; weight = muons per spill of 5e13 p.o.t.). Above
~150 GeV it has only ~9k minimum-bias muons, each with weight 768.75. Those muons make most of what reaches the
SST1 / vessel. With the old file, the rate of the class "SST1 selected" rested on ~20 effective input muons
above 100 GeV: a **13.5 % error that no number of cuda-muons seeds can reduce**, because every seed replays the
same input muons (section 4).

### Convention of the file (checked)
- Momentum = **momentum at production in the target** (the 2018 FairShip production), x / y smeared by ~4 cm,
  z set to -0.0045 m. Rows are the muons that reached the 2018 sensitive plane behind the absorber with
  E > 10 GeV (`bin/old_convention.py` reproduces this).
- cuda-muons (`get_design_from_params(..., add_target=False)`) starts them at z = -0.0045 m. **The hadron stopper
  is transported**: in TRY6 it is the first magnet (row 1 of `params['TRY6']`: z = 0 - 2.3 m, magnetised
  iron; `magnet_simulations.py`: "The first magnet is assumed to be the Hadron absorber"). Only the target (and
  the filler block `IronAfterTarget` that comes with it) is not simulated: no loss / scattering in the rest
  of the target after the production point.
- The enriched file uses exactly the same convention.

### How the enriched file was made
1. **Tail production** (`bin/tail_produce.py`, `--primary pythia6`, tag `tail2`), in the cudabear container
   (Geant4 11.2.2; `bin/g4.sh`), 150 slices x 800k events = **1.2e8 protons on target**:
   - first p-N collision at 400 GeV with **Pythia 6.4.28** (ShipSoft fork v6.4.28-ship1,
     `cpp/tailgen/pythia6_mesons.f`; target nucleon p / n with Z/A of the material at the vertex);
   - vertex by optical depth in the ECN3 target layout (Mo + W blocks, water slits), beam spot from the
     input file;
   - **Geant4 cascade** (FTFP_BERT, `cpp/tailgen/tailtransport.cc`) of every hadron above the threshold, with
     **forced decays**: every step of a pi+- / K+- emits one decay muon with weight BR x step / decay length;
   - muons carried to z = -0.0045 m, momentum at creation stored.
   ```
   bin/run_slices_any.sh tail2 150 150 bin/tail_produce.py --events 800000 --tag tail2 --primary pythia6
   ```
   Output `$R/data/source/tail/tail2_<K>.npz` (~20 min on 150 cores).
2. **Merge** (`bin/build_enriched.py --pcut 150 --tag tail2`):
   - every old row is kept, except the minimum-bias rows (weight 768.75) above 150 GeV (38,730 rows);
   - the tail muons above 150 GeV are added (95,070,569 rows), written in the old convention;
   - their weights get one flat factor (0.9534), so their flux above 150 GeV equals the old minimum-bias flux
     there: **the old normalisation is kept, only the statistics change**;
   - dimuon, charm and all muons below 150 GeV are unchanged.
   Output `$R/data/source/full_sample_after_target_enriched_p6.h5`: **601,405,839 rows**, total weight
   1.8012e10 (old file 1.8018e10), extra columns `source` (1 = tail row) and `orig_index` (row in the old file,
   -1 for tail rows).

Build the generators once: `bin/g4.sh bash cpp/tailgen/build.sh` and `bin/g4.sh bash cpp/tailgen/build_pythia6.sh`.

### Validation: regenerating the whole old file
With the same machinery, every class of the old file was regenerated
(`bin/plot_reproduction.py`, tags `mb2, tail2, dmlow2, dmhigh2, charmc`; plots in `results/reproduction_v2`,
not in git). Fluxes ours / old: minimum bias above 20 GeV 1.02, above 150 GeV 0.84; dimuon above 150 GeV 1.36;
all classes above 150 GeV 1.16. Agreement at the 10 - 20 % level; the difference was not chased further. The
dimuon (`--dimuon --vm-fairship`) and charm (`bin/charm_cascade.py`) productions exist but are **not** used
in the enriched file (the old dimuon / charm rows are kept). Earlier Pythia 8 version: `--primary pythia`
(superseded).

### The enriched cuda-muons production
- Script: `bin/generate_muon_study.py --input <enriched h5>` (the fast-veto-sim copy had the `--input` option first;
  physics, classification and storage unchanged), MuonsAndMatter container
  (`/disk/users/lprate/containers/MuonsAndMatterContainer.sif`), shield TRY6, launcher
  `bin/run_muon_study.sh` (fast-veto-sim used its own copy, one process per GPU taking every n-th seed). One seed =
  one pass of all 601 M muons, ~95-115 s on one GPU.
- Seeds 0 - 1999 (farm-gpu5 / farm-gpu4) and 2000 - 5999 (farm-gpu3, `--first-seed 0`, so they keep the rare classes
  only), output copied to `$R/data/sim/veto_muons_enriched_v1/seed_<N>.npz` (the originals stay in fast-veto-sim's
  `data/storage_study/enriched_v1`).
- **To make a design dataset** use the builder; `--old-types ''` is essential (the default takes the
  SST1 classes 0, 1, 2 from the old-input dataset "alpha", which undoes the enrichment):
  ```
  bin/make_design_dataset_gamma.py --production $R/data/sim/veto_muons_enriched_v1 --old-types '' --max-seed <N> --name <name>
  ```
  Dataset **delta** (`$R/data/store/design-optimization-dataset-delta-enriched-2026-10-19`) is
  this, from 484 seeds.
- `evt_n` in these files is the row in the **enriched** h5, not in the old 506 M file.

**Storage caveat.** The production keeps per class a planned number of rows (budget 15 M), computed in
advance from a fixed table `RATES` = per-seed counts of the old input. It never counts what it stored. With the
enriched input some classes are much more frequent, so rows stored = 15 M x (actual count / RATES):
straight-past 26 M instead of 15 M, and SST1 selected / SST1 other / veto-only (planned "keep all, ~50,000
seeds") grow ~11 - 37 k rows per seed without limit. **Weights are correct regardless** (each seed is
re-weighted to its own totals). For a new production on this input, set `RATES` from one enriched seed.

---

## 2. Interactions after the shield (muons and neutrinos)

Region z = 30.6 - 82 m: vessel helium, PVC walls (exact chords), cavern air, every UBT / fin plane of a design.
Forced interactions: every target gets many draws, each with a weight (interactions per spill), and the full
final state (pdg, momentum, production point). CPU only, numpy only at run time (no Geant4 / Pythia / GENIE).

### Muon library (`data/interactions/dis/mu_library.npz`, 1.9 GB)
- **Geant4 11.2.2 muonNuclear** (Kokoulin cross section + G4MuonVDNuclearModel), one forced interaction per
  event in each material (helium, air, PVC, Mylar, polystyrene, silicon, PVT): `cpp/g4dis.cc` (pybind module).
- **Pythia 6.4.28** `gamma/mu-` on p and n, W > 2 GeV: `cpp/tailgen/pythia6_dis.f`.
- Matched in energy transfer nu at the crossover nu_x(E) (7 GeV at E = 10, 15 at 100, 20 at 300 GeV), where
  Pythia's dsigma/dnu reaches Geant4's: Geant4 below, Pythia above (`split="crossover"`, default; also `w2`,
  `geant4`).
- 50 energies 0.5 - 500 GeV; short-lived products decayed by `lib/fastveto/decay.py`.
- Build: `bin/dis_library.py` (2 min on 300 cores). Closure vs full Geant4: -5 % for >= 1 extra track,
  -9 % for >= 2 (no photon conversions / secondary interactions in the fast chain).

### Neutrino library (`data/interactions/genie/nu_library.npz`, 0.9 GB)
- **Our own GENIE 3.06.02** build (not FairShip's, which is Pythia 8 only and aborted): reference
  configuration, Pythia 6, tune G18_02a_00_000, official splines (genie_xsec v3_06_00 G1802a00000-k250-e1000,
  trimmed to our nuclei: `data/interactions/genie/gxspl-fvs.xml`). Charm and tau decays switched on by the override
  `cpp/genie/config/CommonDecay.xml`.
- nu_mu, anti-nu_mu, nu_e, anti-nu_e on 9 nuclei, 1 - 500 GeV, 2.6 M events.
- Build GENIE (~10 min, once): `bash cpp/genie/build_genie.sh fetch` on the host (curl crashes inside the
  container), then `bin/g4.sh bash cpp/genie/build_genie.sh` -> `$R/external/genie`. Run anything with
  `bin/genie.sh <cmd>`. Splines: `bin/genie_splines.py`; library: `bin/nu_library.py` (4.5 min, 300 cores).
- Neutrino flux: the neutrinos in the MCTrack branch of the 2018 FairShip background files (withCharmandBeauty
  `*_mu.root`, copied from lxplus by `bin/fetch_from_lxplus.sh`) -> `bin/extract_neutrinos.py` ->
  `$R/data/interactions/neutrinos` -> `bin/nu_prepare.py` -> per-flavour column stores
  `$R/data/interactions/neutrinos/store_nu{14,-14,16,-16}` (p > 10 GeV; nu_e not used).

### Running it for a design
```
bin/lcg.sh python3 bin/muon_dis_apply.py --design D6_straw --workers 40   # 37 s -> data/interactions/dis/apply_D6_straw.npz
bin/lcg.sh python3 bin/nu_dis_apply.py   --design D6_straw --workers 40   # ~90 s -> data/interactions/genie/apply_D6_straw.npz
```
`--design` is a name in `designs/` or a path to any design JSON (e.g. one of Veto_System's). From Python:
`fastveto.dis.run(columns, design, lib, draws, rng, workers=40)`. The muon input is a Veto_System dataset store
(`--store`). Output format: `docs/HANDOVER.md`.

Rates for D6_straw, per 15 years: muon interactions helium 1.5e6, PVC 2.2e7, air 1.4e12; neutrino interactions
helium 5.3e3, PVC 3.7e3, air 5.7e3 (CC fraction ~0.75).

---

## 3. Multiple scattering tails (`lib/fastveto/msc.py`)
Gaussian core (veto.scattering) plus a Poisson number of hard single scatters above 2 theta0 (Rutherford, 1/theta^4,
nuclear-size cut-off), one parameter `TAIL_F = 2.0` tuned to Geant4 (`bin/msc_validate.py`): beyond 5 core
widths 0.41 - 0.54 % vs Geant4 0.44 - 0.49 %. Lives on in `lib/veto/scattering.py` (the fast-veto-sim `msc.py` was not ported).

---

## 4. Statistics of the muon dataset (where the errors come from)
Two independent errors on any rate from the cuda-muons production (`bin/storage_study_errors.py`):
- **transport**: randomness of the passes through the shield; shrinks with more seeds;
- **source sample**: the finite input file, replayed in every seed; does **not** shrink with seeds. Estimate
  it with a **bootstrap over input muons** (`evt_n`), effective number of sources N = R^2 / sum (w r)^2.

| class (shield exit) | old input, 10.7 k seeds: transport / source | enriched, 383 seeds: transport / source |
|---|---|---|
| SST1 selected | 0.24 % / **13.5 %** (N = 55) | 0.53 % / 0.57 % (N = 30 k) |
| SST1 other | 0.17 % / 5.1 % | 0.42 % / 0.55 % |
| veto only | 0.12 % / 0.9 % | 0.58 % / 0.76 % |
| SST1 selected, 50 - 100 GeV | 0.8 % / 2.5 % | 4.3 % / 2.5 % (enrichment starts at 150 GeV) |

The low-momentum and bounced classes are transport limited and need many seeds; high-momentum inputs need few.
Errors after veto and selection are larger (smaller subsets): bootstrap over `evt_n` there too
(`lib/veto/interactions.py` does).

---

## 5. Open issues
- **The SST1-selected rate is 1.55 x higher with the enriched input (1.9 x above 100 GeV)**, at the same flux
  above 150 GeV. Either the old ~20-muon estimate fluctuated low (likely for a heavy tail) or the Pythia 6 tail
  has a different angle / momentum shape above 150 GeV. Not resolved; it is bigger than any statistical error.
- The 10 - 20 % difference between our regeneration and the old file (section 1) and the ~x2 model dependence of
  the > 200 GeV tail (Pythia version / diffraction) are systematics of the source.
- Enrichment only above 150 GeV; 50 - 150 GeV still has the old statistics.
- Not simulated: interactions upstream of 30.6 m (shield, cavern walls, rock), nu_e flux, tau library, nuclear
  effects in the Pythia part of the muon library.

---

## 6. Output format of the interaction files (npz)

Interactions (`int_*`), one row each: `muon` (row in the input store), `evt_n`, `type`, `code` (0 helium, 1 air,
2 PVC, 10 + i plane i of the design), `x, y, z` (vertex, m), `E` (GeV), `ux, uy, uz` (parent direction), `weight`
(interactions per spill; x 1.2e7 for 15 years), `p_int`, `material`, `component` (muons: 0 Pythia, 1 Geant4;
neutrinos: 0 CC, 1 NC), `E_lib`; neutrinos also `flavour`, `nucleus`, `proc` (bits QE 1, RES 2, DIS 4, COH 8,
MEC 16), `charm`.
Particles (`part_*`): `int` (index into the interactions), `pdg`, `p` (px, py, pz, E in GeV, lab), `v` (production
point, m: the vertex, or a decay point), `beam` (the scattered muon / outgoing lepton).
Plus a json summary: interactions per spill / 15 years per target, timing.

