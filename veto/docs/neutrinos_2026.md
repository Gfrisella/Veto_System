# Neutrinos with the 2026 target: flux, interactions, backgrounds (2026-11-09)

Shah Rukh, branch `veto-neutrino-2026`. All rates at 4e13 p.o.t. per spill, 15 years = 6e20 p.o.t. (1.5e7 spills).

## 1. Flux (`bin/nu_prod.sbatch`, `bin/nu_flux.py`, `bin/nu_maps.py`)

tailtransport in neutrino mode (FairShip target, FTFP_BERT, hadrons followed above 1.1 GeV), every neutrino with
E >= 1 GeV within 0.3 rad recorded where it is made, upstream of the shield end. Three parts, each normalised to its own
protons (weight per spill = w_decay x 4e13 / (events x rep)):

| part | files | protons | note |
|---|---|---|---|
| min bias (`nuMB_nu_*`) | 500 | 500 x 2e4 | rows with hf > 0 dropped (charm / beauty counted by their parts) |
| charm (`charmc_nu_*`) | 1000 | 1000 x 1e4 | `charm_cascade --hadrons` (charm hadrons > 1 GeV, HF_EMIN = 1), `hf_decay --neutrino`, 50 % kept |
| beauty (`beautyc_nu_*`) | 400 | from the Nov-4 beauty hadrons | 5 % kept, negligible |

Data: `$VETO_SHARED/nu_prod/tail/` (= `/disk/gfs_lhcb/sqasim/nu_prod/tail`), read by `fastveto.neutrinos.load`.

Per proton (per spill): nu_mu 5.50e-3 (2.2e11; charm 4.6 %), anti-nu_mu 3.94e-3 (1.6e11), nu_e 5.55e-4 (2.2e10; charm
46 %), anti-nu_e 5.29e-4 (2.1e10; charm 52 %), nu_tau / anti-nu_tau 3.9e-5 each (1.6e9; D_s). Through the vessel
entrance 4.7e10 / spill, through the SST1 aperture 6.1e10. Soft charm set vs the Nov-4 > 10 GeV set above 10 GeV: ratio
1.0000. Interactive maps per part / flavour / energy: https://claude.ai/artifact/G5eMBfyL5D6ENdPhVxdGm3

### Validation

- **GSimple 2026 final-target flux** (Guglielmo, Study_on_Veto_System/Neutrino_flux_gsimple2026): nu_mu within 1 - 4 %.
- **SHiP 2025** (arXiv 2504.06692, Table 1: CC DIS in a 3 t tungsten SND, 15 years): our flux folded on a 40 - 50 cm W
  block on axis at z = 26 - 30 m (thin target, sigma_CC/E = 0.677 / 0.334 x 1e-38 cm^2/GeV):

  | | SHiP | 2026 flux | 2018 flux |
  |---|---|---|---|
  | nu_mu | 5.8e6 | 3.8 - 5.2e6 | 2.0 - 2.9e6 |
  | anti-nu_mu | 1.3e6 | 1.1 - 1.5e6 | 0.5 - 0.75e6 |
  | nu_e | 2.0e6 | 1.6 - 2.2e6 | - |
  | anti-nu_e | 4.0e5 | 4.3 - 5.7e5 | - |
  | nu_tau | 5.9e4 | ~1e5 (no tau threshold in the estimate) | 4e3 |

### Why the 2018 flux was low

The 2018 sample came from the muon-background files (`*_mu.root`): only neutrinos with p > 10 GeV whose sister muon was
stored, and the stored muon has p >= 15 GeV (5 % quantile 21 GeV). nu_mu above 10 GeV per proton, new / old by parent:
pi 0.94 (1.01 at 10 - 20 GeV: a pi decay with E_nu > 10 GeV always leaves a muon > 13 GeV), K+- 6.5 (9.8 at 10 - 20 GeV,
1.1 above 100 GeV), charm 3.7 - 5.5: the decays that leave a soft muon are the ones it missed. Plus nothing below 10 GeV
(65 % of the sigma-weighted new flux), no nu_e, and a narrower spread (92 % within 1 m of the axis at 30 m vs 23 %).

## 2. GENIE library v2 (`bin/genie_splines.py`, `bin/nu_library.py`)

GENIE 3.06.02, tune G18_02a_00_000; splines `gxspl-fvs.xml` for 15 nuclei (H1, He4, C12, N14, O16, Na23, Mg24, Al27,
Si28, Cl35, Cl37, Ar40, K39, Ca40, Fe56); six flavours, 36 energies 1 - 500 GeV, 2000 events each:
`data/interactions/genie/nu_library_v2.npz` (6.5e6 events). tau decayed by GENIE (its primary-lepton row dropped).
`fastveto.nudis.COMPOSITION` has `concrete` (G4_CONCRETE).

## 3. Interactions in the vessel and planes (`bin/nu_dis_apply.py`, `bin/run_interactions.py --kind nu`)

D6_straw_mod, per 15 years, one process per flavour (`--chunk` bounds the memory):

| flavour | interactions | selected | after veto (b) | selected MC events |
|---|---|---|---|---|
| nu_mu | 8.5e4 | 1.25 | 1.02 | 38 |
| anti-nu_mu | 3.2e4 | 0.57 | 0.51 | 51 |
| nu_e | 2.6e4 | 1.57 | 1.37 | 132 |
| anti-nu_e | 1.2e4 | 0.43 | 0.39 | 95 |
| nu_tau + anti | 2.8e3 | 0.05 | 0.04 | 84 |
| **all** | **1.58e5** | **3.9** | **3.3** | |

Interactions: helium 2.9e4, air 6.6e4, wall 5.8e4, planes 5.8e3. Survivors are all but ~0.07 in helium: the air / wall
interactions fail the pointing cut (0 - 1 selected draws per flavour: < 0.1 - 0.2 each). Like for like, the 2018 sample
through the same code gives 0.89 (the old `D6_straw_nu.json` 469 selected is from the pre-21-Oct selection).

Vertex activity of the selected helium events (`bin/nu_activity.py`): 77 % send a charged particle (p > 0.1 GeV) or a
photon (E > 0.1 GeV) out through the side walls; a lateral veto could leave ~0.8 (mostly anti-nu QE-like). ISSUES D2.

## 4. Cavern rock (`bin/nu_rock.py`, `bin/rock_select.py`, `bin/rock_combinatorial.py`)

`nu_rock.py prepare / task / combine`: neutrino interactions in the cavern concrete (TCC8 / ECN3 walls, floor, ceiling
top 3 m; the face top 3 m, and 3 - 60 m deep for CC muons with dE/ds = 0.46 + 9.2e-4 E), GENIE v2 on concrete, products
through `g4rock`. Per spill: 12.4 interactions, 2.8 charged out, 0.74 into the vessel, 0.045 charged > 1 GeV at SST1,
4.6e-4 with >= 2 (almost all from the floor).

Selection (`rock_select.py`, the same `interactions.evaluate`): 0 selected for both neutrino rock and muon rock (two-track
events reconstruct >= 6.8 m / >= 12.6 m from the target). Rock tracks in the combinatorial term (`rock_combinatorial.py`,
dataset c, 2 M-muon subsample): mu x rock 8.2e3 candidates / 15 y -> 0.023 after the veto (rock leg vetoed 99.99 %),
rock x rock 0.

## Files

| what | where |
|---|---|
| production | `$VETO_SHARED/nu_prod/tail/`, `bin/nu_prod.sbatch` |
| flux stores (6 flavours) | `data/interactions/neutrinos2026/store_nu*` (`bin/nu_prepare.py`) |
| library v2 | `data/interactions/genie/nu_library_v2.npz`, `gxspl-fvs.xml`, `xsec_tot.npz` |
| vessel interactions | `data/interactions/genie/apply2026_<design>_nu<pdg>.{npz,json}` |
| selection results | `results/interactions/<design>_nu2026_nu<pdg>.json`, `rock_select_*`, `rock_comb_*` |
| rock | `$VETO_SHARED/nu_rock/` (`nu_rock_<design>.json`, `_draws.npz`) |
