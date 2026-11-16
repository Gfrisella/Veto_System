# Straw-spectrometer resolution from the FairShip ACTS fit (ISSUES D14, part 2)

2026-11-16. Purpose: replace the analytic constants of `reco.RecoConfig` (sigma_pos 1 mm, sigma_slope 0.1 mrad,
ms_slope 2 mrad GeV, sigma_t_track 0.1 ns) with numbers from a real track fit. **This change is not wired in yet**: `reco.measure`
is unchanged, and `lib/veto/acts_resolution.py` only offers the parametrisation.

## Method

- Chain: FairShip pixi 26.10 (CVMFS; Geant4 11.4, ACTS 48.2) with our PR #1545 `libShipActsTracking`
  (`external/fairship_acts`, `bin/fairship_env.sh`).
- `bin/batch/acts_scan.sbatch` runs one task per momentum (slurm 2095542, 10 tasks, 1 CPU / 8 GB, `%5`, 2-6 min each):
  - **Generation:** `run_simScript.py PG`, mu+ and mu- (`--bothCharges`), p = 1, 2, 3, 5, 10, 20, 50, 100, 200, 400 GeV,
    4000 muons per point.
  - **Starting point:** vertex uniform in |x| < 1.5 m, |y| < 2.5 m at FairShip z = 83.35 m (our 81.21 m, just behind the
    vessel exit window), polar angle 0-1.5 deg.
  - **Geometry and field:** the full FairShip geometry, with FairShip's spectrometer field map
    `2026_07_02_MainSpectrometerField_V21`.
  - **Fit:** `ShipReco.py --trackFitter acts --patRec Truth` (Kalman filter). The fitted state is at the first
    measurement, FairShip z = 84.03 m (our **81.89 m**, the first SST1 layer).
- `bin/acts_resolution_scan.py extract` matches each fitted track to its MC muon (`fitTrack2MC`).
  - **Truth:** the MC straw point nearest the fit z, carried along a straight line to that z.
  - **Momentum truth:** the momentum at that point, so energy loss upstream is not counted as a resolution effect.
- `bin/acts_resolution_scan.py fit` builds the table and the fits and writes `data/acts/resolution_param.json`.
  - **Core sigma:** half the central 68.3 % interval.
  - **rms:** the full rms.
  - **tail:** the fraction of tracks beyond 5 core sigma.
  - **Efficiencies:** fitted / generated, for muons that hit all 4 stations and for muons that hit SST1.
    "Fitted" means a fitted track is written; ShipReco requires hits in at least 3 stations, and with the 1.5 deg cone
    every muon had about 32 measurements.

## Results (FairShip V21 field)

Projections: x is the non-bending projection, poorly measured. The straws run along x, so x comes only from the
±5° stereo views. y is the bending projection, measured directly by the straws.

| p [GeV] | eff (4 stations) | eff (SST1) | sigma(p)/p core | rms | tail >5 sigma | sigma x [um] | sigma y [um] | sigma tx [urad] | sigma ty [urad] | chi2/ndf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.962 | 0.901 | 1.02 % | 1.61 % | 0.6 % | 1207 | 85 | 1267 | 565 | 1.42 |
| 2 | 0.996 | 0.993 | 0.90 % | 1.15 % | 0.5 % | 808 | 66 | 617 | 314 | 1.05 |
| 3 | 0.998 | 0.998 | 0.92 % | 1.05 % | 0.3 % | 706 | 59 | 439 | 230 | 1.00 |
| 5 | 0.999 | 0.999 | 0.92 % | 1.07 % | 0.3 % | 650 | 55 | 279 | 150 | 0.95 |
| 10 | 0.999 | 0.999 | 0.93 % | 1.12 % | 0.3 % | 622 | 51 | 160 | 80 | 0.90 |
| 20 | 1.000 | 0.999 | 1.03 % | 1.19 % | 0.3 % | 616 | 49 | 112 | 51 | 0.87 |
| 50 | 0.999 | 0.999 | 1.34 % | 1.41 % | 0.2 % | 610 | 44 | 86 | 31 | 0.83 |
| 100 | 1.000 | 1.000 | 2.07 % | 2.16 % | 0.1 % | 598 | 44 | 85 | 25 | 0.82 |
| 200 | 1.000 | 1.000 | 3.91 % | 4.05 % | 0.0 % | 602 | 44 | 86 | 23 | 0.82 |
| 400 | 1.000 | 1.000 | 7.42 % | 7.44 % | 0.0 % | 613 | 43 | 83 | 22 | 0.82 |

Other checks:
- **Charge:** no wrong charge in 40 000 tracks.
- **Angle:** splitting each sample at its median slope (0-1.5 deg) changes sigma(p)/p and sigma(tx) by ≤ 5 %, so there
  is no angle dependence inside the SST1 acceptance.
- **1 GeV:** 10 % of the muons that enter SST1 miss station 3 or 4 (bent out of acceptance or stopped). The rest are fitted
  at 96 %.

## Fits (p in GeV; max deviation from the table points)

| quantity | form | parameters | max dev |
|---|---|---|---:|
| sigma(p)/p | sqrt(a² + (b p)²) | a = 0.932 %, b = 1.87e-4 / GeV | 8 % |
| sigma(tx) | sqrt(c² + (d/p)²) | c = 85 urad, d = 1.28 mrad GeV | 5 % |
| sigma(ty) | sqrt(c² + (d/p)²) | c = 24 urad, d = 0.66 mrad GeV | 20 % (at 1 GeV) |
| sigma(x) | sqrt(c² + (d/p)²) | c = 609 um, d = 1.05 mm GeV | 2 % |
| sigma(y) | sqrt(c² + (d/p)²) | c = 46 um, d = 0.078 mm GeV | 11 % |

## Comparison with the current reco constants

| reco.RecoConfig | current | ACTS (this study) |
|---|---|---|
| sigma_pos | 1 mm, both projections | x 0.61 mm, y 0.046 mm (p ≳ 5 GeV); below about 3 GeV the 1/p term adds |
| sigma_slope (high p) | 0.1 mrad | tx 0.085 mrad, ty 0.024 mrad |
| ms_slope | 2 mrad GeV | tx 1.28, ty 0.66 mrad GeV |
| sigma(p)/p | not smeared | 0.93 % ⊕ 1.9e-4 p (V21 field); × 0.78 in our field → 0.73 % ⊕ 1.5e-4 p |
| sigma_t_track | 0.1 ns | not provided: RecoTrack has no time (it comes from the timing detector, not from the straws) |

The current constants are conservative in x and in the slopes. They overestimate y by a factor of about 20 and the
multiple-scattering slope term by about 1.5-3.

## Caveats

- **Field:** FairShip V21 has ∫Bx dz = 0.84 T·m on axis (0.86 at x = 1 m, 0.71 at y = 2 m). Our
  `MainSpectrometerField.h5` has 1.07 T·m (1.08, 1.03). sigma(p)/p scales as 1/∫B dl in both terms, so
  `acts_resolution.sigmas` applies `field_scale` = 0.84/1.07 by default. Positions and slopes do not depend on the field.
  A run with our map loaded into FairShip was not done.
- **Material:** the vessel exit window and everything upstream of FS z = 83.35 m are not in the sample. Their multiple
  scattering stays with our `scattering.py`, so the slope terms above are the spectrometer alone.
- **Pattern recognition:** Truth, so the efficiency is fit efficiency only, an upper bound. The straw hit efficiency and
  the drift smearing are FairShip's digitisation defaults.
- **Fit position:** the state is at the first measurement, our z = 81.89 m, not at 82.08 m.
- **Fit form:** sigma(ty) at 1 GeV is underestimated by 20 % by the simple form. Energy loss and the non-Gaussian
  multiple scattering are visible at 1 GeV (chi2/ndf 1.4, rms/core 1.6).

## Files

- `bin/acts_resolution_scan.py`: extract and fit.
- `bin/batch/acts_scan.sbatch`: the scan. It unsets DISPLAY and sets QT_QPA_PLATFORM=offscreen. Job 2090827 failed on
  every task because, with DISPLAY forwarded from the submitting shell, Geant4 started its Qt UI on the node and aborted
  in `FairRunSim::Init`.
- `data/acts/resolution_param.json`: the raw table and the fits.
- `lib/veto/acts_resolution.py`: `sigmas(p)`.
- Scan outputs: `veto/data/acts/pg_scan_2611/p*/` (sim, reco, point.npz).
