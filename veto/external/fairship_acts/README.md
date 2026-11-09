# FairShip ACTS straw tracker with PR #1545 (ISSUES D14, ACTS option A)

Source: ShipSoft/FairShip PR #1545 "fix(reco): read the ACTS straw material from the geometry" (open), head commit
786b361173d46366eefcd4d00be3041bf93137ae, branch fix/acts-straw-material-from-tgeo, files ActsTracking/*.
LGPL-3.0-or-later (CERN for the SHiP Collaboration). Copied unchanged into src/ on 2026-11-10.

Base: the FairShip pixi environment 26.10 on CVMFS (/cvmfs/ship.cern.ch/pixi-envs/fairship/26.10: tag 26.10,
ROOT 6.40.02, Geant4 11.4.2, ACTS 48.2), which already contains ActsTracking/. Of its sources only
ActsStrawGeometry.cpp differs from the PR head (the straw material read from the TGeo straw volumes: wall, gas, wire,
mean path = cross-section / diameter, 5.3e-4 X0 per straw instead of a fixed 5.7e-4).

- build.sh: rebuilds lib/libShipActsTracking.so with the environment's compiler (conda gcc 15.3).
- bin/fairship_env.sh: runs a command in that environment with lib/ in front of the installed library.

The tracker builds every straw as a free ACTS StrawSurface keyed by the strawtubes detector ID (no layers), fits
drift-radius measurements with a Kalman filter on a DirectNavigator, field from any TVirtualMagField (ship frame,
kGauss); pattern recognition Truth or Segments (StrawPatRec).

Validation 2026-11-10 (data/acts/pixi2610_pg): 500 mu, 10 GeV, gun at z = 50 m, run_simScript (TRY_2026) +
ShipReco.py --trackFitter acts --patRec Truth: 499 tracks, 32 measurements, p median 9.992 GeV, core sigma(p)/p
0.94 %, rms 4.2 % (1 track beyond 5 %), chi2/ndf median 0.92.
