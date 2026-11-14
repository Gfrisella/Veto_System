# V0 production in the MS7 muon DIS chain, checked against data

Branch `veto-v0-production` (base 4904c4f = PR #15), 2026-11-10. ISSUES D16. Script `bin/v0_production.py` (host python,
~25 s, existing files only). Figures and numbers (tracked): `docs/figures/v0_production/` (`v0_vs_W.pdf`,
`v0_ms7_fate.pdf`, `v0_production.json`). All rates per spill at 4e13 p.o.t.; 15 y = 1.5e7 spills.

Inputs: `data/interactions/dis/mu_library_iron_v0stable.npz` (Geant4 11.2.2 FTFP_BERT muonNuclear on G4_Fe, Pythia 6.4.28
on p / n; K_S, Lambda kept undecayed; same model as the chain library) and the 10 PR #15 MS7 runs
`shield_muon_g4v112_s101-110.npz` (2e6 interactions each, Geant4 default, 7.94e5 interactions / spill in the last 50 cm).

## Key messages (slide / report ready)
1. The "2.3e10 V0 decays in the vessel / 15 y" of PR #15 are K_L (1540 / spill of 1547).
2. Every V0 made at the muon vertex: K_L 7.9e3, K_S 7.8e3, Lambda 5.4e3, anti-Lambda 260 per spill (1.2e11 K_L / 15 y);
   only ~1 % per interaction because muonNuclear sits at low W (median 1.3 GeV, below the K Lambda threshold).
3. Data at the same W (6.2 GeV, gamma p at 20 GeV): Geant4 within ~6 %, Pythia 6 x1.9 high -> Geant4 as default (D3)
   is supported by data; the Pythia spread is a one-sided overestimate.
4. K_L chain: 7.9e3 made -> 6.5e3 with T > 0.5 GeV -> 2.0-2.5e3 survive the iron -> 3.1e3 at the exit (20-35 % made by the
   shower in the iron) -> 1.5e3 decay in the vessel (P = 0.49).
5. Measured with a V0 tally in Geant4: 1.5e4 K_L made per spill in total (half by the shower), 37 % of the exit K_L are
   shower products.
6. Keeping K_S / Lambda at the vertex (D9) raises their vessel decays x8-30, but none passes the selection (< 1e5 / 15 y
   each before the IP cut vs 1.7e7 for K_L): the MS7 background stays K_L.
7. Low W (< 3.5 GeV): Geant4 makes >= x2-3 too little strangeness, but only ~8 % of the relevant K_L come from there
   (+8-15 % on the K_L).

## Slide 1: V0 multiplicity vs W, compared with data
Figure: `v0_vs_W.pdf` (left: <n> vs W, Geant4 solid, Pythia 6 dashed, stars = SLAC gamma p 20 GeV; right: W of the
library events).

| W = 6.2 +- 0.5 GeV, per event | K0_S | Lambda (+Sigma0) | anti-Lambda |
|---|---|---|---|
| data: gamma p, E_gamma = 20 GeV (SLAC SHF) | 0.082 +- 0.003 | 0.049 +- 0.002 | 0.0029 +- 0.0003 |
| Geant4 11.2 muonNuclear (Fe), 8381 ev | 0.087 | 0.051 | 0.0020 |
| Pythia 6 (p, n), 1.2e5 ev | 0.153 | 0.090 | 0.0058 |

Note: data on hydrogen, Geant4 on Fe. Reference, not W matched: NOMAD nu_mu CC on C, <E_nu> = 45 GeV: K0_S 0.0676,
Lambda 0.0504, anti-Lambda 0.0037 per event (primary V0). Geant4 K0_S rises ~log W as WA21 / E665 / H1 show.

## Slide 2: every V0 made in the last 50 cm of MS7
Library <n>(E) folded with the vertex-energy spectrum of the PR runs (median 72 GeV; 5-95 % 4-152 GeV).

| per spill | K_L | K_S | Lambda | anti-Lambda | K+ | K- | pi+- |
|---|---|---|---|---|---|---|---|
| Geant4 (chain default) | 7.9e3 | 7.8e3 | 5.4e3 | 2.6e2 | 9.4e3 | 5.8e3 | 7.7e5 |
| Geant4 per interaction | 0.0099 | 0.0098 | 0.0068 | 3.3e-4 | 0.012 | 0.0074 | 0.97 |
| Pythia 6 per interaction | 0.122 | 0.123 | 0.084 | 0.0061 | 0.15 | 0.10 | 3.3 |

K_L = K_S (K0 / K0bar -> 1/2 K_S + 1/2 K_L), and the default and V0-stable libraries give the same K_L (0.0077 per event,
flat in log E). V0 are soft: median p 1.4 GeV (E 8-60 GeV) to 2.5 GeV (E 70-250 GeV); 82 % of K_L above T = 0.5 GeV.

## Slide 3: what reaches the shield exit (z = 29.90 m)
Figure: `v0_ms7_fate.pdf` (left: V0 per interaction vs E with the MS7 spectrum; right: V0 at the exit per interaction vs
vertex depth, with the primary-only expectation).

| K_L per spill | |
|---|---|
| made at the vertex | 7.9e3 |
| with T > 0.5 GeV (exit-plane cut) | 6.5e3 |
| primaries surviving the iron (lambda 17-22 cm, vertex uniform 0-50 cm) | 2.0-2.5e3 |
| at the exit plane, all (Geant4) | 3.1e3 (2.9e3 with p > 1 GeV, the PR summary) |
| -> made by the shower in the iron | 21-37 % (estimate); **37 % measured** with the V0 tally (slide 4) |
| decaying in the vessel (31.13-81.13 m) | 1540 (P = 0.49; PDG c tau 15.34 m at p ~ 2 GeV) |

- K_L vs depth falls with lambda_eff = 29 cm (fit), flatter than the 17-22 cm interaction length: shower K_L.
- K_S / Lambda at the exit rise from depth 0 to ~12 cm and fall after: build-up of shower products.

## Slide 4: every V0 made in the iron and its fate (V0 tally, measured)
`cpp/g4shield` `v0_tally()` + `shield_dis.py generate --v0-tally` (2026-11-10): every K_L / K_S / Lambda / Xi0 created in the
last 50 cm, injected at the muon vertex or made by the shower (creator process), and its fate. Run: 2e6 interactions,
Geant4 11.4.2 (FairShip pixi 26.10 via `bin/g4pixi.sh`; the 11.2 container cannot start in the sandbox), PR #15 parents,
V0-stable library. Analysis `bin/v0_fate.py`; figures `v0_fate_bars.pdf`, `v0_fate_origin.pdf`; numbers `v0_fate.json`.

| per spill | made | T < 0.5 GeV (not tracked) | decayed in the iron | absorbed | exit (z = 29.90 m) |
|---|---|---|---|---|---|
| K_L vertex | 7.8e3 | 1.6e3 | 8 | 4.3e3 | 2.0e3 |
| K_L shower | 7.6e3 | 4.3e3 | 3 | 2.1e3 | 1.2e3 |
| K_S vertex | 7.7e3 | 1.7e3 | 2.8e3 | 2.4e3 | 0.9e3 |
| K_S shower | 7.7e3 | 4.2e3 | 1.8e3 | 1.1e3 | 0.5e3 |
| Lambda vertex | 5.4e3 | 2.6e3 | 1.3e3 | 1.1e3 | 0.4e3 |
| Lambda shower | 6.0e3 | 4.7e3 | 0.7e3 | 0.4e3 | 0.2e3 |
| anti-Lambda (all) | 350 | 2 | 107 | 151 | 77 |

- In total 1.5e4 K_L, 1.5e4 K_S, 1.1e4 Lambda per spill are made (2.3e11 K_L / 15 y): the shower doubles the vertex
  number, but its V0 are soft (median p ~0.8 GeV vs ~2 GeV) and mostly below the 0.5 GeV cut.
- K_L at the exit: 3.2e3 / spill, 63 % from the vertex, **37 % from the shower** (estimate on slide 3: 21-37 %).
  Vertex K_L above the cut: 32 % survive the iron (lambda ~ 19 cm, as assumed).
- Shower V0 are made by pi+- inelastic (~45 %), K+ (20 %), K_L (19 %: K_L -> K_L / K_S regeneration-like), K_S (11 %),
  K- (6 %); Lambda also from K- (17 %) and Sigma decays (7 %). They build up over the first ~15 cm (slide 3).
- Note: tracks below 0.5 GeV are killed at creation, so V0 that such soft hadrons would make are not counted (small:
  near threshold).

## Slide 5: K_S / Lambda kept at the vertex (D9) and the Geant4 version
| per spill | PR #15 (G4 11.2, default library) | G4 11.4, default library | G4 11.4, V0-stable library |
|---|---|---|---|
| K_L at exit (p > 1 GeV) | 2923 | 2905 | 2983 |
| K_S at exit (p > 1 GeV) | 473 | 429 | 1398 (x3.3) |
| Lambda at exit (p > 1 GeV) | 262 | 200 | 742 (x3.7) |
| decays in the vessel: K_L / K_S / Lambda / anti-Lambda | 1543 / 6.7 / 1.8 / 1.0 | 1502 / 6.2 / 0.7 / 0.9 | 1545 / 51 / 22 / 11 |
| V0 decays in the vessel per 15 y | 2.33e10 | 2.27e10 | 2.44e10 |
| passing all but the IP cut, per 15 y (D6_straw_mod) | 1.4e7 | 1.5e7 (K_L only) | 1.7e7 (K_L only) |

- Geant4 11.4 reproduces 11.2 within 1-3 % for K_L: the version does not matter here.
- Keeping K_S / Lambda at the vertex (D9) multiplies their exit rate by 3-4 and their vessel decays by 8-30, but they decay
  right behind the entrance window (median z 31.25 m vs 31.13): of 18 250 K_S and 8 385 Lambda forced-decay copies none
  passes the selection even without the IP cut -> < 1e5 per 15 y each (90 % CL), vs 1.7e7 for K_L. The MS7 background
  stays K_L; D9 matters only for the DIS x mu combinatorial (single tracks), not for the V0 vertices.
- Bin statistics: K_L all-but-IP rests on 65-71 forced copies; IP < 1 m on 10-14 copies.

## Slide 6: low W (1.6 - 3.5 GeV), where most Geant4 events are (ISSUES D16 c)
Script `bin/v0_lowW.py`; data `data/reference/lowW_strangeness_photoproduction.json` (SAPHIR Glander 2004 / Lawall 2005,
CLAS McCracken 2010 / Dey 2010 summed over measured angles, ABBHHM Erbe 1969, SLAC Ballam 1969 / Bingham 1973, sigma_tot
Armstrong 1972 / Caldwell 1973). Figure `v0_lowW.pdf`.

| W [GeV] | Geant4: events with strangeness | data (gamma p) | Geant4: Lambda / event | data: (K+Lambda + K+Sigma0) / sigma_tot |
|---|---|---|---|---|
| 1.75 | 0.010 | >= 0.021 (KY sum), ABBHHM >= 0.013-0.029 | 0.0075 | 0.018 |
| 1.85 | 0.014 | >= 0.030 | 0.0096 | 0.026 |
| 2.05 | 0.011 | >= 0.021 | 0.0066 | 0.018 |
| 2.25 | 0.012 | >= 0.013; ABBHHM >= 0.021 | 0.0055 | 0.012 |
| 2.5 - 3.1 | 0.016 - 0.024 | SLAC visible strange decays >= 0.058 - 0.061 | 0.006 - 0.007 | (CLAS, partial angles) 0.006 - 0.002 |

- Geant4 muonNuclear is low by >= x2 near threshold (W < 2.2 GeV: below the exclusive K Y channels alone) and by
  >= x2.5-3 at W 2.5-3 GeV (below the SLAC visible-decay lower limit). Data are lower limits: the true deficit is larger.
- Pythia 6 (W > 2 GeV only) is above the data, x3-5 over the exclusive Lambda channels.
- Impact on the background is small: of the vertex K_L with T > 0.5 GeV (those that can reach the exit), 78 % come from
  W > 5 GeV (where Geant4 agrees with data, slide 1), 14 % from 3.5-5 GeV, 7.6 % from W < 3.5 GeV. A x2-3 deficit
  there -> +8-15 % on the relevant K_L; W 3.5-5 GeV not checked with data.
- Caveats: Fe vs p (Fermi motion smears the thresholds), quasi-real photons (median Q2 0.03 GeV2) vs real.

## PDG check of the decay tables (lib/fastveto/decay.py, bin/shield_dis.py)
c tau K_L 15.34 m, K_S 2.6844 cm, Lambda 7.89 cm, Xi0 8.71 cm; K_S -> pi+pi- 0.692 / pi0pi0 0.308; Lambda -> p pi- 0.639 /
n pi0 0.361 (PDG 0.358); K_L semileptonic / 3pi as PDG (sum 0.9966). Missing: K_L -> pi+ pi- (2.0e-3; ~5e7 / 15 y in the
vessel, a 2-track vertex pointing to the shield) and K_L -> gamma gamma.

## Open (ISSUES D16)
Done 10 Nov: V0 tally (b), V0-stable chain (a, D9 for V0 vertices), low-W data (c), K_L -> pi+ pi- in KL_CHANNELS (d).
Open: W = 3.5-5 GeV not compared with data (14 % of the relevant K_L); K_L -> gamma gamma (not needed: no tracks);
eta' undecayed in the library (minor); D9 DIS x mu combinatorial with the V0-stable single tracks; the K_L IP tail
statistics (65-71 copies all-but-IP): K_L-enriched generation (STATUS 10 Nov).

## References
- K. Abe et al., Inclusive photoproduction of neutral strange particles at 20 GeV, PRD 30 (1984) 1, SLAC-PUB-3162, Table II.
- P. Astier et al. (NOMAD), A study of strange particle production in nu_mu CC interactions, NPB 621 (2002) 3, hep-ex/0111057.
- S. Aid et al. (H1), Strangeness production in DIS at HERA, hep-ex/9607010 (W dependence incl. E665, WA21).
