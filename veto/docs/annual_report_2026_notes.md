# SHiP Annual Report 2026 — what matters for the veto study (read 2026-11-09)

Source: /disk/users/gfrise/SHiP_Annual_Report_2026.pdf (75 pages, chapter 5 "Backgrounds", report line numbers below).

| Topic | Report | Our chain | Action |
|---|---|---|---|
| Normalisation (l. 1326) | 4e13 p.o.t. per 1 s spill, 6e20 for 15 years | adopted 2026-11-09 (ISSUES D13) | confirms D13 |
| Muons at SST1 (l. 1480) | ~16 kHz in the TD acceptance, < 50 % with p > 1 GeV (TRY2026) | quote at 4e13 from now on | compare our SST1 rate at 4e13 |
| Combinatorial (l. 1484-1489) | DOCA < 2 cm, vertex in DV, IP < 2.5 m: ~7500 pairs / spill = 9e10 / 15 y; +-200 ps TD: 36 pairs / 15 y; UBT inefficiency needed ~ few e-2 | same loose selection | cross-check our pair rate (per 4e13) |
| Muon categories at SST1 (Table 8) | straight 66.1 % (100 % p > 1), wall 25.8 % (10 %), floor 7.7 % (2.5 %) | our types 0-2 | compare shares |
| DUBT size (l. 1494) | lateral aperture ~ exit face of the last magnet tags all straight muons | geometry.SHIELD_END_HALF_X/Y = 1.71 / 1.27 | consistent with the fin / UBT designs |
| Muon DIS in the shield (l. 1402) | last interaction length of the shield, ~20 cm | last 50 cm of MS7 (shield_dis) | ours is the larger target; 20 cm would lower our MS7 K_L |
| Muon DIS generators (l. 1424-1477, Table 7) | Pythia6 vs Pythia8 vs GiBUU vs Geant4; photoproduction (low Q2) dominates sigma; Pythia8 PP needs W_min set 10 -> 5 GeV and Q2 < 1.5, DIS Q2 > 1.5; Pythia6 PP W > 2.5, DIS W > 2; for W > 5 GeV all agree on sigma 0.5-4 ub (20-200 GeV), Geant4 always above and diverging | D3 study: Pythia 8.230 photoproduction hung at 20 GeV (default W_min = 10) | retry p8 photoproduction with W_min = 5, Q2 < 1.5 (Table 7 settings) |
| Neutrino background (l. 1341-1400) | He + Mylar: below 1 event in 15 y (full DSBT); last interaction length of the shield and DUBT material "in preparation" | shield nu DIS chain exists (D4) | our MS7 nu numbers are new input for them |
