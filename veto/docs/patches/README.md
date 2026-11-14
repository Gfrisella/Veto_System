# Patches for other repositories

## C9_categorize_muons.patch (ISSUES C9, 2026-11-11) — MuonsAndMatter `muons_and_matter/cuda_muons_ship.py`

Made against Guglielmo's working copy of 2026-11-11 (which has uncommitted changes to this file; the patch is NOT
against the last commit). His repository: he reviews and applies it:

    cd /disk/users/gfrise/MuonsAndMatter && git apply /path/to/C9_categorize_muons.patch

What it changes: `categorize_muons(..., momenta_at_UBT=None, bounce_cut=0.5)`. With the real state at z = 30.6 m
(position AND momentum, passed by `run_from_params` from the 30.6 m sensitive plane), a muon is bounced / scattered when
its z = 82 m state is more than 0.5 m off the straight line of that state (the veto definition, Veto_System
`muon_types.classify`). Before, "bounced" came from the last plane extrapolated back to 30.6 m, which flagged straight
muons. Without `momenta_at_UBT` the old behaviour is kept.

Check on 113 762 Nov-4 muons reaching both planes (states from the design store):

| category | old | new | veto types inside (new) |
|---|---|---|---|
| bounced_back | 7 804 (all straight: sst1_selected / sst1_other) | 10 | 9 sst1_bounced, 1 veto_only |
| straight_past_vessel | 112 867 (incl. 4 621 sst1_selected) | 99 593 | straight_past 97 937, veto_only 1 597, sst1_other 59 |
| bounced_no_come_back | 19 101 (17 995 straight_past) | 2 738 | rock_bounced 2 627, sst1_bounced 73, veto_only 38 |

The 73 sst1_bounced in bounced_no_come_back are a definition difference, not a bug: MuonsAndMatter's SST1 acceptance
requires p > 1 GeV and 89 % of the real bounced SST1 muons have p <= 1 GeV at 82 m (median 0.34 GeV).
