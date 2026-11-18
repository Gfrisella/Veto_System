#!/usr/bin/env python3
"""Vessel interaction caches for the objective with what the current rules need (17 Nov 2026): vertex and visible momentum
(V0 rule), the own SBT deposits of the interaction products and of the parent muon (selftag_hits / zone rule, D19) and the
parent's stopping point (shell_energy_loss). Same path as bin/sbt_prepare.py, on the vessel event stores (helium, air,
walls, upstream air), instead of the 16 Nov prepared_muon / prepared_nu caches (bin/prepare_interactions.py), which have none.

    python3 bin/vessel_prepare.py --kind muon --store <results>/evstore/vessel_mu_R2_sbt_ubt <...>_kl10 --tag _nov4full_d19
    python3 bin/vessel_prepare.py --kind nu --store <results>/evstore/vessel_nu2026_R2_sbt_ubt --tag _nov4full_d19
"""
import _bootstrap  # noqa: F401
import argparse
import time

import numpy as np

import sbt_prepare as sp
from veto import io, paths, reco
from veto import vertex_selection as vsel

VESSEL_CODES = (0, 1, 2, 8)        # helium, air, wall, air upstream (interactions.CODE_NAMES)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("muon", "nu"), required=True)
    ap.add_argument("--store", nargs="+", required=True)
    ap.add_argument("--ip-cut", choices=("full", "partial"), default="full")
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--tag", default="_nov4full_d19")
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    t0 = time.time()
    sp.CODES = VESSEL_CODES
    sp.KEEP_NATURAL_KL = a.kind == "nu"     # the neutrino store has no forced-decay companion: keep its natural K_L decays
    cfg = reco.RecoConfig(multiple_scattering=True, ip_max=vsel.IP_CUTS[a.ip_cut])
    muons = io.open_store(paths.store_dir(a.dataset))[0] if a.kind == "muon" else None
    rng = np.random.default_rng(a.seed)
    parts = []
    for s in a.store:
        pr = sp.prepare_store(str(paths.evstore(s)), cfg, muons, rng)
        parts.append(pr)
        print(f"{s}: {len(pr['event_weight15']):,} selected, {pr['event_weight15'].sum():.4g} / 15 y before the veto, "
              f"{int(len(pr['sbt_dep_event']))} own deposits, {time.time() - t0:.0f} s", flush=True)
    out = sp.merge(parts)
    out.pop("sbt_material", None)
    f = paths.results_dir() / "interactions" / f"prepared_{a.kind}{a.tag}{'_partial' if a.ip_cut == 'partial' else ''}.npz"
    np.savez(f, **out)
    print(f"-> {f}: {len(out['event_weight15']):,} events, {out['event_weight15'].sum():.4g} / 15 y ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
