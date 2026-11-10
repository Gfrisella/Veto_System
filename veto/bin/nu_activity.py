#!/usr/bin/env python3
"""Extra activity of the SELECTED neutrino interactions (bin/nu_dis_apply.py output through the selection of
interactions.evaluate): what else comes out of the vertex besides the two vertex tracks, and where it goes. A veto that
surrounds the vessel (SBT-like, on the walls) could tag the events by the particles leaving through the walls; the
upstream-track veto of reco cannot (no incoming charged particle).

Per selected event, straight lines from the vertex: charged (p > PMIN) and photons (E > EMIN; pi0 -> counted as photons)
leaving through the side walls (outside the vessel cross-section at the exit window, or going backwards), and extra
charged reaching SST1 outside the vertex pair. Fractions (weighted) of events with >= 1 such particle, per 15 years.

    bin/remote.sh run bin/nu_activity.py --design designs/modules/D6_straw_mod.json --pot-spill 4e13 <apply npz> ...
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from run_interactions import load_npz
from veto import design as dsg
from veto import geometry as G
from veto import interactions as ix
from veto import reco
from veto import vertex_selection as vsel

PMIN, EMIN = 0.1, 0.1     # GeV


def wall_exit(v, p):
    """Straight line from v leaves the vessel through the side (or the entrance) before the exit window."""
    back = p[:, 2] <= 0
    t = (G.Z_VESSEL_MAX - v[:, 2]) / np.where(back, 1.0, p[:, 2])
    x, y = v[:, 0] + p[:, 0] * t, v[:, 1] + p[:, 1] * t
    return back | (np.abs(x) > G.VESSEL_X_OUT) | (np.abs(y) > G.VESSEL_Y_OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+"); ap.add_argument("--design", required=True)
    a = ap.parse_args()
    p_ = Path(a.design)
    design = dsg.load(p_ if p_.exists() else a.design)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS["full"])
    tot = {}
    for f in a.files:
        inter, part = load_npz(f)
        w15 = inter["weight"] * ix.SPILLS_15Y
        n = len(w15)
        tr = ix.charged_tracks(part)
        sel, pair, vertex = ix.select(tr, n, cfg=cfg)
        n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n)[sel]
        keep = n_acc <= cfg.max_tracks
        sel, pair = sel[keep], pair[keep]
        sel = sel[inter["code"][sel] == 0]                          # helium only
        if not len(sel):
            print(f"{Path(f).name}: no selected helium events"); continue
        pi = part["int"]
        m = np.isin(pi, sel)
        pdg, pp, vv, ev = part["pdg"][m], part["p"][m, :3], part["v"][m], pi[m]
        pabs = np.linalg.norm(pp, axis=1)
        q = ix.charge(pdg)
        out = wall_exit(vv, pp)
        vertex_rows = tr["row"][pair.ravel()]                       # the two vertex tracks (part rows)
        is_vtx = np.isin(np.nonzero(m)[0], vertex_rows)
        ch_wall = (q != 0) & (pabs > PMIN) & out & ~is_vtx
        ga = np.isin(pdg, (22, 111)) & (pabs > EMIN) & out
        ch_sst1_extra = np.isin(np.nonzero(m)[0], tr["row"][tr["acc"]]) & ~is_vtx
        cnt = lambda mask: np.bincount(np.searchsorted(sel, ev[mask]), minlength=len(sel))
        c_ch, c_ga, c_sst = cnt(ch_wall), cnt(ga), cnt(ch_sst1_extra)
        lep = np.isin(np.abs(pdg), (11, 13, 15))
        ws = w15[sel]
        name = Path(f).stem.split("_nu")[-1]
        rows = {"selected": np.ones(len(sel), bool), ">=1 charged (p>0.1) out the walls": c_ch >= 1,
                ">=1 photon (E>0.1) out the walls": c_ga >= 1, "either": (c_ch >= 1) | (c_ga >= 1),
                "neither (only neutrons / neutrinos / soft)": (c_ch == 0) & (c_ga == 0),
                "extra charged in SST1 (not in the pair)": c_sst >= 1}
        print(f"\nnu {name}: {len(sel)} selected helium events, {ws.sum():.3g} per 15 y; "
              f"mean multiplicity {np.bincount(np.searchsorted(sel, ev), minlength=len(sel)).mean():.1f}, "
              f"charged leptons in the pair {np.isin(np.nonzero(m)[0][lep], vertex_rows).sum() / len(sel):.2f} per event")
        for k, msk in rows.items():
            print(f"   {k:44s} {ws[msk].sum():9.3g}  ({100 * ws[msk].sum() / ws.sum():5.1f} %, n {int(msk.sum())})")
            tot[k] = tot.get(k, 0.0) + ws[msk].sum()
    print("\nall files:")
    for k, v in tot.items():
        print(f"   {k:44s} {v:9.3g}  ({100 * v / tot['selected']:5.1f} %)")


if __name__ == "__main__":
    main()
