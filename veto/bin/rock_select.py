#!/usr/bin/env python3
"""Selection and veto of the cavern-rock products (the same interactions.evaluate as the vessel interactions): the
draws saved by bin/rock_apply.py (muon interactions in the rock, *_draws.npz) or bin/nu_rock.py combine (neutrino
interactions in the rock, nu_rock_<design>_draws.npz). Each draw is one interaction; its particles start as straight
lines at their exit points on the rock surface. Weights per spill of 4e13 p.o.t. (rescaled to interactions.SPILLS_15Y).
Grouped by rock region (muons: wall / floor; neutrinos: -x wall, +x wall, floor, ceiling, face).

    python3 bin/rock_select.py --kind muon --draws /disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz \
        --seeds 300 --design designs/modules/D6_straw_mod.json --out rock_select_mu.json
    python3 bin/rock_select.py --kind nu --draws /disk/lhcb_data/sqasim/nu_rock/nu_rock_D6_straw_mod_draws.npz \
        --design designs/modules/D6_straw_mod.json --out rock_select_nu.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_reco import load_inputs
from veto import design as dsg
from veto import interactions as ix
from veto import reco
from veto import vertex_selection as vsel

POT_SPILL = 4e13
GROUPS = {"muon": ["rock wall (muons)", "rock floor (muons)"],
          "nu": ["rock -x wall (nu)", "rock +x wall (nu)", "rock floor (nu)", "rock ceiling (nu)", "rock face (nu)"]}


def build(z, kind, seeds):
    draw = np.asarray(z["draw"])
    ids, first, inv = np.unique(draw, return_index=True, return_inverse=True)
    w = np.asarray(z["w"], float)[first] * (5e13 / POT_SPILL)
    if kind == "muon":
        w = w / seeds                                   # rock_apply: each draw carries S / M per seed
        grp = np.asarray(z["surf"])[first].astype(int)  # 0 wall, 1 floor
        evt = np.asarray(z["seed"])[first]
    else:
        reg = np.asarray(z["region"])[first].astype(int)
        grp = np.minimum(reg, 4)                        # 0 -x wall, 1 +x wall, 2 floor, 3 ceiling, 4+ face (top, deep)
        evt = ids
    inter = dict(weight=w, code=5 + grp, evt_n=evt, x=np.asarray(z["vx"], float)[first], y=np.asarray(z["vy"], float)[first],
                 z=np.asarray(z["vz"], float)[first])
    n = len(draw)
    part = dict(pdg=np.asarray(z["pdg"]).astype(np.int64), int=inv,
                p=np.stack([z["px"], z["py"], z["pz"]], 1).astype(float),
                v=np.stack([z["x"], z["y"], z["z"]], 1).astype(float),
                beam=np.asarray(z["beam"], bool) if "beam" in z.files else np.zeros(n, bool))
    return inter, part


def cutflow(inter, part, cfg, seed):
    """Weighted cut flow per 15 years: >= 2 good tracks (SST1 + timing acceptance, p > 1 GeV), a vertex pair (no IP cut),
    measured IP to the target below 10 cm / 1 m / 5 m; the IP of the best pair."""
    n = len(inter["weight"])
    w15 = inter["weight"] * ix.SPILLS_15Y
    tr = ix.charged_tracks(part)
    ng = np.bincount(tr["inter"][tr["good"]], minlength=n)
    rows = [("all saved draws", np.ones(n, bool)), (">= 2 good tracks", ng >= 2)]
    sel, pair, _ = ix.select(tr, n, cfg=None)
    has = np.zeros(n, bool); has[sel] = True
    rows.append(("vertex pair (no IP cut)", has))
    good = np.nonzero(tr["good"])[0]
    best = np.full(n, np.inf)
    if len(good):                                    # best measured IP over all good opposite-charge pairs of an event
        gi = tr["inter"][good]
        o = np.argsort(gi, kind="stable"); good, gi = good[o], gi[o]
        starts = np.r_[0, np.nonzero(np.diff(gi))[0] + 1]; ends = np.r_[starts[1:], len(gi)]
        i0, i1 = [], []
        for s, e in zip(starts, ends):
            if e - s >= 2:
                g = good[s:e][:20]
                a_, b_ = np.triu_indices(len(g), 1)
                i0.append(g[a_]); i1.append(g[b_])
        if i0:
            i0, i1 = np.concatenate(i0), np.concatenate(i1)
            opp = tr["q"][i0] != tr["q"][i1]
            i0, i1 = i0[opp], i1[opp]
            ip = vsel.measured_ip(tr["b"][i0], tr["p"][i0], tr["b"][i1], tr["p"][i1], cfg, np.random.default_rng(seed))
            np.minimum.at(best, tr["inter"][i0], ip)
    for cut in (0.1, 1.0, 5.0):
        rows.append((f"opposite pair with IP < {cut:g} m", best < cut))
    print(f"\n  cut flow (per 15 years, n draws):")
    for name, m in rows:
        print(f"    {name:34s} {w15[m].sum():10.4g}  ({int(m.sum())})")
    fin = np.isfinite(best)
    if fin.any():
        q = np.percentile(best[fin], [0, 1, 10, 50])
        print(f"    best-pair IP: min {q[0]:.3g} m, 1 % {q[1]:.3g}, 10 % {q[2]:.3g}, median {q[3]:.3g} m ({int(fin.sum())} draws)")
    print(f"    mean weight of a draw: {w15.mean():.3g} per 15 years", flush=True)
    wsp = inter["weight"][tr["inter"]] * (POT_SPILL / 5e13)          # per spill of 4e13
    pid = np.abs(part["pdg"][tr["row"]])
    print(f"    single tracks per spill (4e13): in SST1 acceptance {wsp[tr['acc']].sum():.3g}, good (p > 1 GeV, timing) "
          f"{wsp[tr['good']].sum():.3g}, of which muons {wsp[tr['good'] & (pid == 13)].sum():.3g}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("muon", "nu"), required=True)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--seeds", type=int, default=300, help="muons: seeds (spills) of the rock_apply run")
    ap.add_argument("--design", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    p = Path(a.design)
    design = dsg.load(p if p.exists() else a.design)
    for k, name in enumerate(GROUPS[a.kind]):
        ix.CODE_NAMES[5 + k] = name
    inter, part = build(np.load(a.draws), a.kind, a.seeds)
    print(f"{len(inter['weight']):,} draws, {len(part['pdg']):,} particles, {inter['weight'].sum():.4g} per spill (5e13), "
          f"{time.time() - t0:.0f} s", flush=True)
    _, maps = load_inputs()
    cfg = reco.RecoConfig(seed=a.seed, multiple_scattering=True, ip_max=vsel.IP_CUTS["full"])
    cutflow(inter, part, cfg, a.seed)
    res = ix.evaluate(inter, part, design, maps, cfg, np.random.default_rng(a.seed))
    print(f"\n{'region':<22}{'draws w/15y':>13}{'selected':>12}{'+-':>9}{'after a':>11}{'after b':>11}{'+-':>9}   (per 15 years; n selected)")
    tot = dict(selected=0.0, survive_a=0.0, survive_b=0.0)
    for name, r in res.items():
        print(f"{name:<22}{r['interactions']:>13.3g}{r['selected']:>12.3g}{r['selected_err']:>9.2g}{r.get('survive_a', 0):>11.3g}"
              f"{r.get('survive_b', 0):>11.3g}{r.get('survive_b_err', 0):>9.2g}   ({r['n_selected']})")
        for k in tot:
            tot[k] += r.get(k, 0.0)
    print(f"{'total':<22}{'':>13}{tot['selected']:>12.3g}{'':>9}{tot['survive_a']:>11.3g}{tot['survive_b']:>11.3g}")
    json.dump(dict(kind=a.kind, draws=a.draws, design=design.name, per_region=res, total=tot, seconds=round(time.time() - t0, 1)),
              open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
