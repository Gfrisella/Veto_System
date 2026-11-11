#!/usr/bin/env python3
"""Selection and veto of one design on event stores (lib/veto/evstore.py): the stored tracks are intersected with the
design's planes by the existing reconstruction (interactions.evaluate), nothing is regenerated.

    bin/remote.sh run bin/evstore_eval.py --design designs/modules/D6_straw_mod.json --store <dir> [<dir> ...] [--ip-tail]
Prints per target code: interactions, selected, after veto (levels a, b) per 15 years; --ip-tail adds the rate of events
passing everything but the IP cut below IP = 0.1 / 0.25 / 0.5 / 1 / 2 m (the MS7 V0 extrapolation, ~IP^2 near 0).
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_reco import load_inputs
from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import paths
from veto import reco
from veto import vertex_selection as vsel

CUTS = (0.1, 0.25, 0.5, 1.0, 2.0)


def ip_tail(inter, part, cfg, g4=None):
    n = len(inter["weight"])
    tr = ix.charged_tracks(part, g4)
    sel, pair, _ = ix.select(tr, n, cfg=None)
    if not len(sel):
        return {c: (0.0, 0) for c in CUTS}
    n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n)[sel]
    ok = n_acc <= cfg.max_tracks
    sel, pair = sel[ok], pair[ok]
    ip = vsel.measured_ip(tr["b"][pair[:, 0]], tr["p"][pair[:, 0]], tr["b"][pair[:, 1]], tr["p"][pair[:, 1]], cfg, np.random.default_rng(779))
    w15 = inter["weight"][sel] * ix.SPILLS_15Y
    return {c: (float(w15[ip < c].sum()), int((ip < c).sum())) for c in CUTS} | {"all": (float(w15.sum()), len(sel))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True); ap.add_argument("--store", nargs="+", required=True)
    ap.add_argument("--ip-tail", action="store_true"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--export-roots", default=None, help="write the store's roots as an int_/part_ npz (for bin/g4_vessel_transport.py) and stop")
    ap.add_argument("--g4", default=None, help="bin/g4_vessel_transport.py output for the (single) store: evaluate with the Geant4 tracks")
    a = ap.parse_args()
    a.store = [str(paths.evstore(s)) for s in a.store]   # bare process names resolve under paths.EVSTORE_DIR
    if a.export_roots:
        meta, ev, tr = es.read(a.store[0])
        inter, part = es.roots_to_fastveto(ev, tr)
        inter["E"] = np.asarray(ev["E"], float)
        np.savez(a.export_roots, **{f"int_{k}": v for k, v in inter.items()}, **{f"part_{k}": v for k, v in part.items()})
        print(f"{a.store[0]}: {len(inter['weight']):,} events, {len(part['pdg']):,} root particles -> {a.export_roots}")
        return
    p = Path(a.design)
    design = dsg.load(p if p.exists() else a.design)
    _, maps = load_inputs()
    cfg = reco.RecoConfig(seed=a.seed, multiple_scattering=True, ip_max=vsel.IP_CUTS["full"])
    res = {}
    for s in a.store:
        t0 = time.time()
        meta, ev, tr = es.read(s)
        g4 = dict(np.load(a.g4)) if a.g4 else None
        inter, part = es.roots_to_fastveto(ev, tr) if g4 is not None else es.to_fastveto(ev, tr)
        r = ix.evaluate(inter, part, design, maps, cfg, np.random.default_rng(a.seed), g4=g4)
        print(f"\n{s}: {meta['source']} ({meta['n_events']:,} events, {meta['n_tracks']:,} tracks), {time.time() - t0:.0f} s")
        print(f"  {'target':<22}{'interactions/15y':>17}{'selected':>11}{'after a':>11}{'after b':>11}   (n selected)")
        for name, v in r.items():
            print(f"  {name:<22}{v['interactions']:>17.3g}{v['selected']:>11.3g}{v.get('survive_a', 0):>11.3g}{v.get('survive_b', 0):>11.3g}   ({v['n_selected']})")
        out = dict(meta=meta, per_target=r)
        if a.ip_tail:
            t = ip_tail(inter, part, cfg, g4)
            print("  passing all but the IP cut: " + ", ".join(f"IP<{c}: {v[0]:.3g} ({v[1]})" for c, v in t.items()))
            out["ip_tail"] = {str(k): v for k, v in t.items()}
        res[s] = out
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
