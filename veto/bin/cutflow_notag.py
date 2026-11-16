#!/usr/bin/env python3
"""Cut flow WITHOUT tagging on event stores, for the report's Section 6 tables (16 Nov 2026).

Same selection as interactions.evaluate (charged_tracks -> select -> <= max_tracks), but every level is recorded:
    interactions | >= 2 good tracks | pair cuts, no IP | IP < 10 cm (full) | IP < 2.5 m (partial)
per target code, and split by
    - neutrino flavour (--flavour-from: the apply*.npz the store was built from, rows aligned one-to-one;
      the store's meta declares 'flavour' but events.npz does not carry it),
    - V0 species for the MS7 stores (events.npz 'v0': 130 K_L, 310 K_S, |3122| Lambda, |3322| Xi, 0 = no V0).
Per 15 years (SPILLS_15Y, 4e13 PoT / spill); errors by bootstrap over the parent (evt_n), as evaluate.
No tagging -> design-independent apart from the target codes of the tagger material (use R2's stores).

    bin/cutflow_notag.py --design designs/R2_sbt_ubt.json --store <dir> [--flavour-from apply.npz] ... --out cutflow.json
Each --store may be followed by its own --flavour-from (positional pairing by order; '-' = none).
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import reco
from veto import vertex_selection as vsel

V0_NAME = {0: "no V0", 130: "K_L", 310: "K_S", 3122: "Lambda", 3322: "Xi0"}


def levels(inter, part, cfg_max_tracks):
    """Boolean masks over interactions for each cut level (same rng as evaluate: default_rng(779) inside select)."""
    n = len(inter["weight"])
    tr = ix.charged_tracks(part)
    n_good = np.bincount(tr["inter"][tr["good"]], minlength=n)
    n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n)
    out = {"interactions": np.ones(n, bool), "ge2_tracks": n_good >= 2}
    for name, ip in (("pair_noIP", None), ("ip_full", vsel.IP_CUTS["full"]), ("ip_partial", vsel.IP_CUTS["partial"])):
        cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=ip)
        sel, _, _ = ix.select(tr, n, cfg=cfg)
        if cfg_max_tracks is not None and len(sel):
            sel = sel[n_acc[sel] <= cfg_max_tracks]
        m = np.zeros(n, bool); m[sel] = True
        out[name] = m
    return out


def table(inter, masks, group, names, rng):
    w15 = inter["weight"] * ix.SPILLS_15Y
    g = inter["evt_n"] if "evt_n" in inter else np.arange(len(w15))
    res = {}
    for key in np.unique(group):
        mg = group == key
        row = {}
        for lev, m in masks.items():
            mm = mg & m
            row[lev] = float(w15[mm].sum())
            row[f"n_{lev}"] = int(mm.sum())
            if lev != "interactions":
                row[f"{lev}_err"] = ix.bootstrap(w15[mm], g[mm], rng=rng)
        res[names(key)] = row
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--store", nargs="+", required=True)
    ap.add_argument("--flavour-from", nargs="*", default=[], help="one per store, '-' for none")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    design = dsg.load(Path(a.design))
    max_tracks = reco.RecoConfig().max_tracks
    flav = a.flavour_from + ["-"] * (len(a.store) - len(a.flavour_from))
    res = {}
    for s, ff in zip(a.store, flav):
        t0 = time.time()
        meta, ev, tr = es.read(s)
        inter, part = es.to_fastveto(ev, tr)
        masks = levels(inter, part, max_tracks)
        rng = np.random.default_rng(1)
        code = inter["code"]
        out = dict(meta=meta, max_tracks=max_tracks,
                   per_target=table(inter, masks, code, lambda c: ix.target_name(int(c), design), rng))
        if ff != "-":
            z = np.load(ff)
            assert len(z["int_x"]) == len(code) and np.allclose(z["int_x"], ev["x"]) and np.array_equal(z["int_code"], code), \
                f"{ff} not row-aligned with {s}"
            fl = np.asarray(z["int_flavour"])
            region = np.where(code == 0, "helium", "other")
            key = np.char.add(np.char.add(region, " nu"), fl.astype(str))   # helium vs rest x flavour
            out["per_flavour_helium_vs_rest"] = table(inter, masks, key, str, rng)
            out["per_flavour"] = table(inter, masks, fl, lambda f: f"nu{int(f)}", rng)
        if "v0" in ev and np.any(ev["v0"] != 0):
            v0 = np.abs(np.asarray(ev["v0"]))
            out["per_v0"] = table(inter, masks, v0, lambda v: V0_NAME.get(int(v), str(int(v))), rng)
        res[s] = out
        print(f"\n{s} ({meta['n_events']:,} events) {time.time() - t0:.0f} s")
        for name, r in out["per_target"].items():
            print(f"  {name:<42}" + "".join(f"{r[l]:>11.3g}" for l in masks))
        for extra in ("per_flavour", "per_v0"):
            for name, r in out.get(extra, {}).items():
                print(f"  {extra}:{name:<32}" + "".join(f"{r[l]:>11.3g}" for l in masks))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
