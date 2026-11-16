#!/usr/bin/env python3
"""The reference designs R1 / R2 / R3 (V0 rule reco.v0_tag + strips in MS7, 16 Nov 2026) on the interaction backgrounds: MS7 V0 chain and MS7 neutrinos (event stores,
partial and full selection) and the cached vessel muon / neutrino interactions (prepared_*<tag>, full selection), for a
list of designs (with and without the rule). Signal loss and combinatorial pairs: bin/run_reco.py per design (separate).

    python3 bin/v0rule_study.py --designs R1_ubt_only designs/tests/R1_v0.json R2_sbt_ubt designs/tests/R2_v0.json ... \
        --out results_rerun/reco/v0rule_study.json
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
from veto import paths, reco
from veto import vertex_selection as vsel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    R = paths.results_dir()
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--stores", nargs="+", default=[str(R / "evstore" / "ms7_muon_v0s"), str(R / "evstore" / "ms7_nu2026")])
    ap.add_argument("--prep-tag", default="_nov4full")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    designs = {}
    for d in a.designs:
        p = Path(d)
        x = dsg.load(p if p.exists() else d)
        designs[x.name] = x
    _, maps = load_inputs()
    res = {"stores": {}, "prepared": {}}
    for store in a.stores:
        _, ev, trs = es.read(store)
        inter, part = es.to_fastveto(ev, trs)
        del ev, trs
        n = len(inter["weight"])
        tr = ix.charged_tracks(part)
        for ip in ("partial", "full"):
            cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS[ip])
            sel, pair, vertex = ix.select(tr, n, cfg=cfg)
            if len(sel) and cfg.max_tracks is not None:
                n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n)[sel]
                k = n_acc <= cfg.max_tracks
                sel, pair, vertex = sel[k], pair[k], vertex[k]
            key = f"{Path(store).name}/{ip}"
            if not len(sel):
                res["stores"][key] = {}; continue
            w = inter["weight"][sel] * ix.SPILLS_15Y
            trk, evx, z_a, z_b, _ = ix.event_tracks(inter, tr, sel, pair, vertex, None)
            pv = tr["p"][pair[:, 0]] + tr["p"][pair[:, 1]]
            row = dict(n=int(len(sel)), selected_15y=float(w.sum()), n_eff=float(w.sum() ** 2 / (w ** 2).sum()))
            for name, d in designs.items():
                r = reco.run_events(trk, evx, z_b, d, maps, cfg, np.random.default_rng(1), v0=dict(vtx=vertex, pvis=pv))
                v = r["vetoed"][:len(sel)].copy()
                strips = d.shield_strips
                n_strip = 0
                if strips is not None and len(strips["stations"]):     # shield strips: products + parent muon per layer
                    from veto import shield_strips
                    prow = inter["evt_n"] if "muon" in Path(store).name else None
                    st = shield_strips.tag_events(strips, part, inter["z"], prow, sel, vertex, pv, np.random.default_rng(2))["tagged"]
                    n_strip = int((st & ~v).sum()); v |= st
                row[name] = dict(tagged=float((w * v).sum() / w.sum()), left_15y=float((w * ~v).sum()), n_left=int((~v).sum()),
                                 v0_only=r.get("v0_only", 0), strips_only=n_strip)
            res["stores"][key] = row
            print(f"{key}: {len(sel)} selected, {w.sum():.3g} / 15 y; " +
                  ", ".join(f"{k} {v['tagged']:.3f}" for k, v in row.items() if isinstance(v, dict)) + f"  ({time.time() - t0:.0f} s)", flush=True)
        del inter, part, tr
    for kind in ("muon", "nu", "ms7"):
        f = R / "interactions" / f"prepared_{kind}{a.prep_tag}.npz"
        if not f.exists():
            continue
        prep = dict(np.load(f))
        row = {}
        for name, d in designs.items():
            o = ix.veto(prep, d, maps, reco.RecoConfig(seed=1, multiple_scattering=True), np.random.default_rng(8))
            row[name] = dict(total_15y=o["total"], total_err=o["total_err"], n_survive=o["n_survive"],
                             per_target={k: v["survive"] for k, v in o["per_target"].items()})
        res["prepared"][kind] = dict(selected_15y=float(prep["event_weight15"].sum()), n=int(len(prep["event_weight15"])), **row)
        print(f"prepared {kind}: " + ", ".join(f"{k} {v['total_15y']:.3g}" for k, v in row.items()), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
