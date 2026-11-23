#!/usr/bin/env python3
"""Charged-track multiplicity at SST1 per interaction, per source and location (Guglielmo, 23 Nov 2026).

For every interaction ("event") of an event store / the cavern-rock draws: N_SST1 = charged particles (e, mu, pi, K, p,
Sigma, Xi, Omega) reaching SST1, the incoming (target / scattered) muon EXCLUDED, any momentum, any timing acceptance;
N_good = the good ones (p > 1 GeV + SST1 + timing acceptance, the vertex-track definition). Weighted rates per spill
(4e13 PoT) of N_SST1 = 0, 1, 2, 3, >= 4, split by the store's location code, plus the combinatorial single-track
class (N_SST1 = 1 and that track good) and the vertex class (N_good >= 2).

Reaching SST1: event stores (Geant4 / evstore.propagate transport, decays in flight) -> track end code SST1; cavern rock
(straight lines from the rock surface, rock_select.build) -> straight-line crossing of the SST1 aperture.
MS7 stores: the V0 products of the code-3 originals are dropped (their code-4 copies carry them), scattered muon = mu+-
leaving the MS7 exit plane with the parent-muon charge (combi_single rule). Neutrino stores: nothing excluded.
Output: json with per (source, code) rates; bin/sst1_multiplicity.py --table prints the markdown table.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time

import numpy as np

from dis_combinatorial import CHARGED, drop_v0_products_of_originals
from veto import evstore as es
from veto import interactions as ix
from veto import paths

CODE_NAMES = {0: "helium", 1: "vessel air", 2: "vessel wall", 3: "MS7 iron", 4: "MS7 iron (V0 copies)", 8: "air upstream",
              5: "rock wall", 6: "rock floor",
              **{c: f"UBT plane {c - 10}" for c in range(10, 18)},
              105: "rock -x wall", 106: "rock +x wall", 107: "rock floor", 108: "rock ceiling", 109: "rock face",
              9101: "SBT Al inner", 9102: "SBT liquid scintillator", 9103: "SBT ribs (frame)", 9104: "SBT Al outer"}
BINS = ("0", "1", "2", "3", ">=4")


def summarise(w, code, n_all, n_good, good_of_single):
    out = {}
    for c in np.unique(code):
        m = code == c
        ww = w[m]; na = n_all[m]; ng = n_good[m]
        tot = float(ww.sum())
        r = {b: float(ww[(na == k) if k < 4 else (na >= 4)].sum()) for b, k in zip(BINS, range(5))}
        out[str(int(c))] = dict(name=CODE_NAMES.get(int(c), str(int(c))), n_events=int(m.sum()), rate_all=tot,
                                rate_by_nsst1=r, rate_single_good=float(ww[(na == 1) & good_of_single[m]].sum()),
                                rate_single_good_any=float(ww[ng == 1].sum()), rate_vertex=float(ww[ng >= 2].sum()),
                                mean_nsst1_if_any=float((ww * na)[na > 0].sum() / max(ww[na > 0].sum(), 1e-300)))
    return out


def from_store(path, kind):
    meta, ev, tr = es.read(path)
    tr = drop_v0_products_of_originals(ev, tr)
    n_ev = len(ev["event"])
    eo = np.argsort(ev["event"], kind="stable")
    evi = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr["event"])]
    pdg = np.asarray(tr["pdg"])
    scat = np.zeros(len(pdg), bool)
    if kind == "mu":
        if np.asarray(tr["beam"]).any():
            scat = np.asarray(tr["beam"], bool)
        else:                                                         # MS7: parent-muon charge rule
            pr = np.asarray(ev["parent_row"]).astype(np.int64)
            if (pr < 0).any():
                copy_of = np.asarray(ev["copy_of"])
                idx = {int(x): i for i, x in enumerate(ev["event"])}
                po = np.array([pr[idx[int(c)]] if c >= 0 else -1 for c in copy_of])
                pr = np.where(pr >= 0, pr, po)
            qp = np.asarray(np.load(paths.ms7_parents(), mmap_mode="r")["q"])[np.maximum(pr, 0)].astype(int)
            scat = (np.abs(pdg) == 13) & (np.asarray(tr["parent"]) < 0) & (np.sign(-pdg) == qp[evi])
    at = np.isin(np.abs(pdg), CHARGED) & (np.asarray(tr["end"]) == es.END_SST1) & ~scat
    n_all = np.bincount(evi[at], minlength=n_ev)
    # good tracks (fastveto view, as the combinatorial code), scattered muon excluded
    keep = (tr["end"] == es.END_SST1) | ((tr["end"] != es.END_DECAY) & (tr["end"] != es.END_FORCED))
    tr_rows = np.nonzero(keep)[0]
    inter, part = es.to_fastveto(ev, tr)
    t = ix.charged_tracks(part)
    g = np.nonzero(t["good"] & ~scat[tr_rows[t["row"]]])[0]
    n_good = np.bincount(t["inter"][g], minlength=n_ev)
    good_at = np.zeros(n_ev, int)                                     # good tracks that also end at SST1
    g2 = g[at[tr_rows[t["row"][g]]]]
    good_at = np.bincount(t["inter"][g2], minlength=n_ev)
    w = np.asarray(inter["weight"], float)
    return w, np.asarray(ev["code"]).astype(int), n_all, n_good, good_at >= 1


def from_rock(path, kind, seeds):
    from rock_select import build
    inter, part = build(np.load(path), kind, seeds)
    t = ix.charged_tracks(part)
    notbeam = ~np.asarray(t["beam"], bool)
    n_ev = len(inter["weight"])
    a = t["acc"] & notbeam
    n_all = np.bincount(t["inter"][a], minlength=n_ev)
    n_good = np.bincount(t["inter"][t["good"] & notbeam], minlength=n_ev)
    code = np.asarray(inter["code"]).astype(int)
    if kind != "muon":                       # rock_select.build: nu regions 0 -x wall, 1 +x wall, 2 floor, 3 ceiling, 4 face
        code = code + 100                    # -> 105..109 (muon draws: 5 wall, 6 floor)
    return np.asarray(inter["weight"], float), code, n_all, n_good, n_good >= 1


def table(res):
    lines = ["| source | location | interactions / spill | N_SST1 = 0 | 1 | 2 | 3 | >= 4 | single (1, good) | vertex (>= 2 good) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for src, v in res["sources"].items():
        for c, r in v["codes"].items():
            tot = r["rate_all"]
            f = lambda x: f"{x:.3g} ({100 * x / tot:.1f} %)" if tot > 0 else "0"
            lines.append(f"| {src} | {r['name']} | {tot:.3g} | " + " | ".join(f(r['rate_by_nsst1'][b]) for b in BINS)
                         + f" | {f(r['rate_single_good'])} | {f(r['rate_vertex'])} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", nargs="*", default=[], help="label=path:kind (kind mu | nu)")
    ap.add_argument("--rock", nargs="*", default=[], help="label=draws.npz:kind (kind muon | nu)")
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", action="store_true", help="only print the table of an existing --out")
    a = ap.parse_args()
    if a.table:
        print(table(json.load(open(a.out)))); return
    res = dict(method=__doc__.split("\n\n")[0], sources={})
    t0 = time.time()
    for spec in a.stores + a.rock:
        label, rest = spec.split("=", 1)
        path, kind = rest.rsplit(":", 1)
        if spec in a.rock:
            data = from_rock(path, kind, a.seeds)
        else:
            data = from_store(path, kind)
        res["sources"][label] = dict(path=path, kind=kind, codes=summarise(*data))
        print(f"{label}: {time.time() - t0:.0f} s", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(table(res))


if __name__ == "__main__":
    main()
