#!/usr/bin/env python3
"""ISSUES D19: sum the bin/sbt_eval.py outputs per source / layer / IP cut into one table (markdown + json).

    python3 bin/sbt_summary.py --dir $VETO_RESULTS/sbt_d19 --design R2_sbt_ubt [--out summary_R2.json]

Per source (mu, nu) and SBT layer: enriched store (strange / displaced final states, natural K_L pairs dropped) +
its forced-K_L store (_kl10) + the unflagged part of the unenriched store (pairs with decay-in-flight products, e.g.
pi -> mu). Errors added in quadrature. Columns per 15 y: selected, after the design as it is (std), + V0 line,
+ V0 line + own deposits, bound (own_any), zone rule.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

COLS = ("selected", "std", "v0line", "v0line_self", "own_any", "zone")


def add(acc, key, row):
    a = acc.setdefault(key, {c: [0.0, 0.0, 0] for c in COLS})
    for c in COLS:
        if c in row:
            a[c][0] += row[c]
            a[c][1] += row.get(f"{c}_err", 0.0) ** 2
            a[c][2] += row.get(f"n_{c}" if c != "selected" else "n_selected", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--design", default="R2_sbt_ubt"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    acc = {}
    short = a.design.split("_")[0]                     # per-store and unflagged files carry the short name (R2)
    files = glob.glob(str(d / "per_store" / f"eval_mu_*_{short}_*.json")) + glob.glob(str(d / f"eval_nu_{a.design}_*.json")) \
        + glob.glob(str(d / f"eval_*_unflagged_{short}_*.json"))
    for f in sorted(files):
        r = json.load(open(f))
        ip = r["ip_cut"]
        for s, v in r["stores"].items():
            name = Path(s).name
            if "noenrich" in name and "unflagged" not in Path(f).name:
                continue
            src = "mu" if "mu_" in name else ("nu_He" if "vessel_nu" in name else "nu")
            for t, row in v["per_target"].items():
                if src == "nu_He" and t != "helium":
                    continue
                add(acc, (ip, src, t), row)
    out = {}
    for ip in ("full", "partial"):
        print(f"\n### IP cut {ip} ({a.design}), per 15 y\n")
        print("| source | layer | selected | std (as is) | + V0 line | + V0 line + own deposits | bound: no own fired cell | zone |")
        print("|---|---|---|---|---|---|---|---|")
        for (ip_, src, t), v in sorted(acc.items()):
            if ip_ != ip:
                continue
            cell = lambda c: f"{v[c][0]:.3g} +- {np.sqrt(v[c][1]):.2g} ({v[c][2]})" if v[c][1] > 0 or v[c][0] > 0 else f"0 ({v[c][2]})"
            print(f"| {src} | {t} | " + " | ".join(cell(c) for c in COLS) + " |")
            out[f"{ip}/{src}/{t}"] = {c: dict(value=v[c][0], err=float(np.sqrt(v[c][1])), n=v[c][2]) for c in COLS}
    if a.out:
        json.dump(out, open(d / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
