#!/usr/bin/env python3
"""Collect the outputs of bin/combi_single.py (raw_*.json) and bin/combi_cutflow.py (cutflow.json) into
results_rerun/combi_single/numbers.json: single-track DIS x mu per 15 y for the three selection levels, with and without
the events whose only good track is the scattered muon, DIS x DIS, R2 after tagging, and the step-by-step cut flow.

    python3 bin/combi_single_summary.py --dir $R/combi_single
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

MUMU_REF = dict(source="results_rerun/reco/rock_comb_R2_sbt_ubt{,_partial}.json and the report (reco inputs, same veto_term)",
                cand=dict(per_15y=35.8), ip25=dict(per_15y=34.4, err=2.1), ip01=dict(per_15y=2.36, err=0.52))
ROWS = (("1 mu DIS vessel region x mu", "raw_vessel_mu.json", "vessel_mu x mu", "vessel_mu"),
        ("2 mu DIS MS7 (all products) x mu", "raw_ms7_nu.json", "ms7_mu x mu", "ms7_mu"),
        ("3 nu DIS vessel region x mu", "raw_ms7_nu.json", "vessel_nu x mu", "vessel_nu"),
        ("4 nu DIS MS7 x mu", "raw_ms7_nu.json", "ms7_nu x mu", "ms7_nu"))
LEV = {"cand": "pair cuts (no IP)", "ip25": "IP < 2.5 m", "ip01": "IP < 10 cm"}


def lev(d):
    return {LEV[k]: dict(per_15y=v["per_15y"], err=v["err"], raw_pairs=v["n_pairs_mc"], raw_dis_events=v.get("n_dis_events_mc"))
            for k, v in d.items() if k in LEV}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    raw = {f: json.load(open(d / f)) for f in ("raw_vessel_mu.json", "raw_ms7_nu.json")}
    cf = json.load(open(d / "cutflow.json"))
    out = dict(units="per 15 years = 1.5e7 spills of 4e13 PoT, before tagging unless stated",
               time_factor=raw["raw_ms7_nu.json"]["factor_15y_per_pair_weight"],
               muons=raw["raw_ms7_nu.json"]["muons"], mu_x_mu_reference=MUMU_REF,
               mu_x_mu_closure_same_code={LEV[k]: v for k, v in raw["raw_ms7_nu.json"]["terms"]["mu x mu (closure)"].items() if k in LEV},
               rows={}, dis_x_dis={}, cutflow=cf)
    for name, f, term, store in ROWS:
        t = raw[f]["terms"][term]
        s = raw[f]["stores"][store]
        row = dict(store=s["store"], single_track_events=s["single_track_events"], multi_track_events_rejected=s["multi_track_events"],
                   single_rate_per_spill=s["single_rate_per_spill"], single_rate_per_spill_noscat=s["single_rate_per_spill_noscat"],
                   scattered_muon_rule=s["scattered_muon_rule"], opposite_charge_pairs_simulated=t["opposite_pairs"],
                   with_scattered_muon=lev(t["all"]), without_scattered_muon=lev(t["noscat"]),
                   by_code_without_scattered_muon={c: lev(v) for c, v in t["by_code_noscat"].items()},
                   single_rate_by_code=s["single_rate_by_code"], single_rate_by_code_noscat=s["single_rate_by_code_noscat"])
        for k in ("after_veto_all", "after_veto_noscat", "after_veto_error"):
            if k in t:
                row[k.replace("after_veto", "R2_after_tagging_ip10cm")] = t[k]
        out["rows"][name] = row
    for k, v in raw["raw_ms7_nu.json"]["terms"].items():
        if k in ("ms7_mu x ms7_mu", "ms7_nu x ms7_nu"):
            out["dis_x_dis"][k] = {tag: dict(lev(x), opposite_pairs=x["opposite_pairs"]) for tag, x in v.items()}
    json.dump(out, open(d / "numbers.json", "w"), indent=1, default=float)
    print(f"wrote {d / 'numbers.json'}")
    print(f"{'row':38s}" + "".join(f"{x:>28s}" for x in LEV.values()))
    for name, r in out["rows"].items():
        for tag in ("without_scattered_muon", "with_scattered_muon"):
            print(f"{name[:30] + (' (no scat)' if tag[0] == 'w' and tag.startswith('without') else ''):38s}"
                  + "".join(f"{r[tag][x]['per_15y']:>14.4g} +- {r[tag][x]['err']:<8.2g}({r[tag][x]['raw_pairs']:.3g})" for x in LEV.values()))


if __name__ == "__main__":
    main()
