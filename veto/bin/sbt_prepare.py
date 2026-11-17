#!/usr/bin/env python3
"""ISSUES D19 (a): the SBT-material interactions as a cached background class for the objective (prepared_sbt<tag>.npz,
read by evolve/evaluate_designs.py as interactions kind "sbt", vetoed per design by interactions.veto; weight
w_src_muon = 1 in veto.objective, neutrinos included: nu DIS outside the helium counts like the muons, D1).

    python3 bin/sbt_prepare.py --dir $VETO_RESULTS/sbt_d19 [--ip-cut full|partial] [--tag _nov4full]

Stores (bin/sbt_interactions.py + evstore_build + evstore_redecay): enriched muon / neutrino stores without their natural
K_L decay products (the _kl10 stores carry them), the _kl10 stores, and the unenriched stores restricted to the final
states the enrichment leaves out (decay-in-flight pairs). Per selected event the raw LS deposits of its own particles and
parent muon are stored (sbt_dep_*), so that the design's threshold / efficiency and the flags selftag_hits / zone_rule
act at veto time. A design without SBT (R1) gets no SBT-material events; one with some rings only, the events whose
interaction lies in an instrumented ring (interactions.veto).
"""
import _bootstrap  # noqa: F401
import argparse
import time
from pathlib import Path

import numpy as np

from prepare_interactions import merge_prepared
from sbt_eval import store_inputs
from veto import evstore as es
from veto import geometry as geo
from veto import interactions as ix
from veto import io, paths, reco
from veto import sbt as sbtm
from veto import sbt_material as sm
from veto import vertex_selection as vsel

CODES = tuple(sm.CODES)


def flagged_events(ev, tr, inter):
    """Events whose final state has a strange hadron or a displaced root (what the enriched stores contain)."""
    n = len(inter["weight"])
    eo = np.argsort(ev["event"], kind="stable")
    evi = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr["event"])]
    vx = np.stack([inter["x"], inter["y"], inter["z"]], 1)[evi]
    disp = np.linalg.norm(np.stack([tr["x0"], tr["y0"], tr["z0"]], 1) - vx, axis=1) > sm.DISPLACED_MIN
    fl = (np.asarray(tr["parent"]) < 0) & (np.isin(np.abs(tr["pdg"]), sm.STRANGE) | disp)
    return np.bincount(evi[fl], minlength=n) > 0


def prepare_store(path, cfg, muons, rng):
    name = Path(path).name
    meta, ev, tr = es.read(path)
    inter, part = store_inputs(ev, tr)
    inter["E"] = np.asarray(ev["E"], float)
    if muons is not None:
        inter["muon"] = np.asarray(ev["parent_row"])
    if "noenrich" in name:                               # only the complement of the enriched stores
        inter["weight"] = np.where(flagged_events(ev, tr, inter), 0.0, inter["weight"])
    elif "_kl" not in name:                              # natural K_L decays: carried by the forced-decay store
        keep = part["parent_pdg"] != 130
        part = {k: v[keep] for k, v in part.items()}
    del ev, tr
    pr = ix.prepare(inter, part, cfg, muons, codes=CODES)
    sel = pr["event_inter"]
    if len(sel):
        # pair arrival at SST1: the earliest charged start within 2 cm of the vertex + light speed along the pair line
        vtx, pv = pr["event_vtx"], pr["event_pvis"]
        pos = np.full(len(inter["weight"]), -1); pos[sel] = np.arange(len(sel))
        rows = np.nonzero(pos[part["int"]] >= 0)[0]
        e = pos[part["int"][rows]]
        near = (np.linalg.norm(part["v"][rows] - vtx[e], axis=1) < 0.02) & (ix.charge(part["pdg"][rows]) != 0)
        t_v = np.full(len(sel), np.inf)
        np.minimum.at(t_v, e[near], part["t0"][rows][near])
        bad = ~np.isfinite(t_v)
        x_int = np.stack([inter["x"], inter["y"], inter["z"]], 1)[sel]
        t_v[bad] = inter["t"][sel][bad] + np.linalg.norm(vtx[bad] - x_int[bad], axis=1) / sbtm.C
        k = np.sqrt(1 + (pv[:, 0] / pv[:, 2]) ** 2 + (pv[:, 1] / pv[:, 2]) ** 2)
        t_arr = t_v + (geo.Z_SST1 - vtx[:, 2]) * k / sbtm.C
        d = sm.own_deposits_of_events(sel, inter, part, t_arr, rng, muons)
        ok = d["edep"] > min(sbtm.THRESHOLDS) * 0.5          # keep what can fire at any threshold in use
        pr.update(sbt_dep_event=d["pid"][ok], sbt_dep_cell=d["cell"][ok], sbt_dep_edep=d["edep"][ok], sbt_dep_t=d["t"][ok])
    else:
        pr.update(sbt_dep_event=np.zeros(0, np.int64), sbt_dep_cell=np.zeros(0, np.int64), sbt_dep_edep=np.zeros(0),
                  sbt_dep_t=np.zeros(0))
    pr.pop("event_inter")
    return pr


def merge(parts):
    out = merge_prepared(parts) if len(parts) > 1 else parts[0]
    if len(parts) > 1:                                    # merge_prepared shifts track / strip events; the deposits too
        n_ev, ev = 0, []
        for p in parts:
            ev.append(p["sbt_dep_event"] + n_ev); n_ev += len(p["event_weight15"])
        out["sbt_dep_event"] = np.concatenate(ev)
    out["sbt_material"] = np.array(True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="results_rerun/sbt_d19 (store_mu_*, store_nu_*)")
    ap.add_argument("--ip-cut", choices=("full", "partial"), default="full")
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=19)
    a = ap.parse_args()
    sm.register()
    t0 = time.time()
    d = Path(a.dir)
    cfg = reco.RecoConfig(multiple_scattering=True, ip_max=vsel.IP_CUTS[a.ip_cut])
    muons = io.open_store(paths.store_dir(a.dataset))[0]
    rng = np.random.default_rng(a.seed)
    parts = []
    for s in sorted(d.glob("store_*")):
        if not s.is_dir() or not (s / "meta.json").exists():
            continue
        is_mu = s.name.startswith("store_mu_")
        pr = prepare_store(str(s), cfg, muons if is_mu else None, rng)
        parts.append(pr)
        print(f"{s.name}: {len(pr['event_weight15']):,} selected, {pr['event_weight15'].sum():.4g} / 15 y before the veto, "
              f"{time.time() - t0:.0f} s", flush=True)
    out = merge(parts)
    f = paths.results_dir() / "interactions" / f"prepared_sbt{a.tag}{'_partial' if a.ip_cut == 'partial' else ''}.npz"
    np.savez(f, **out)
    print(f"-> {f}: {len(out['event_weight15']):,} events, {out['event_weight15'].sum():.4g} / 15 y ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
