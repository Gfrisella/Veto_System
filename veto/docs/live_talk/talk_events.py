"""Heavy inputs of the live talk, computed once per design (on the cluster) and read by veto_talk.py.

For every design in veto/designs: the full reconstruction (196 736 muon pairs, 60 k signal decays,
200 k single muons), then
  events  up to 10 example events of each kind (pair vetoed, pair missed, signal lost to a random
          hit, bounced muon missed): paths, predictions, search windows, own and random hits
  fom     the confusion-matrix numbers (TP, FN in events over 15 years; FP, TN as fractions)
  why     why each muon of a surviving pair was not explained
Writes veto/results/reco/talk_events.pkl. With the straw UBT (+-25 ns) an event holds ~470 random
hits, so one design needs a few GB and a few minutes: run it on the cluster, not on a laptop.

  bin/remote.sh run docs/live_talk/talk_events.py          # one process per design
  bin/remote.sh fetch-results reco                         # then present as usual
"""
from __future__ import annotations

import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VETO = HERE.parents[1]
sys.path[:0] = [str(VETO / "lib"), str(VETO / "bin")]

from run_reco import TYPE_NAMES, load_inputs, take  # noqa: E402
from veto import design as dsg  # noqa: E402
from veto import paths, reco  # noqa: E402

SPILL_S = 1.0
SPILLS_15Y = 6e20 / 5e13
N_EXAMPLES = 10
KINDS = ["background pair, vetoed", "background pair, missed", "signal lost to a random hit", "bounced muon, missed"]
REASONS = ["crosses no detector upstream", "hit lost (inefficiency)", "bounced: prediction off by metres",
           "outside the space window", "outside the time window"]
OUT = paths.results_dir() / "reco" / "talk_events.pkl"


def run_design(name, tr, inp, cfg):
    d = dsg.load(name)
    rng = np.random.default_rng(11)
    cand = inp["cand"]
    sub = take(tr, cand.ravel())
    vz = np.repeat(inp["cand_vertex"][:, 2], 2)
    b = reco.run_events(sub, np.repeat(np.arange(len(cand)), 2), vz, d, inp, cfg, rng, details=True)
    sig, zv, ev, _ = reco.signal_from_target(60_000, rng)
    s = reco.run_events(sig, ev, zv, d, inp, cfg, rng, with_own_hits=False, details=True)
    single = inp["single"]
    st = take(tr, single)
    m = reco.run_events(st, np.arange(len(single)), np.full(len(single), 1e9), d, inp, cfg, rng, details=True)
    return dict(design=d, sub=sub, vz=vz, b=b, sig=sig, s=s, st=st, m=m, cand_w=inp["cand_w"])


def pick_events(r, kind):
    """Event ids and their tracks for up to N_EXAMPLES events of a kind."""
    if kind in (0, 1):
        ev = np.where(r["b"]["vetoed"] if kind == 0 else ~r["b"]["vetoed"])[0]
        if kind == 0 and len(ev):
            ev = ev[(np.arange(N_EXAMPLES) * 7919) % len(ev)]
        return [(e, [2 * e, 2 * e + 1], r["sub"], r["b"]) for e in ev[:N_EXAMPLES]]
    if kind == 2:
        n = len(r["s"]["vetoed"])
        return [(e, [e, e + n], r["sig"], r["s"]) for e in np.where(r["s"]["vetoed"])[0][:N_EXAMPLES]]
    ev = np.where(~r["m"]["vetoed"] & (r["st"]["type"] == 0))[0]
    return [(e, [e], r["st"], r["m"]) for e in ev[:N_EXAMPLES]]


def path_points(tr, t):
    a = [tr["ax"][t], tr["ay"][t], tr["az"][t]]
    b = [tr["bx"][t], tr["by"][t], tr["bz"][t]]
    k = [tr["kx"][t], tr["ky"][t], tr["kz"][t]]
    return [a, k, b] if np.isfinite(k[2]) else [a, b]


def event_parts(r, kind, e, tracks, tr, res, inp):
    """Everything the event slides draw, as plain lists (windows refer to planes by index)."""
    d = r["design"]
    hits, q, meas = res["hits"], res["queries"], res["meas"]
    parts = dict(paths=[], preds=[], windows=[], matched=[], own=[], acc=[], vtx=None, info=[])
    for t in tracks:
        if "ax" in tr:
            parts["paths"].append(path_points(tr, t))
        else:
            parts["paths"].append([[tr["vx"][t], tr["vy"][t], tr["vz"][t]], [tr["bx"][t], tr["by"][t], tr["bz"][t]]])
            parts["vtx"] = [tr["vx"][t], tr["vy"][t], tr["vz"][t]]
        for j in np.where(q["track"] == t)[0]:
            plane = d.planes[int(q["plane"][j])]
            xp, yp = plane.to_global(q["u"][j]), q["y"][j]
            parts["preds"].append([[meas["x"][t], meas["y"][t], meas["z"][t]], [xp, yp, plane.z]])
            parts["windows"].append((plane.index, float(xp), float(yp), float(q["r"][j])))
            if res["found"][j]:
                h = res["hit_of_query"][j]
                hp = d.planes[int(hits["plane"][h])]
                parts["matched"].append([hp.to_global(hits["u"][h]), hits["y"][h], hp.z])
                parts["info"].append(f"track {tracks.index(t) + 1} hit: {hp.name.replace(' fin z=', ' ')} m")
                parts["info"].append(f"   Δt = {hits['t'][h] - q['t'][j]:+.2f} ns")
        p = math.sqrt(tr["bpx"][t] ** 2 + tr["bpy"][t] ** 2 + tr["bpz"][t] ** 2)
        typ = TYPE_NAMES.get(int(tr["type"][t]), "") if "type" in tr else "signal"
        short = {"SST1 bounced": "bounced", "SST1 p>1 GeV & timing": "selected", "SST1 other": "other"}.get(typ, typ)
        parts["info"].insert(0, f"track {tracks.index(t) + 1}: {short} muon, {p:.1f} GeV")
    for h in np.where(hits["event"] == e)[0]:
        hp = d.planes[int(hits["plane"][h])]
        pt = [hp.to_global(hits["u"][h]), hits["y"][h], hp.z]
        (parts["own"] if hits["track"][h] in tracks else parts["acc"]).append(pt)
    if kind in (0, 1):
        parts["vtx"] = list(inp["cand_vertex"][e])
    return parts


def diagnose(r, cfg):
    """Why each muon of a surviving pair was not explained."""
    d, sub, b, vz = r["design"], r["sub"], r["b"], r["vz"]
    hits, meas = b["hits"], b["meas"]
    counts = np.zeros(len(REASONS), int)
    for e in np.where(~b["vetoed"])[0]:
        for t in (2 * e, 2 * e + 1):
            one = {k: v[t:t + 1] for k, v in sub.items()}
            crossed = [p for p in d.planes if p.z < vz[t] and reco.true_crossing(one, p)[3][0]]
            if not crossed:
                counts[0] += 1
                continue
            on = [h for h in np.where(hits["track"] == t)[0] if d.planes[int(hits["plane"][h])] in crossed]
            if not on:
                counts[1] += 1
                continue
            h = on[0]
            plane = d.planes[int(hits["plane"][h])]
            dz = meas["z"][t] - plane.z
            pu = plane.to_local(meas["x"][t] - meas["tx"][t] * dz)
            py = meas["y"][t] - meas["ty"][t] * dz
            dist = dz * math.sqrt(1 + meas["tx"][t] ** 2 + meas["ty"][t] ** 2)
            rr = cfg.r0 + (cfg.r_slope + cfg.r_ms / max(meas["p"][t], 0.1)) * dist
            far = max(abs(hits["u"][h] - pu), abs(hits["y"][h] - py))
            if far > rr + plane.cell_u / 2:
                counts[2 if (np.isfinite(sub["kz"][t]) and far > 1) else 3] += 1
            else:
                counts[4] += 1
    return counts.tolist()


def fom_numbers(r):
    w15 = r["cand_w"] * 2 * 200e-12 / SPILL_S * SPILLS_15Y
    veto_b, veto_s = r["b"]["vetoed"], r["s"]["vetoed"]
    surv = ~veto_b
    return dict(tp=float(w15[veto_b].sum()), fn=float(w15[surv].sum()),
                fn_err=float(math.sqrt((w15[surv] ** 2).sum()) if surv.any() else 2.3 * w15.mean()),
                n_surv=int(surv.sum()), n_cand=int(len(veto_b)), fp=float(veto_s.mean()), tn=float(1 - veto_s.mean()))


def one_design(name):
    tr, inp = load_inputs()
    cfg = reco.RecoConfig(spill_s=SPILL_S)
    r = run_design(name, tr, inp, cfg)
    events = {kind: [event_parts(r, kind, e, tracks, trk, res, inp) for e, tracks, trk, res in pick_events(r, kind)]
              for kind in range(len(KINDS))}
    out = dict(events=events, fom=fom_numbers(r), why=diagnose(r, cfg))
    print(f"{name}: {out['fom']['n_surv']} pairs survive, signal lost {100 * out['fom']['fp']:.2f} %, "
          f"events per kind {[len(v) for v in events.values()]}", flush=True)
    return name, out


def main():
    names = sorted(p.stem for p in dsg.DESIGN_DIR.glob("*.json"))
    with Pool(len(names)) as pool:
        result = dict(pool.map(one_design, names))
    OUT.write_bytes(pickle.dumps(result))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
