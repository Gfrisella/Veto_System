#!/usr/bin/env python3
"""Decay chains of the MS7 events (bin/ms7_events_viz.py json) for the event-by-event page: every particle that leaves
the shield iron flies on as a straight line (no field after the shield, no hadronic interactions in air / helium) and
decays in flight with its lifetime (pi+-, K+-, K_L, K_S, Lambda, Sigma, Xi, Omega; pi0 at once), the products again,
until it leaves the cavern (ECN3 box) or passes z = Z_END. In the forced-decay events the V0 keeps its simulated decay.

    python3 bin/ms7_event_chains.py <ms7_events.json> --out <ms7_chains.json>
Each event gets "chain": rows [pdg, parent row (-1 = out of the iron), x0, y0, z0, x1, y1, z1, |p|, end] with end
0 = leaves the cavern / passes Z_END, 1 = decays, 2 = the forced V0 decay, 3 = stable particle.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np

from fastveto import decay as dc

Z_END = 95.0
CAV = dict(x0=-4.56, x1=11.43, y0=-3.36, y1=10.64)
CTAU = {**dc.CTAU, 211: 7.8045, 321: 3.712}
CH = {**dc.CHANNELS, **dc.KL_CHANNELS}
CH[211] = [(1.0, (-13, 14))]
CH[321] = [(0.6356, (-13, 14)), (0.2067, (211, 111)), (0.0559, (211, 211, -211)), (0.0507, (111, -11, 12)),
           (0.0335, (111, -13, 14)), (0.0176, (211, 111, 111))]
MASS = {**dc.MASS, 16: 0.0}
NEUTRINO = (12, 14, 16)


def exit_length(x, u):
    """Path length to leave the cavern box or pass Z_END (u unit vector)."""
    s = np.inf
    for k, (lo, hi) in enumerate(((CAV["x0"], CAV["x1"]), (CAV["y0"], CAV["y1"]))):
        if u[k] > 0:
            s = min(s, (hi - x[k]) / u[k])
        elif u[k] < 0:
            s = min(s, (lo - x[k]) / u[k])
    if u[2] > 0:
        s = min(s, (Z_END - x[2]) / u[2])
    elif u[2] < 0:
        s = min(s, (29.9 - x[2]) / u[2])
    return max(s, 0.0)


def products(pdg, p, rng):
    """Decay products (pdg list, lab momenta) of one particle of momentum p (3,)."""
    a = abs(pdg)
    M = MASS[a]
    br = np.array([c[0] for c in CH[a]])
    prods = CH[a][int(np.searchsorted(np.cumsum(br / br.sum()), rng.random(), side="right"))][1]
    ms = [MASS[abs(q)] for q in prods]
    if len(prods) == 2:
        q = dc._two_body(M, ms[0], ms[1], 1, rng)[0]
        rest = [q, -q]
    else:
        rest = list(dc._three_body(M, ms, 1, rng)[0])
    out = []
    for q_rest, code, mi in zip(rest, prods, ms):
        e_rest = np.sqrt((q_rest ** 2).sum() + mi ** 2)
        plab, _ = dc._boost(q_rest[None], np.array([e_rest]), p[None], M)
        code = code if pdg > 0 else dc._conj(code)
        out.append((int(code), plab[0]))
    return out


def chain(e, rng):
    rows = []
    todo = []                                      # (pdg, parent row, start (3,), p (3,), forced decay or None)
    forced = None
    if e.get("v0line"):
        dau = [(int(q[0]), np.array(q[4:7], float)) for q in e["parts"] if q[7]]
        forced = (np.array(e["dec"], float), dau)
        todo.append((e["v0"], -1, np.array(e["v0line"][:3], float), np.array(e["v0line"][3:6], float), forced))
    for q in e["parts"]:
        if not q[7]:
            todo.append((int(q[0]), -1, np.array(q[1:4], float), np.array(q[4:7], float), None))
    while todo:
        pdg, par, x, p, fd = todo.pop(0)
        pm = float(np.linalg.norm(p))
        if pm <= 0:
            continue
        u = p / pm
        a = abs(pdg)
        s_exit = exit_length(x, u)
        row = len(rows)
        if fd is not None:
            end, kind = fd[0], 2
            for code, pp in fd[1]:
                todo.append((code, row, end.copy(), pp, None))
        elif a in CH and a in CTAU and a not in NEUTRINO:
            L = rng.exponential(CTAU[a]) * pm / MASS[a]
            if L < s_exit:
                end, kind = x + u * L, 1
                for code, pp in products(pdg, p, rng):
                    todo.append((code, row, end.copy(), pp, None))
            else:
                end, kind = x + u * s_exit, 0
        else:
            end, kind = x + u * s_exit, (0 if a in NEUTRINO or s_exit > 0 else 3)
        rows.append([int(pdg), par, *np.round(x, 3).tolist(), *np.round(end, 3).tolist(), round(pm, 4), kind])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events"); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    D = json.load(open(a.events))
    rng = np.random.default_rng(a.seed)
    n = 0
    for e in D["random"] + D["selected"]:
        e["chain"] = chain(e, rng)
        n += len(e["chain"])
    D["chain_info"] = dict(z_end=Z_END, cavern=CAV, rows=n, note="straight lines, natural decays in flight, no field / interactions after the shield")
    json.dump(D, open(a.out, "w"), separators=(",", ":"))
    print(f"{len(D['random']) + len(D['selected'])} events, {n:,} chain rows -> {a.out}")


if __name__ == "__main__":
    main()
