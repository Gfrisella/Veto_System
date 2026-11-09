#!/usr/bin/env python3
"""Adaptive weighted splitting through the SHiP muon shield with the existing GPU transport (sandbox prototype, README.md).

    python3 splitting.py --input <pool chunk .h5> --n-src 4000000 --batches 10 --out split.npz            # splitting
    python3 splitting.py --input <pool chunk .h5> --n-src 4000000 --batches 10 --out analog.npz --analog   # plain MC

Per checkpoint z_k: transport every live muon from its exact paused state to z_k (generate_muon_study.Simulation's
propagate, untouched physics), score, then (splitting mode) population control: split muons in thin cells of
(charge, log p, predicted x, y at z = 82 m) into independent continuations with divided weights, Russian-roulette crowded
cells (never below K_MIN), cap the population per batch. Every step is unbiased whatever the cells; genealogy is kept.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VETO = HERE.parent.parent
sys.path.insert(0, str(VETO / "bin"))
sys.path.insert(0, str(VETO / "lib"))
import generate_muon_study as G                                          # noqa: E402

SHIELD_ENDS = [2.30, 5.45, 10.10, 14.76, 19.41, 22.19, 25.70, 29.89]     # TRY6 pieces (geometry.SHIELD_TRY6: gaps + 2 dZ)
CHECKPOINTS = SHIELD_ENDS + [30.6, 40.0, 50.0, 60.0, 70.0, 82.0]
Z_SST = 82.0
XY = np.linspace(-12, 12, 97)                 # m, 25 cm: x-y scoring grid at every checkpoint
PB = np.geomspace(0.1, 400, 41)               # GeV
SL = np.linspace(-0.3, 0.3, 61)               # slopes px/pz, py/pz
ZSTOP = np.arange(0, 82.5, 0.5)
# control cells: charge x log p x predicted (x, y) at z = 82 m
PRED = np.linspace(-15, 15, 31)               # 1 m
PCUT = np.array([0, 2, 5, 10, 20, 40, 80, 160, 1e9])
FIELDS = ("x", "y", "z", "px", "py", "pz", "q", "w", "root", "id", "parent", "depth", "batch")


def take(P, idx):
    return {k: v[idx] for k, v in P.items()}


def cells(P):
    p = np.sqrt(P["px"] ** 2 + P["py"] ** 2 + P["pz"] ** 2)
    pz = np.where(P["pz"] > 1e-6, P["pz"], 1e-6)
    dz = Z_SST - P["z"]
    xp = np.where(P["pz"] > 1e-6, P["x"] + P["px"] / pz * dz, 99.0)     # straight-line prediction at the SST plane
    yp = np.where(P["pz"] > 1e-6, P["y"] + P["py"] / pz * dz, 99.0)
    ix = np.clip(np.digitize(xp, PRED), 0, len(PRED))                    # 0 and len(PRED) = outside the grid
    iy = np.clip(np.digitize(yp, PRED), 0, len(PRED))
    ip = np.clip(np.digitize(p, PCUT) - 1, 0, len(PCUT) - 2)
    iq = (P["q"] > 0).astype(np.int64)
    n = len(PRED) + 1
    return ((iq * (len(PCUT) - 1) + ip) * n + ix) * n + iy


def cells_state(P):
    """cells of the CURRENT state: charge x log p x position x direction (no prediction of where it goes)"""
    p = np.sqrt(P["px"] ** 2 + P["py"] ** 2 + P["pz"] ** 2)
    pz = np.where(P["pz"] > 1e-6, P["pz"], 1e-6)
    XE, SE = np.linspace(-6, 6, 25), np.linspace(-0.1, 0.1, 9)
    ix, iy = np.digitize(P["x"], XE), np.digitize(P["y"], XE)
    sx, sy = np.digitize(P["px"] / pz, SE), np.digitize(P["py"] / pz, SE)
    ip = np.clip(np.digitize(p, PCUT) - 1, 0, len(PCUT) - 2)
    iq = (P["q"] > 0).astype(np.int64)
    return ((((iq * 8 + ip) * 26 + ix) * 26 + iy) * 10 + sx) * 10 + sy


def control_tempered(P, a, rng, next_id, z):
    """resample each current-state cell to C n_c^alpha rows (alpha = 1: analog; 0: flat), total kept ~ the current
    population (at most N_MAX per batch): split (weight / s) or roulette (p >= K_MIN, weight / p); no splitting of muons
    that cannot leave the remaining iron (p below ~1.6 GeV/m x the iron left before the shield end). Unbiased."""
    c = cells_state(P)
    u, inv, cnt = np.unique(c, return_inverse=True, return_counts=True)
    m = cnt.astype(np.float64) ** a.alpha
    N = min(len(c), a.n_max * len(np.unique(P["batch"])))
    r = (m * N / m.sum())[inv] / cnt[inv]
    p = np.sqrt(P["px"].astype(np.float64) ** 2 + P["py"] ** 2 + P["pz"] ** 2)
    trapped = p < 1.6 * max(0.0, SHIELD_ENDS[-1] - z)
    r = np.where(trapped & (r > 1), 1.0, r)
    s = np.where(r > 1, np.floor(r) + (rng.random(len(r)) < (r - np.floor(r))), 1).astype(np.int64)
    s = np.minimum(s, a.s_max)
    keep_p = np.where(r < 1, np.maximum(a.k_min, r), 1.0)
    kept = rng.random(len(r)) < keep_p
    w = P["w"] / keep_p
    idx = np.nonzero(kept)[0]
    s, w = s[idx], w[idx]
    rep = np.repeat(idx, s)
    Q = take(P, rep)
    Q["w"] = np.repeat(w / s, s)
    Q["parent"] = Q["id"].copy()
    Q["id"] = np.arange(next_id, next_id + len(rep), dtype=np.int64)
    Q["depth"] = Q["depth"] + np.repeat(s > 1, s).astype(np.int16)
    return Q, next_id + len(rep), dict(cells=int(len(u)), split_rows=int((s > 1).sum()), killed=int((~kept).sum()),
                                       trapped=int(trapped.sum()), after=int(len(Q["w"])))


def control(P, a, rng, next_id):
    """split thin cells, roulette crowded ones (>= K_MIN), cap per batch; unbiased. Returns (P, next_id, stats)."""
    c = cells(P)
    u, inv, cnt = np.unique(c, return_inverse=True, return_counts=True)
    r = a.target / cnt[inv]
    s = np.where(r > 1, np.floor(r) + (rng.random(len(r)) < (r - np.floor(r))), 1).astype(np.int64)
    s = np.minimum(s, a.s_max)
    keep_p = np.where(r < 1, np.maximum(a.k_min, r), 1.0)
    kept = rng.random(len(r)) < keep_p
    w = P["w"] / keep_p
    idx = np.nonzero(kept)[0]
    s, w = s[idx], w[idx]
    rep = np.repeat(idx, s)
    Q = take(P, rep)
    Q["w"] = np.repeat(w / s, s)
    split = np.repeat(s > 1, s)
    Q["parent"] = Q["id"].copy()
    Q["id"] = np.arange(next_id, next_id + len(rep), dtype=np.int64)
    Q["depth"] = Q["depth"] + split.astype(np.int16)
    next_id += len(rep)
    for b in np.unique(Q["batch"]):                                    # budget: at most N_MAX per batch
        m = np.nonzero(Q["batch"] == b)[0]
        if len(m) > a.n_max:
            pk = a.n_max / len(m)
            drop = m[rng.random(len(m)) >= pk]
            Q["w"][m] /= pk
            Q["w"][drop] = -1
    keep = Q["w"] >= 0
    Q = take(Q, np.nonzero(keep)[0])
    return Q, next_id, dict(cells=int(len(u)), split_rows=int((s > 1).sum()), killed=int((~kept).sum()), after=int(len(Q["w"])))


def advance(sim, P, z, seed, chunk=40_000_000):
    """transport every row of P to z (or to where it stops); returns (alive P, stopped z, stopped w, stopped batch)"""
    torch = sim.torch
    out = {k: [] for k in ("x", "y", "z", "px", "py", "pz")}
    n = len(P["w"])
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        pos = torch.from_numpy(np.stack([P["x"][sl], P["y"][sl], P["z"][sl]], 1).astype(np.float32))
        mom = torch.from_numpy(np.stack([P["px"][sl], P["py"][sl], P["pz"][sl]], 1).astype(np.float32))
        q = torch.from_numpy(P["q"][sl].astype(np.float32))
        pos, mom = sim.propagate(pos, mom, q, sim.environment, z - G.PLANE_DZ / 2, G.N_STEPS, sim.step_length, True,
                                 seed + c0, sim.device)
        p, m = pos.numpy(), mom.numpy()
        for k, arr, j in (("x", p, 0), ("y", p, 1), ("z", p, 2), ("px", m, 0), ("py", m, 1), ("pz", m, 2)):
            out[k].append(arr[:, j].astype(np.float32))
    for k in out:
        P[k] = np.concatenate(out[k]) if out[k] else np.zeros(0, np.float32)
    alive = P["z"] > z - 0.01
    st = np.nonzero(~alive)[0]
    return take(P, np.nonzero(alive)[0]), P["z"][st].copy(), P["w"][st].copy(), P["batch"][st].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pool chunk (h5: px py pz x y z pdg weight); weights in muons per spill")
    ap.add_argument("--n-src", type=int, default=4_000_000, help="source muons (the first N rows of the chunk)")
    ap.add_argument("--batches", type=int, default=10, help="independent batches (disjoint sources, own seeds)")
    ap.add_argument("--analog", action="store_true", help="plain Monte Carlo: no splitting, no roulette")
    ap.add_argument("--target", type=float, default=200, help="splitting: target rows per control cell")
    ap.add_argument("--s-max", type=int, default=20, help="splitting: most copies of one muon at one checkpoint")
    ap.add_argument("--k-min", type=float, default=0.5, help="splitting: lowest survival probability in crowded cells")
    ap.add_argument("--n-max", type=int, default=2_000_000, help="splitting: population cap per batch")
    ap.add_argument("--mode", choices=["predicted", "tempered"], default="tempered",
                    help="tempered: current-state cells resampled to n_c^alpha (default); predicted: v1, cells of the "
                         "predicted SST position equalised to --target")
    ap.add_argument("--alpha", type=float, default=0.5, help="tempered: 1 = analog, 0 = flat")
    ap.add_argument("--random-src", action="store_true", help="a random subset of the chunk as sources (default: the first N "
                    "rows, which are NOT a random sample of a pool chunk)")
    ap.add_argument("--record-tracks", default=None, help="analog only: write every muon's state at every checkpoint (and the "
                    "source state) to this npz: tracks[k, source, (x y z px py pz)], NaN where it no longer lives")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mm-dir", default=str(G.MM_DIR))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    with h5py.File(a.input, "r") as f:
        n_chunk = len(f["px"])
    sim = G.Simulation(a.mm_dir, "TRY6", a.gpu, 0 if a.random_src else a.n_src, a.input)
    mu = sim.muons
    if a.random_src and a.n_src and a.n_src < len(mu):
        pick = np.sort(rng.choice(len(mu), a.n_src, replace=False))
        mu = mu[pick]
        sim.charge = sim.charge[sim.torch.from_numpy(pick)]
    n = len(mu)
    P = dict(x=mu[:, 3].copy(), y=mu[:, 4].copy(), z=mu[:, 5].copy(), px=mu[:, 0].copy(), py=mu[:, 1].copy(), pz=mu[:, 2].copy(),
             q=sim.charge.numpy().astype(np.int8), w=mu[:, 7].astype(np.float64) * n_chunk / n,     # the subset stands for the chunk
             root=np.arange(n, dtype=np.int64), id=np.arange(n, dtype=np.int64), parent=np.full(n, -1, np.int64),
             depth=np.zeros(n, np.int16), batch=(np.arange(n) % a.batches).astype(np.int16))
    P["w"] *= a.batches                                       # each batch alone estimates the whole spill
    next_id = n
    K, B = len(CHECKPOINTS), a.batches
    H = dict(xy_s1=np.zeros((K, B, len(XY) - 1, len(XY) - 1)), xy_s2=np.zeros((K, B, len(XY) - 1, len(XY) - 1)),
             xy_n=np.zeros((K, B, len(XY) - 1, len(XY) - 1)), p_s1=np.zeros((K, B, len(PB) - 1)), p_n=np.zeros((K, B, len(PB) - 1)),
             sx_s1=np.zeros((K, B, len(SL) - 1)), sy_s1=np.zeros((K, B, len(SL) - 1)), stop_s1=np.zeros((B, len(ZSTOP) - 1)),
             alive_w=np.zeros((K, B)), alive_n=np.zeros((K, B)), rows_in=np.zeros(K), cost=np.zeros(K))
    log = []
    z_prev = float(np.median(P["z"]))
    if a.record_tracks:
        assert a.analog, "--record-tracks needs --analog (one trajectory per source muon)"
        TR = np.full((K, n, 6), np.nan, np.float32)
        SRC = np.stack([P[c] for c in ("x", "y", "z", "px", "py", "pz")], 1).astype(np.float32)
    for k, z in enumerate(CHECKPOINTS):
        H["rows_in"][k] = len(P["w"])
        H["cost"][k] = len(P["w"]) * (z - z_prev)                          # transported row-metres: the GPU budget
        P, sz, sw, sb = advance(sim, P, z, a.seed * 1000 + k * 7919 + 1)
        if a.record_tracks:
            TR[k, P["root"]] = np.stack([P[c] for c in ("x", "y", "z", "px", "py", "pz")], 1)
        for b in range(B):
            m = sb == b
            H["stop_s1"][b] += np.histogram(sz[m], ZSTOP, weights=sw[m])[0]
            mb = P["batch"] == b
            x, y, w = P["x"][mb], P["y"][mb], P["w"][mb]
            H["xy_s1"][k, b] += np.histogram2d(x, y, [XY, XY], weights=w)[0]
            H["xy_s2"][k, b] += np.histogram2d(x, y, [XY, XY], weights=w * w)[0]
            H["xy_n"][k, b] += np.histogram2d(x, y, [XY, XY])[0]
            p = np.sqrt(P["px"][mb].astype(np.float64) ** 2 + P["py"][mb] ** 2 + P["pz"][mb] ** 2)
            H["p_s1"][k, b] += np.histogram(p, PB, weights=w)[0]
            H["p_n"][k, b] += np.histogram(p, PB)[0]
            pz = np.maximum(P["pz"][mb], 1e-6)
            H["sx_s1"][k, b] += np.histogram(P["px"][mb] / pz, SL, weights=w)[0]
            H["sy_s1"][k, b] += np.histogram(P["py"][mb] / pz, SL, weights=w)[0]
            H["alive_w"][k, b] = w.sum(); H["alive_n"][k, b] = mb.sum()
        st = {}
        if not a.analog and k < K - 1:
            if a.mode == "tempered":
                P, next_id, st = control_tempered(P, a, rng, next_id, z)
            else:
                P, next_id, st = control(P, a, rng, next_id)
        log.append(dict(z=z, rows_in=int(H["rows_in"][k]), alive=int(H["alive_n"][k].sum()), **st, s=round(time.time() - t0)))
        print(json.dumps(log[-1]), flush=True)
        z_prev = z
    np.savez(a.out, checkpoints=np.array(CHECKPOINTS), XY=XY, PB=PB, SL=SL, ZSTOP=ZSTOP, **H,
             **{f"final_{k}": v for k, v in P.items()}, n_src=n, n_chunk=n_chunk, batches=B,
             meta=json.dumps(dict(vars(a), log=log, seconds=time.time() - t0)))
    if a.record_tracks:
        np.savez(a.record_tracks, tracks=TR, source=SRC, q=sim.charge.numpy().astype(np.int8)[:n] if len(sim.charge) == n else
                 np.zeros(n, np.int8), w=mu[:, 7].astype(np.float64) * n_chunk / n, checkpoints=np.array(CHECKPOINTS))
        print(f"tracks -> {a.record_tracks}", flush=True)
    print(f"-> {a.out} ({time.time() - t0:.0f} s), final rows {len(P['w']):,}, row-metres {H['cost'].sum():.4g}")


if __name__ == "__main__":
    main()
