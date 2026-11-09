#!/usr/bin/env python3
"""Muon and neutrino DIS in the last 50 cm of the muon shield (iron of MS7, z = 29.39 - 29.89 m), transported by
Geant4 out of the shield, as interactions for the selection and veto (bin/run_interactions.py).

    python3 bin/shield_dis.py prepare                                   # muons crossing the iron (host python)
    bin/g4.sh python3 bin/shield_dis.py generate --kind muon --n 2000000 --workers 90
    bin/g4.sh python3 bin/shield_dis.py generate --kind nu   --n 500000  --workers 90
    python3 bin/run_interactions.py --design D6_straw --kind nu --interactions results/shield_dis/shield_muon.npz \\
        --out results/shield_dis/D6_straw_shield_muon.json

1. Parents (veto.shield_iron). Muons: Muons_2950_50cm_before_end_MS.pkl, one full run of the old 506 M input, true
   plane z = 29.20 m; bent by the MS7 field to the vertex (shield_iron.curved_points, since 2026-11-05, D4; mean
   dE/dx 1.6 GeV/m). Neutrinos: the nu_mu / anti-nu_mu (and nu_e / anti-nu_e where a store and the iron library
   have them, NU_FLAVOURS) stores of fastveto (2018 FairShip flux, p > 10 GeV, straight lines from the target).
   Target: the iron box |x| < 1.71, |y| < 1.27 m, z = 29.39 - 29.89 m (chord length L of each line).
2. Forced interactions: P_i = n sigma(E_i) L_i, rate per spill S = sum w_i P_i. N interactions are drawn among the
   parents with probability proportional to w_i P_i (each then carries S / N), the vertex uniform along the chord.
   Final states: fastveto.dis.MuonLibrary "iron" (Geant4 muonNuclear on G4_Fe + Pythia 6 on p / n, crossover split)
   and fastveto.nudis.NuLibrary "iron" (GENIE 3.06.02 on Fe56), rotated to the parent direction.
3. Geant4 (cpp/g4shield: FTFP_BERT, the MS7 field map) carries every final-state particle through the iron to
   z = 29.90 m; everything above 0.5 GeV kinetic that gets there is kept (neutrinos dropped).
4. Forced decays in the vessel: every K_L, K_S, Lambda, Xi0 at the plane gets --v0-copies copies of its interaction
   (weight / copies each) in which it decays inside the vessel (z = geometry.Z_VESSEL_MIN .. MAX, 31.13 - 81.13 m since 2026-11-05, point from the truncated exponential), weight x P(decay there);
   the rest of the event (the scattered muon, charged debris) comes along. The original interaction (no forced
   decay) is kept with its weight. Charged particles are straight lines from the plane, OR (D4, 2026-11-05) the
   whole output goes through Geant4 from the plane / decay points (decays in flight, K_L / n interactions in air and
   helium, spectrometer field): bin/g4_vessel_transport.py --codes 3 4 --drop-v0-codes 3, then
   bin/run_interactions.py --g4 <file>.

Output (results/shield_dis/shield_<kind>.npz) in the fastveto format: int_* (code 3 = shield iron, code 4 = a copy
with a forced V0 decay: its weights sum to the V0 decays in the vessel, not to interactions; v0 = pdg of the decayed
neutral or 0, parent = the interaction it copies), part_* (beam = False: the parent hits upstream of the
shield exit are not part of the event), and shield_<kind>.json (rates per spill, what leaves the shield).
"""

import _bootstrap  # noqa: F401
import argparse
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import paths
from veto import shield_iron as si

CODE_SHIELD, CODE_SHIELD_V0 = 3, 4             # veto.interactions.CODE_NAMES
from veto.config import POT_PER_SPILL, SPILLS_15Y  # noqa: E402  (6e20 / 4e13 p.o.t.)
from veto import weights  # noqa: E402
UB = 1e-34                                    # microbarn -> m^2
V0 = {130: 15.34, 310: 0.026844, 3122: 0.0789, -3122: 0.0789, 3322: 0.0871, -3322: 0.0871}   # c tau [m]
MASS = {130: 0.497611, 310: 0.497611, 3122: 1.115683, 3322: 1.31486}
OUT = paths.results_dir() / "shield_dis"
PREPARED = paths.PROJECT_DIR / "data" / "shield_dis" / "muons_iron.npz"
NU_STORES = paths.interactions_data() / "neutrinos2026"      # Shah Rukh 2026-11-09 flux (bin/nu_prepare.py); 2018 flux discarded
MU_SPLIT = "pythia"                           # D3 (Guglielmo 2026-11-05): Pythia 6 part only of the iron library
BEND_PARENT = True                            # D4: parent muon bent by the MS7 field (si.curved_points)
NU_FLAVOURS = (14, -14, 12, -12, 16, -16)     # D4: every flavour that has a store and an iron library entry
MU_LIB = paths.PROJECT_DIR / "data/interactions/dis/mu_library_iron.npz"
NU_LIB = paths.interactions_data() / "genie/nu_library_v2.npz"   # GENIE v2 (Shah Rukh 2026-11-09): Fe56, all 6 flavours
NU_STORE_PREFIX = NU_STORES / "store"         # stores <prefix>_nu<pdg> (bin/nu_prepare.py)


# ------------------------------------------------------------------ 1. parents
def prepare(a):
    """Muons of the pickle whose straight line crosses the iron target: state at the true plane, chord."""
    t0 = time.time()
    d = pickle.load(open(si.MUON_FILE, "rb"))
    pl = d["planes"].item() if hasattr(d["planes"], "item") else d["planes"]
    pl = pl[0] if isinstance(pl, (dict, list)) and 0 in (pl if isinstance(pl, dict) else range(len(pl))) else pl
    x, y, z = (np.asarray(pl[k], np.float64) for k in ("x", "y", "z"))
    px, py, pz = (np.asarray(pl[k], np.float64) for k in ("px", "py", "pz"))
    z = z + si.MUON_Z_SHIFT
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    L, s_in, s_out = si.iron_chord(x, y, z, px / p, py / p, pz / p)
    w = weights.to_spill(np.asarray(d["weight"], np.float64), weights.source_pot(path=si.MUON_FILE))   # D13: per 4e13
    keep = (L > 0) & (pz > 0) & (w > 0)
    print(f"{len(x):,} muons ({w.sum():.4g} per spill), {keep.sum():,} cross the iron ({w[keep].sum():.4g} per spill), "
          f"z plane {np.median(z):.3f} m, {time.time() - t0:.0f} s")
    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PREPARED, x=x[keep], y=y[keep], z=z[keep], px=px[keep], py=py[keep], pz=pz[keep],
             q=np.asarray(d["pdg"])[keep].astype(np.int8), weight=w[keep], evt_n=np.asarray(d["evt_n"])[keep],
             L=L[keep], s_in=s_in[keep], s_out=s_out[keep],
             meta=np.array(json.dumps(dict(source=str(si.MUON_FILE), z_shift=si.MUON_Z_SHIFT, n_all=len(x),
                                           w_all=float(w.sum()), iron=[si.Z_IRON0, si.Z_SHIELD_END, si.HALF_X, si.HALF_Y],
                                           pot_per_spill=POT_PER_SPILL))))
    print(f"wrote {PREPARED}")


def parents(kind):
    """Lines through the iron: dict(x, y, z, u (n,3), p, L, s_in, s_out, weight, evt_n, pdg)."""
    if kind == "muon":
        f = np.load(PREPARED)
        fs = POT_PER_SPILL / weights.source_pot(json.loads(str(f["meta"])), PREPARED)   # D13: legacy file 5e13 -> 4e13
        pv = np.stack([f["px"], f["py"], f["pz"]], 1)
        p = np.linalg.norm(pv, axis=1)
        return dict(x=f["x"], y=f["y"], z=f["z"], u=pv / p[:, None], p=p, L=f["L"], s_in=f["s_in"], s_out=f["s_out"],
                    weight=f["weight"] * fs, evt_n=f["evt_n"], pdg=-13 * f["q"].astype(np.int64))
    out = []
    have = set(np.load(NU_LIB)["nus"].tolist()) if Path(NU_LIB).exists() else {14, -14}
    for fl in NU_FLAVOURS:
        st = Path(f"{NU_STORE_PREFIX}_nu{fl}")
        if fl not in have or not st.exists():
            print(f"  nu {fl}: {'no library entry' if fl not in have else 'no flux store'} ({st}), skipped "
                  "(nu_e: rebuild the iron library with bin/nu_library.py --nuclei 1000260560 --nus 14 -14 12 -12)")
            continue
        c = {k: np.load(st / f"{k}.npy") for k in ("x_30", "y_30", "z_30", "px_30", "py_30", "pz_30", "weight", "evt_n")}
        pv = np.stack([c["px_30"], c["py_30"], c["pz_30"]], 1).astype(np.float64)
        p = np.linalg.norm(pv, axis=1)
        u = pv / p[:, None]
        L, s_in, s_out = si.iron_chord(c["x_30"], c["y_30"], c["z_30"], u[:, 0], u[:, 1], u[:, 2])
        k = (L > 0) & (u[:, 2] > 0)
        out.append(dict(x=c["x_30"][k].astype(np.float64), y=c["y_30"][k].astype(np.float64),
                        z=c["z_30"][k].astype(np.float64), u=u[k], p=p[k], L=L[k], s_in=s_in[k], s_out=s_out[k],
                        weight=c["weight"][k].astype(np.float64), evt_n=c["evt_n"][k].astype(np.int64),
                        pdg=np.full(k.sum(), fl)))
    return {key: np.concatenate([o[key] for o in out]) for key in out[0]}


# ------------------------------------------------------------------ 2. forced interactions
def draw(kind, par, n, rng, per_track=False, mu_lib=None, mu_split=None):
    """N interactions among the parents, probability proportional to w P (each S / N); per_track: one interaction on
    every parent instead, weight w_i P_i (every track contributes once). Returns (inter, part, summary)."""
    from fastveto import dis, nudis
    if kind == "muon":
        lib = dis.MuonLibrary(str(mu_lib or MU_LIB), split=mu_split or MU_SPLIT)
        E_mid = par["p"] - si.DEDX_MU * 0.5 * (par["s_in"] + par["s_out"])
        sig = np.where(E_mid > 0.5, lib.sigma("iron", np.maximum(E_mid, 0.5)), 0.0)        # microbarn per nucleon
        P = si.nucleons_per_m3() * sig * UB * par["L"]
        libs = {None: lib}                     # one library for mu+ and mu- (sign from the parent)
    else:
        libs = {fl: nudis.NuLibrary(str(NU_LIB), flavour=fl) for fl in np.unique(par["pdg"]).tolist()}
        P = np.zeros(len(par["p"]))
        for fl, lb in libs.items():
            m = par["pdg"] == fl
            P[m] = si.nucleons_per_m3() * lb.sigma("iron", par["p"][m]) * UB * par["L"][m]
    wP = par["weight"] * P
    S = float(wP.sum())
    if per_track:
        idx = np.nonzero(wP > 0)[0]
        n = len(idx)
        w_int = wP[idx]
    else:
        cdf = np.cumsum(wP)
        idx = np.minimum(np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right"), len(cdf) - 1)
        w_int = np.full(n, S / n)
    s = rng.uniform(par["s_in"][idx], par["s_out"][idx])
    u = par["u"][idx]
    start = np.stack([par["x"][idx], par["y"][idx], par["z"][idx]], 1)
    if kind == "muon" and BEND_PARENT:
        # D4 (2026-11-05): the muon bends in the MS7 field on its way to the vertex (chord / path length still from the
        # straight line: ~1 cm at 20 GeV, a negligible change of L)
        vtx, u, _ = si.curved_points(start, u, par["p"][idx], -np.sign(par["pdg"][idx]), s)
    else:
        vtx = start + u * s[:, None]
    E = par["p"][idx] - (si.DEDX_MU * s if kind == "muon" else 0.0)
    ok = E > 0.5
    inter = dict(parent_row=idx, evt_n=par["evt_n"][idx], pdg_parent=par["pdg"][idx], code=np.full(n, CODE_SHIELD, np.int8),
                 x=vtx[:, 0], y=vtx[:, 1], z=vtx[:, 2], E=E, ux=u[:, 0], uy=u[:, 1], uz=u[:, 2],
                 weight=np.where(ok, w_int, 0.0), p_int=P[idx], w_parent=par["weight"][idx], muon=np.full(n, -1, np.int64), type=np.zeros(n, np.int8))
    parts = []
    inter["component"] = np.zeros(n, np.int8)
    inter["E_lib"] = np.zeros(n)
    for code, lb in libs.items():
        m = np.nonzero(ok & ((inter["pdg_parent"] == code) if code is not None else True))[0]
        if not len(m):
            continue
        o = lb.sample("iron", E[m], u[m], vtx[m], inter["pdg_parent"][m], rng)
        o["int"] = m[o["int"]]
        inter["component"][m] = o["component"]
        inter["E_lib"][m] = o["E_lib"]
        parts.append({k: o[k] for k in ("int", "pdg", "p", "v", "beam")})
    part = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    order = np.argsort(part["int"], kind="stable")
    part = {k: v[order] for k, v in part.items()}
    summ = dict(parents=len(P), parents_per_spill=float(par["weight"].sum()),
                interactions_per_spill=S, pot_per_spill=POT_PER_SPILL, interactions_15y=S * SPILLS_15Y, draws=n, dropped_low_E=int((~ok).sum()),
                sampling="one per track, weight w P" if per_track else "proportional to w P, weight S / N")
    return inter, part, summ


# ------------------------------------------------------------------ 3. Geant4 transport
_G4 = None


def _worker_init(seed, natural=None):
    import os
    global _G4
    _G4 = si.init_g4shield(seed=seed * 1000003 + os.getpid(), natural=natural)   # a different stream in every worker


def _transport_chunk(args):
    ev0, off, pdg, p, v = args
    r = _G4.transport(off, pdg, p[:, 0], p[:, 1], p[:, 2], v[:, 0], v[:, 1], v[:, 2])
    r = {k: np.asarray(x) for k, x in r.items()}
    r["ev"] = r["ev"] + ev0
    return r


def transport(part, n_inter, workers, chunk=2000, seed=1, natural=None):
    """Every final state through the iron with Geant4, in chunks of interactions over forked workers."""
    starts = np.searchsorted(part["int"], np.arange(n_inter + 1))
    jobs = []
    for e0 in range(0, n_inter, chunk):
        e1 = min(n_inter, e0 + chunk)
        a, b = starts[e0], starts[e1]
        jobs.append((e0, (starts[e0:e1 + 1] - a).astype(np.int64), part["pdg"][a:b].astype(np.int32),
                     part["p"][a:b, :3].astype(np.float64), part["v"][a:b].astype(np.float64)))
    ctx = mp.get_context("spawn")
    res = []
    with ctx.Pool(workers, initializer=_worker_init, initargs=(seed, natural)) as pool:
        for i, r in enumerate(pool.imap_unordered(_transport_chunk, jobs)):
            res.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  transport {i + 1}/{len(jobs)} chunks", flush=True)
    out = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    order = np.argsort(out["ev"], kind="stable")
    return {k: v[order] for k, v in out.items()}


# ------------------------------------------------------------------ 4. forced decays in the vessel
def plane_particles(g):
    """Geant4 output -> particles at the plane in the fastveto format (v = the point at the plane)."""
    from fastveto import decay
    p3 = np.stack([g["px"], g["py"], g["pz"]], 1)
    masses = {**decay.MASS, 13: 0.105658, 321: 0.493677, 11: 0.000511, 130: 0.497611}
    codes, inv = np.unique(np.abs(g["pdg"]), return_inverse=True)
    M = np.array([masses.get(int(c), 0.0) for c in codes])[inv]
    E = np.sqrt((p3 ** 2).sum(1) + M ** 2)
    v = np.stack([g["x"], g["y"], np.full(len(g["x"]), si.Z_PLANE)], 1)
    return dict(int=g["ev"].astype(np.int64), pdg=g["pdg"].astype(np.int64), p=np.concatenate([p3, E[:, None]], 1), v=v,
                beam=np.zeros(len(v), bool))


def force_v0(inter, part, rng, copies=1):
    """`copies` extra interactions per neutral long-lived particle at the plane (weight / copies each): it decays in
    the vessel, at an independent point and in an independent channel in every copy."""
    from fastveto import decay
    n_inter = len(inter["weight"])
    rows = np.nonzero(np.isin(part["pdg"], list(V0)))[0]
    rows = np.repeat(rows[part["p"][rows, 2] > 0], copies)
    if not len(rows):
        return inter, part
    pdg = part["pdg"][rows]
    p3 = part["p"][rows, :3]
    pm = np.linalg.norm(p3, axis=1)
    u = p3 / pm[:, None]
    lam = pm / np.array([MASS[abs(int(c))] for c in pdg]) * np.array([V0[int(c)] for c in pdg])
    d1 = (geo.Z_VESSEL_MIN - si.Z_PLANE) / u[:, 2]
    d2 = (geo.Z_VESSEL_MAX - si.Z_PLANE) / u[:, 2]
    e1, e2 = np.exp(-d1 / lam), np.exp(-d2 / lam)
    Pdec = e1 - e2
    good = Pdec > 0
    rows, pdg, p3, pm, u, lam, d1, e1, e2, Pdec = (x[good] for x in (rows, pdg, p3, pm, u, lam, d1, e1, e2, Pdec))
    d = -lam * np.log(e1 - rng.random(len(rows)) * (e1 - e2))                     # truncated exponential
    vdec = part["v"][rows] + u * d[:, None]
    k = len(rows)
    new_id = n_inter + np.arange(k)
    ev_d, pdg_d, p_d, v_d = decay.force_decay(new_id, pdg, part["p"][rows], vdec, rng)
    # the rest of each copied event: every other particle of the interaction
    src = part["int"][rows]
    starts = np.searchsorted(part["int"], np.arange(n_inter + 1))
    cnt = starts[src + 1] - starts[src]
    rest_rows, owner = (np.repeat(starts[src], cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt),
                        np.repeat(np.arange(k), cnt))
    keep = rest_rows != rows[owner]
    rest_rows, owner = rest_rows[keep], owner[keep]
    add = dict(int=np.concatenate([new_id[owner], ev_d]), pdg=np.concatenate([part["pdg"][rest_rows], pdg_d]),
               p=np.concatenate([part["p"][rest_rows], p_d]), v=np.concatenate([part["v"][rest_rows], v_d]),
               beam=np.zeros(len(rest_rows) + len(ev_d), bool))
    new_inter = {kk: np.concatenate([vv, vv[src]]) for kk, vv in inter.items()}
    new_inter["weight"][n_inter:] = inter["weight"][src] * Pdec / copies
    new_inter["code"][n_inter:] = CODE_SHIELD_V0
    new_inter["v0"] = np.concatenate([np.zeros(n_inter, np.int64), pdg])
    new_inter["parent"] = np.concatenate([np.arange(n_inter), src])
    new_inter["z_decay"] = np.concatenate([np.full(n_inter, np.nan), vdec[:, 2]])
    new_part = {kk: np.concatenate([part[kk], add[kk]]) for kk in part}
    order = np.argsort(new_part["int"], kind="stable")
    return new_inter, {kk: vv[order] for kk, vv in new_part.items()}


def exit_summary(inter, part):
    """What leaves the shield per spill (kHz at 4e13 PoT/s = per spill / 1e3), by species, p > 1 GeV."""
    w = inter["weight"][part["int"]]
    pm = np.linalg.norm(part["p"][:, :3], axis=1)
    out = {}
    for name, codes in (("mu", [13]), ("pi+-", [211]), ("K+-", [321]), ("p", [2212]), ("n", [2112]), ("K_L", [130]),
                        ("K_S", [310]), ("Lambda", [3122]), ("gamma", [22]), ("e+-", [11])):
        m = np.isin(np.abs(part["pdg"]), codes) & (pm > 1)
        out[name] = dict(per_spill=float(w[m].sum()), n=int(m.sum()))
    return out


def generate(a):
    global NU_LIB, NU_STORE_PREFIX
    NU_LIB = Path(a.nu_lib) if a.nu_lib else NU_LIB
    NU_STORE_PREFIX = Path(a.nu_stores) if a.nu_stores else NU_STORE_PREFIX
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    par = parents(a.kind)
    print(f"{a.kind}: {len(par['p']):,} parents cross the iron ({par['weight'].sum():.4g} per spill), {time.time() - t0:.0f} s",
          flush=True)
    inter, part, summ = draw(a.kind, par, a.n, rng, a.per_track, a.mu_lib, a.mu_split)
    summ["mu_library"] = str(a.mu_lib or MU_LIB)
    summ["mu_split"] = a.mu_split or MU_SPLIT
    print(f"  {summ['interactions_per_spill']:.4g} interactions per spill ({summ['interactions_15y']:.4g} in 15 years); "
          f"{len(part['pdg']):,} final-state particles, {time.time() - t0:.0f} s", flush=True)
    g = transport(part, len(inter["weight"]), a.workers, a.chunk, a.seed)
    plane = plane_particles(g)
    plane["primary"] = g["primary"].astype(bool)
    print(f"  Geant4: {len(plane['pdg']):,} particles at z = {si.Z_PLANE:.2f} m, {time.time() - t0:.0f} s", flush=True)
    summ["exit"] = exit_summary(inter, plane)
    plane.pop("primary")
    inter2, part2 = force_v0(inter, plane, rng, a.v0_copies)
    v0 = inter2["v0"] != 0
    summ["v0_copies"] = int(v0.sum())
    summ["copies_per_v0"] = a.v0_copies
    summ["v0_decays_in_vessel_per_spill"] = {str(c): float(inter2["weight"][inter2["v0"] == c].sum())
                                             for c in np.unique(inter2["v0"][v0])}
    summ["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"shield_{a.kind}{a.tag}.npz"
    np.savez(out, **{f"int_{k}": v for k, v in inter2.items()}, **{f"part_{k}": v for k, v in part2.items()})
    json.dump(summ, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    g = sub.add_parser("generate")
    g.add_argument("--kind", choices=["muon", "nu"], required=True)
    g.add_argument("--n", type=int, default=200000, help="forced interactions")
    g.add_argument("--workers", type=int, default=64)
    g.add_argument("--chunk", type=int, default=2000, help="interactions per Geant4 job")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--tag", default="")
    g.add_argument("--per-track", action="store_true", help="one interaction per parent track, weight w P (--n ignored)")
    g.add_argument("--mu-split", choices=["pythia", "geant4", "crossover", "w2"], default=None,
                   help=f"which generator of the muon library (default {MU_SPLIT!r}, D3); geant4 = Geant4 muonNuclear only")
    g.add_argument("--nu-lib", default=None, help=f"neutrino library (default {NU_LIB})")
    g.add_argument("--nu-stores", default=None, help=f"neutrino flux store prefix, <prefix>_nu<pdg> (default {NU_STORE_PREFIX})")
    g.add_argument("--v0-copies", type=int, default=1, help="forced vessel decays per V0 at the plane (weight / copies)")
    g.add_argument("--mu-lib", help="muon DIS library (default data/interactions/dis/mu_library_iron.npz; "
                   "mu_library_iron_v0stable.npz keeps K_S, Lambda for Geant4 to transport and decay)")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else generate(a)


if __name__ == "__main__":
    main()
