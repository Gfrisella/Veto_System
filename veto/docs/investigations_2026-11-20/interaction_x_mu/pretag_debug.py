# Debug (20 Nov 2026): are the legs of the pairs that survive the full pair veto tagged when traced back alone (pretag)?
import sys
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import numpy as np
import _bootstrap  # noqa: F401
import dis_combinatorial as dc
from rock_combinatorial import KEYS, cross_pairs, muon_tracks, veto_term
from run_reco import load_inputs, shell_loss_populations
from veto import design as dsg, reco, paths, sbt_material

R = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto/results_rerun/"
design = dsg.load(R + "full_chain_extra/designs/Z2_opt_ga7c3_full.json")
cfg = reco.RecoConfig(seed=1, multiple_scattering=True, random_hits=False)
tr_in, inp = load_inputs()
mu = muon_tracks(paths.MUON_DATASET, 20000)
scfg = design.sbt
st = sbt_material.shell_energy_loss(mu, True)["stopped"]
mu = {k: v[~st] for k, v in mu.items()}
tr_in, inp, _ = shell_loss_populations(tr_in, inp, scfg)
B, info, comp = dc.dis_tracks(R + "evstore/ms7_muon_v0s")
ia, ib, v, n_pairs, pre = cross_pairs(mu, B, "cuda", 4096, sel=cfg)
tr = {k: np.concatenate([np.asarray(mu[k], float), np.asarray(B[k], float)]) for k in KEYS}
tr["draw"] = np.concatenate([np.full(len(mu["ax"]), -1, np.int64), np.asarray(B["draw"], np.int64)])
cand = np.stack([ia, ib + len(mu["ax"])], 1)
w = mu["weight"][ia] * B["weight"][ib]
rk = veto_term(tr, cand, w, v, design, inp, cfg, np.random.default_rng(2), pre=pre, companions=comp, with_v0=True)
s = rk["_surv"]
print("full: after", rk["after_15y"], "n", rk["n_after"])
mrows = np.unique(s["rows"][:, 0]); drows = np.unique(s["rows"][:, 1]) - len(mu["ax"])
tm = dc.pretag({k: mu[k][mrows] for k in list(KEYS) + ["draw"]}, design, inp, cfg, np.random.default_rng(11))
td = dc.pretag({k: B[k][drows] for k in list(KEYS) + ["draw"]}, design, inp, cfg, np.random.default_rng(13), comp=comp)
print(f"surviving pairs use {len(mrows)} muons: {tm.sum()} tagged alone; {len(drows)} DIS tracks: {td.sum()} tagged alone")
# per tagged-alone muon: which part tags it? rerun with details
d = reco.run_events({k: mu[k][mrows] for k in list(KEYS) + ["draw"]}, np.arange(len(mrows)), np.full(len(mrows), dc.Z_PRETAG),
                    design, inp, cfg, np.random.default_rng(11), details=True)
print("alone, vetoed:", d["vetoed"].sum(), "by planes", d["by_plane"], "sbt_only", d["sbt_only"], "v0_only", d["v0_only"])
print("types of those muons:", np.unique(mu["type"][mrows][d["vetoed"]], return_counts=True))
# same muons, z_limit = their pair vertex
zv = {}
for (m_, _), vv in zip(s["rows"], s["vtx"]):
    zv.setdefault(m_, vv[2])
zl = np.array([zv[m_] for m_ in mrows])
d2 = reco.run_events({k: mu[k][mrows] for k in list(KEYS) + ["draw"]}, np.arange(len(mrows)), zl, design, inp, cfg,
                     np.random.default_rng(11), details=True)
print("alone with z_limit = pair vertex, vetoed:", d2["vetoed"].sum(), "sbt_only", d2["sbt_only"])
print("pair vertex z range", zl.min(), zl.max())
# variants on the surviving pairs
rows = s["rows"]; n = len(rows)
ev = np.repeat(np.arange(n), 2); zl2 = np.repeat(s["vtx"][:, 2], 2)
T = {k: v[rows.ravel()] for k, v in tr.items()}
pv = np.stack([T["bpx"], T["bpy"], T["bpz"]], 1).reshape(-1, 2, 3).sum(1)
cmp_ = dc.companions_of(comp, T["draw"], ev)
for name, kw in [("pair, comp+v0", dict(v0=dict(vtx=s["vtx"], pvis=pv), companions=cmp_)), ("pair, v0 only", dict(v0=dict(vtx=s["vtx"], pvis=pv))),
                 ("pair, comp only", dict(companions=cmp_)), ("pair, plain", {})]:
    r = reco.run_events(T, ev, zl2, design, inp, cfg, np.random.default_rng(5), **kw)
    print(name, "vetoed", r["vetoed"].sum(), "of", n, "explained legs", r["explained"].reshape(-1, 2).sum(0))
# muon leg alone but same event numbering as a pair (one track per event)
r = reco.run_events({k: v[rows[:, 0]] for k, v in tr.items()}, np.arange(n), s["vtx"][:, 2], design, inp, cfg, np.random.default_rng(5))
print("muon legs alone (tr dict of the pair):", r["vetoed"].sum(), "of", n)
r = reco.run_events({k: np.asarray(mu[k])[rows[:, 0]] for k in list(KEYS) + ["draw"]}, np.arange(n), s["vtx"][:, 2], design, inp, cfg, np.random.default_rng(5))
print("muon legs alone (mu dict):", r["vetoed"].sum(), "of", n)
print("dtypes mu:", {k: np.asarray(mu[k]).dtype for k in ("ax", "kz", "type", "pdg")}, "tr:", {k: tr[k].dtype for k in ("ax", "kz", "type", "pdg")})
