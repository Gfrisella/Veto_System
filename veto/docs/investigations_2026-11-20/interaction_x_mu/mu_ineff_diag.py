# Diagnostic (20 Nov 2026): partner-muon inefficiency of dis_combinatorial (1 - tagged fraction of the single SST1 muons),
# 4 ways per design: all single muons vs the ones that stop in the SBT shell removed (run_reco.shell_loss_populations),
# with random hits vs own hits only (RecoConfig.random_hits).
import sys, json, time
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import numpy as np
import _bootstrap  # noqa: F401
from run_reco import load_inputs, shell_loss_populations
from veto import design as dsg, reco

FC = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto/results_rerun/full_chain_extra/designs/"
designs = {"R1+V0": FC + "R1_ubt_only_v0.json", "R2 (ref)": W + "/designs/R2_sbt_ubt.json",
           "R2 noV0": FC + "R2_sbt_ubt_nov0.json", "Z2": FC + "Z2_opt_ga7c3_full.json"}
tr0, inp0 = load_inputs()
out = {}
for name, path in designs.items():
    t0 = time.time()
    d = dsg.load(path)
    scfg = getattr(d, "sbt", None) or {}
    res = {}
    for filt in (False, True):
        tr, inp, info = tr0, inp0, None
        if filt and scfg.get("shell_energy_loss"):
            tr, inp, info = shell_loss_populations(tr0, inp0, scfg)
        single, sw = np.asarray(inp["single"]), np.asarray(inp["single_w"], float)
        ty = tr["type"][single].astype(int)
        for rnd in (True, False):
            cfg = reco.RecoConfig(seed=1, multiple_scattering=True, random_hits=rnd)
            m = reco.run_events({k: tr[k][single] for k in tr}, np.arange(len(single)), np.full(len(single), 1e9), d, inp, cfg,
                                np.random.default_rng(3))
            v = m["vetoed"]
            key = ("stopped removed" if filt else "all") + (", random hits" if rnd else ", own hits only")
            res[key] = dict(ineff=float((sw * ~v).sum() / sw.sum()),
                            by_type={k: float((sw * ~v)[ty == k].sum() / max(sw[ty == k].sum(), 1e-300)) for k in range(3)},
                            w_by_type={k: float(sw[ty == k].sum() / sw.sum()) for k in range(3)}, n=int(len(single)))
        if info:
            res["stopped_w_frac"] = info["single_stopped_w_frac"]
            res["stopped_w_frac_by_type"] = info["single_stopped_w_frac_by_type"]
    out[name] = res
    print(name, f"{time.time() - t0:.0f} s",
          json.dumps({k: (round(v["ineff"], 4) if isinstance(v, dict) and "ineff" in v else v) for k, v in res.items()}), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
