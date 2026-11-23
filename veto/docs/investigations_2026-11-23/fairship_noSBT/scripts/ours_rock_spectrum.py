#!/usr/bin/env python3
"""Dump our cavern-rock tracks (all charged downstream, SST1-acceptance flag, good flag) for the FairShip comparison.
Read-only use of the veto library (worktree nov4-source) via cavern_rock/rock_study.rock_all. Weights per spill of 4e13.
    cd <veto checkout>; export VETO_THREADS=8; source bin/batch/env.sh; python3 ours_rock_spectrum.py
"""
import os, sys
import numpy as np
C = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto/docs/investigations_2026-11-20/cavern_rock"
sys.path.insert(0, C)
from rock_study import rock_all  # noqa: E402
R, inter, part = rock_all("/disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz", 300)
p = np.asarray(R["p"]); p = np.linalg.norm(p, axis=1) if p.ndim == 2 else p
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ours_rock_tracks.npz")
np.savez_compressed(out, pdg=R["pdg"], q=R["q"], p=p, w=R["w"], acc=R["acc"], good=R["good"], v=R["v"])
for lab, s in (("all charged downstream", np.ones(len(p), bool)), ("SST1 acc", R["acc"]), ("good", R["good"])):
    print(f"{lab}: {R['w'][s].sum():.4g} / spill ({s.sum()} rows), median p {np.median(p[s]):.2f}")
