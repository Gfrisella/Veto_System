#!/usr/bin/env python3
"""K_L made by the forced muon DIS in the last 50 cm of MS7 (library final state, before transport) vs the K_L at the
plane behind the shield after Geant4 (results/shield_dis/shield_muon.json): is the iron attenuation applied?

    python3 bin/shield_kl_check.py [--n 300000]
"""
import _bootstrap  # noqa: F401
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("sd", Path(__file__).with_name("shield_dis.py"))
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300000)
a = ap.parse_args()
inter, part, _ = sd.draw("muon", sd.parents("muon"), a.n, np.random.default_rng(3))
w = inter["weight"][part["int"]]
pm = np.linalg.norm(part["p"][:, :3], axis=1)
kl = (part["pdg"] == 130) & (pm > 1)
d = (sd.si.Z_SHIELD_END - part["v"][kl, 2]) / (part["p"][kl, 2] / pm[kl])     # iron left along the K_L
print(f"K_L made (p > 1 GeV): {w[kl].sum():.4g} per spill ({kl.sum()} MC), mean iron left {np.mean(d) * 100:.1f} cm")
for lam in (0.17, 0.18):
    print(f"  x exp(-d / {lam * 100:.0f} cm): {(w[kl] * np.exp(-d / lam)).sum():.4g} per spill")
ex = json.load(open(sd.OUT / "shield_muon.json"))["exit"]["K_L"]
print(f"K_L at the plane after Geant4 (shield_muon.json): {ex['per_spill']:.4g} per spill ({ex['n']} MC)")
