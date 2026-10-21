"""Ratio ours / h5 (min. bias) using the muon momentum at the plane vs at its creation (no energy loss in the target)."""

import _bootstrap  # noqa: F401
import glob

import h5py
import numpy as np

from veto import paths


def load(tag):
    ps = [dict(np.load(f)) for f in sorted(glob.glob(f"data/tail/{tag}_[0-9]*.npz"))]
    ev = sum(int(p["events"]) for p in ps)
    d = {k: np.concatenate([p[k] for p in ps]) for k in ("px", "py", "pz", "p_prod")}
    d["w"] = np.concatenate([p["w_decay"] / p["rep"] for p in ps]) * 5e13 / ev
    d["p"] = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    return d


lo, hi = load("mb2"), load("tail2")
with h5py.File(str(paths.MUONS_AND_MATTER / "data/muons/full_sample_after_target.h5"), "r") as f:
    w = f["weight"][:]
    m = np.abs(w - 768.75) < 1
    po = np.sqrt(sum(f[k][:][m].astype(np.float64) ** 2 for k in ("px", "py", "pz")))
b = np.array([20, 30, 45, 60, 80, 100, 130, 160, 200, 250, 300, 350])
ho, _ = np.histogram(po, b, weights=np.full(len(po), 768.75))
for key, lab in (("p", "at the plane"), ("p_prod", "at creation")):
    a = np.concatenate([lo[key][lo[key] < 150], hi[key][hi[key] >= 150]])
    ww = np.concatenate([lo["w"][lo[key] < 150], hi["w"][hi[key] >= 150]])
    h, _ = np.histogram(a, b, weights=ww)
    print(f"{lab:13s}: " + "  ".join(f"{b[i]}-{b[i + 1]}: {h[i] / ho[i]:.3f}" for i in range(len(b) - 1)))
