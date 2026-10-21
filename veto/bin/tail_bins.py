"""Per-bin ratio of tail files to the input min.-bias class: python3 bin/tail_bins.py FILE [FILE...] (tail_*.npz or tail_combined.npz)."""

import _bootstrap  # noqa: F401
import sys

import h5py
import numpy as np

from veto import paths

with h5py.File(str(paths.MUONS_AND_MATTER / "data/muons/full_sample_after_target.h5"), "r") as f:
    wi = f["weight"][:]
    m = np.abs(wi - 768.75) < 1
    pi = np.sqrt(sum(f[k][:][m].astype(np.float64) ** 2 for k in ("px", "py", "pz")))
b = np.array([20, 45, 80, 130, 150, 175, 200, 225, 250, 275, 300, 350, 400])
hi, _ = np.histogram(pi, b)
for fn in [a for a in sys.argv[1:] if not a.startswith("--")]:
    d = np.load(fn)
    w = d["w"] if "w" in d.files else d["w_decay"] / d["rep"] * 5e13 / d["events"]
    p = d["p"] if "p" in d.files else np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    if len(sys.argv) > 2 and sys.argv[-1] == "--create":
        p = d["p_prod"]
    ho, _ = np.histogram(p, b, weights=w)
    h2, _ = np.histogram(p, b, weights=w ** 2)
    print(fn)
    for i in range(len(b) - 1):
        if hi[i] == 0 and ho[i] == 0:
            continue
        r = ho[i] / max(768.75 * hi[i], 1e-30)
        e = r * np.hypot(1 / np.sqrt(max(hi[i], 1)), np.sqrt(h2[i]) / max(ho[i], 1e-30))
        print(f"  {b[i]:3d}-{b[i + 1]:3d} GeV: ratio {r:5.3f} +- {e:.3f}   (input {hi[i]:8d} muons)")
