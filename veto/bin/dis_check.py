"""Quick checks of a muon_dis_apply output: energies, library split per target, sigma table."""

import _bootstrap  # noqa: F401
import sys

import numpy as np

from fastveto import dis

z = np.load(sys.argv[1])
lib = dis.MuonLibrary(sys.argv[2])
for m in ("helium", "pvc", "mylar"):
    for E in (3, 10, 30, 100, 300):
        a, b = lib.sigma_parts(m, np.array([float(E)]))
        print(f"  sigma {m:8s} E={E:4d}: pythia {a[0]:.3f}  geant4(W<2) {b[0]:.3f} ub/nucleon")
code, E, w, comp = z["int_code"], z["int_E"], z["int_weight"], z["int_component"]
for c in np.unique(code):
    m = code == c
    print(f"code {c:3d}: n {m.sum():7d}, E q10/50/90 {np.round(np.quantile(E[m], [.1, .5, .9]), 1)}, "
          f"weighted E median {np.round(np.quantile(np.repeat(E[m], 1), .5), 1)}, pythia frac {np.mean(comp[m] == 0):.4f}")
print("types of helium interactions:", dict(zip(*np.unique(z["int_type"][code == 0], return_counts=True))))
