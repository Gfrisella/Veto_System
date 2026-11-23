#!/usr/bin/env python3
"""FairShip spectrometer field map (ROOT, trees Data(x, y, z [cm], Bx, By, Bz [T]) + Range) -> our .h5 (points [m], B [T]),
FairShip polarity kept (Guglielmo 23 Nov 2026: use FairShip 2026_07_02_MainSpectrometerField_V21_2455, as in the
TRY_2026_EM_noSBT_full production). The map's local z is centred on 0; spectrometer.field_map() re-centres it at
geometry.Z_MAGNET (FairShip places it at Bfield.z, the magnet centre).
Two steps (ROOT lives in the FairShip env, h5py on the host):
  bin/fairship_env.sh python3 bin/convert_fairship_field.py read  <map.root> <tmp.npz>
  python3 bin/convert_fairship_field.py write <tmp.npz> <out.h5>
"""
import sys

import numpy as np


def read(root_file, out_npz):
    import ROOT
    d = ROOT.RDataFrame("Data", root_file).AsNumpy(["x", "y", "z", "Bx", "By", "Bz"])
    pts = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64) / 100.0        # cm -> m
    B = np.stack([d["Bx"], d["By"], d["Bz"]], 1).astype(np.float64)                # T
    np.savez_compressed(out_npz, points=pts, B=B, source=root_file)
    print(f"{len(pts)} points, x {pts[:, 0].min():.2f}..{pts[:, 0].max():.2f}, y {pts[:, 1].min():.2f}..{pts[:, 1].max():.2f}, "
          f"z {pts[:, 2].min():.2f}..{pts[:, 2].max():.2f} m, max |B| {np.linalg.norm(B, axis=1).max():.3f} T")


def write(npz, out_h5):
    import h5py
    f = np.load(npz)
    pts, B = f["points"], f["B"]
    o = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))                              # x-major, as the old .h5
    with h5py.File(out_h5, "w") as h:
        h["points"] = pts[o]
        h["B"] = B[o]
        h.attrs["source"] = str(f["source"])
        h.attrs["polarity"] = "FairShip (as in the ROOT map)"
    print(f"wrote {out_h5}")


if __name__ == "__main__":
    {"read": read, "write": write}[sys.argv[1]](*sys.argv[2:])
