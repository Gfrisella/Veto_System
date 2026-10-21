"""The 506 M input muon file enriched with the Pythia 6 forced-decay tail, in the OLD file's convention.

    python3 bin/build_enriched.py [--pcut 150] [--tag tail2] [--out data/source/full_sample_after_target_enriched_p6.h5]

Old file (MuonsAndMatter full_sample_after_target.h5): production momenta of the 2018 production's muons, position
smeared (~4 cm), z = -0.0045 m (labelled as the shield entrance; no energy loss in target / absorber applied).
Recipe:
  * keep every old row except the minimum-bias rows (weight 768.75) with momentum > PCUT;
  * add our tail muons (data/tail/<tag>_*.npz: Pythia 6.4.28 first collision, Geant4 cascade, forced pi / K decays,
    Mo+W target) with production momentum > PCUT, written in the old convention: momentum = momentum at creation
    along the muon direction, x / y at the plane, z = -0.0045 m;
  * scale their weights by one flat factor so their flux above PCUT equals the old minimum-bias flux above PCUT
    (the old normalisation is kept; above PCUT the old file's 2018-plane selection keeps every muon);
  * dimuon, charm and the rest unchanged.
Datasets as the old file (px, py, pz, x, y, z, pdg, weight) + source (0 old, 1 new) + orig_index (-1 for new rows).
"""

import _bootstrap  # noqa: F401
import argparse
import glob
import time
from pathlib import Path

import h5py
import numpy as np

from veto import paths

INPUT = str(paths.MUONS_AND_MATTER / "data/muons/full_sample_after_target.h5")
COLS = ("px", "py", "pz", "x", "y", "z", "pdg", "weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcut", type=float, default=150.0)
    ap.add_argument("--tag", default="tail2")
    ap.add_argument("--out", default=str(paths.source_data() / "full_sample_after_target_enriched_p6.h5"))
    ap.add_argument("--dry-run", action="store_true", help="everything up to the normalisation, then report; nothing written")
    a = ap.parse_args()
    t0 = time.time()
    parts = [dict(np.load(f)) for f in sorted(glob.glob(str(paths.source_data() / "tail" / f"{a.tag}_[0-9]*.npz")))]
    events = sum(int(p["events"]) for p in parts)
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("px", "py", "pz", "x", "y", "pdg", "p_prod")}
    w = np.concatenate([p["w_decay"] / p["rep"] for p in parts]) * 5e13 / events
    keep = d["p_prod"] > a.pcut
    p_plane = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    scale_p = d["p_prod"] / p_plane                       # momentum at creation along the plane direction
    new = {"px": (d["px"] * scale_p)[keep], "py": (d["py"] * scale_p)[keep], "pz": (d["pz"] * scale_p)[keep],
           "x": d["x"][keep], "y": d["y"][keep], "z": np.full(keep.sum(), -0.0045), "pdg": d["pdg"][keep], "weight": w[keep]}
    with h5py.File(INPUT, "r") as f:
        n_in = len(f["px"])
        wo = f["weight"][:]
        po = np.sqrt(f["px"][:].astype(np.float64) ** 2 + f["py"][:] ** 2 + f["pz"][:] ** 2)
        drop = (np.abs(wo - 768.75) < 1) & (po > a.pcut)
        flux_old = float(wo[drop].astype(np.float64).sum())
        factor = flux_old / new["weight"].sum()
        new["weight"] = new["weight"] * factor
        keep_idx = np.nonzero(~drop)[0]
        del wo, po
        n_out = len(keep_idx) + len(new["px"])
        if a.dry_run:
            print(f"dry run: {len(parts)} tail files ({a.tag}, {events:.4g} p.o.t.), {n_in:,} old rows, {int(drop.sum()):,} dropped, "
                  f"{len(new['px']):,} added, factor {factor:.4f} -> {n_out:,} rows (nothing written), {time.time() - t0:.0f} s")
            return
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out + ".tmp"
        with h5py.File(tmp, "w") as g:
            for c in COLS:
                g.create_dataset(c, data=np.concatenate([f[c][:][keep_idx], new[c].astype(np.float32)]), dtype="f4", chunks=True)
                print(f"  {c} ({time.time() - t0:.0f} s)", flush=True)
            g.create_dataset("source", data=np.concatenate([np.zeros(len(keep_idx), np.int8), np.ones(len(new["px"]), np.int8)]))
            g.create_dataset("orig_index", data=np.concatenate([keep_idx.astype(np.int64), np.full(len(new["px"]), -1, np.int64)]))
            g.attrs["description"] = (f"MuonsAndMatter full_sample_after_target.h5, min.-bias rows above {a.pcut} GeV replaced by "
                                      f"Pythia 6.4.28 forced-decay tail muons ({a.tag}, {events:.3g} p.o.t.) in the old convention "
                                      f"(production momentum, z = -0.0045 m), weights x {factor:.4f} to the old flux above {a.pcut} GeV; "
                                      "source = 1 for those rows (Veto_System bin/build_enriched.py, ported from fast-veto-sim)")
            g.attrs["pcut"] = a.pcut
            g.attrs["normalisation_factor"] = factor
            g.attrs["rows_dropped"] = int(drop.sum())
            g.attrs["rows_added"] = int(len(new["px"]))
    Path(tmp).rename(a.out)
    print(f"{n_in:,} old rows, {int(drop.sum()):,} dropped, {len(new['px']):,} added (flux {flux_old:.4g}/spill, factor {factor:.4f}, "
          f"Kish {new['weight'].sum() ** 2 / np.sum(new['weight'] ** 2):.4g}) -> {n_out:,} rows in {a.out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
