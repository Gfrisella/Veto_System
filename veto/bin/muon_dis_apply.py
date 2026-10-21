"""Muon interactions in the vessel helium, the air, the PVC walls and the veto planes of one design (fastveto.dis).

    bin/lcg.sh python3 bin/muon_dis_apply.py --design D6_straw [--store DIR] [--lib data/dis/mu_library.npz]
        [--helium 1e6] [--pvc 1e6] [--air 2e5] [--planes 2e5] [--max-rows N] [--out data/dis/apply_<design>.npz]

Input: the design-optimisation muon dataset (cuda-muons output, Veto_System column store: states at z = 30.6 / 82 m,
bounce point, weight per spill, evt_n, type). Output npz: one row per forced interaction (muon row, evt_n, target
code, vertex, muon energy and direction, weight = interactions per spill, library component) and the particles
of the final states (interaction index, pdg, p (GeV, lab), production point, beam = the scattered muon), plus a
json summary: interactions per spill and per 15 years per target, timing.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from veto import paths

from fastveto import dis
from fastveto import targets as tg
from veto import design as vdesign

DATASET = "design-optimization-dataset-gamma-2026-10-17"           # the muon dataset (--dataset or --store)
SPILLS_15Y = 6e20 / 5e13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D6_straw")
    ap.add_argument("--dataset", default=DATASET, help="a dataset in paths.store_dir")
    ap.add_argument("--store", default=None, help="a dataset directory (overrides --dataset)")
    ap.add_argument("--lib", default=str(paths.interactions_data() / "dis/mu_library.npz"))
    for g, v in (("helium", 1e6), ("pvc", 1e6), ("air", 2e5), ("planes", 2e5)):
        ap.add_argument(f"--{g}", type=float, default=v, help=f"draws for {g} (planes: each plane)")
    ap.add_argument("--split", default="crossover", help="crossover | w2 | geant4 (fastveto.dis.MuonLibrary)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    p = Path(a.design)
    design = vdesign.load(p if p.exists() else a.design)          # a path, or a name in designs/
    lib = dis.MuonLibrary(a.lib, split=a.split)
    t_lib = time.time() - t0
    store = Path(a.store) if a.store else paths.store_dir(a.dataset)
    meta = json.load(open(store / "meta.json"))
    cols = {k: np.load(store / f"{k}.npy", mmap_mode="r") for k in meta["columns"]}
    if a.max_rows:
        cols = {k: v[: a.max_rows] for k, v in cols.items()}
    draws = {"helium": a.helium, "pvc": a.pvc, "air": a.air, "planes": a.planes}
    inter, part, totals = dis.run(cols, design, lib, draws, np.random.default_rng(a.seed), workers=a.workers)
    t_run = time.time() - t0 - t_lib
    out = Path(a.out or paths.interactions_data() / f"dis/apply_{design.name}{'' if a.split == 'crossover' else '_' + a.split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"int_{k}": v for k, v in inter.items()}, **{f"part_{k}": v for k, v in part.items()})
    summary = {"design": design.name, "rows": int(len(cols["weight"])), "library": a.lib, "split": a.split, "draws": draws,
               "interactions_per_spill": totals, "interactions_15y": {k: v * SPILLS_15Y for k, v in totals.items()},
               "pythia_fraction": {tg.target_name(int(c), design): float(np.mean(inter["component"][inter["code"] == c] == 0))
                                   for c in np.unique(inter["code"])},
               "n_interactions": int(len(inter["E"])), "n_particles": int(len(part["pdg"])),
               "time_s": {"library": round(t_lib, 1), "run": round(t_run, 1)}, "workers": a.workers}
    json.dump(summary, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
