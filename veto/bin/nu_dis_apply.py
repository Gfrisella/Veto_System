"""Neutrino interactions in the vessel helium, the air, the PVC walls and the veto planes of one design (GENIE library).

    bin/lcg.sh python3 bin/nu_dis_apply.py --design D6_straw [--lib data/genie/nu_library.npz]
        [--helium 5e5] [--pvc 5e5] [--air 2e5] [--planes 1e5] [--flavours 14,-14] [--out data/genie/apply_<design>.npz]

Per flavour: the flux store (bin/nu_prepare.py: straight lines at z = 30.6 / 82 m, weight per spill) through
fastveto.dis.run with fastveto.nudis.NuLibrary (forced interactions per target group, final states from GENIE).
Output npz: interactions (int_*, with flavour, nucleus, process bits, charm, CC/NC as component 0/1) and particles
(part_*; part_int indexes the interactions), and a json summary: interactions per spill / 15 years per target and
flavour, CC fraction, timing.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from veto import paths

from fastveto import dis, nudis
from fastveto import targets as tg
from veto import design as vdesign

SPILLS_15Y = 6e20 / 5e13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D6_straw")
    ap.add_argument("--lib", default=str(paths.interactions_data() / "genie/nu_library.npz"))
    ap.add_argument("--store", default=str(paths.interactions_data() / "neutrinos/store"))
    ap.add_argument("--flavours", default="14,-14")
    for g, v in (("helium", 5e5), ("pvc", 5e5), ("air", 2e5), ("planes", 1e5)):
        ap.add_argument(f"--{g}", type=float, default=v, help=f"draws per flavour for {g} (planes: each plane)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    p = Path(a.design)
    design = vdesign.load(p if p.exists() else a.design)          # a path, or a name in designs/
    draws = {"helium": a.helium, "pvc": a.pvc, "air": a.air, "planes": a.planes}
    rng = np.random.default_rng(a.seed)
    I, P, summary = [], [], {"design": design.name, "library": a.lib, "draws_per_flavour": draws, "flavours": {}}
    n_int = 0
    for fl in [int(x) for x in a.flavours.split(",")]:
        store = Path(f"{a.store}_nu{fl}")
        meta = json.load(open(store / "meta.json"))
        cols = {k: np.load(store / f"{k}.npy", mmap_mode="r") for k in meta["columns"]}
        lib = nudis.NuLibrary(a.lib, flavour=fl)
        inter, part, totals = dis.run(cols, design, lib, draws, rng, workers=a.workers)
        inter["flavour"] = np.full(len(inter["E"]), fl, np.int32)
        part["int"] = part["int"] + n_int
        n_int += len(inter["E"])
        I.append(inter); P.append(part)
        cc = {tg.target_name(int(c), design): float(np.mean(inter["component"][inter["code"] == c] == 0))
              for c in np.unique(inter["code"])}
        summary["flavours"][str(fl)] = {"rows": int(meta["n"]), "interactions_per_spill": totals,
                                        "interactions_15y": {k: v * SPILLS_15Y for k, v in totals.items()}, "cc_fraction": cc}
        print(f"nu {fl}: " + ", ".join(f"{k} {v * SPILLS_15Y:.3g}" for k, v in totals.items()) + " per 15 years", flush=True)
    inter = {k: np.concatenate([x[k] for x in I]) for k in I[0]}
    part = {k: np.concatenate([x[k] for x in P]) for k in P[0]}
    out = Path(a.out or paths.interactions_data() / f"genie/apply_{design.name}.npz")
    np.savez(out, **{f"int_{k}": v for k, v in inter.items()}, **{f"part_{k}": v for k, v in part.items()})
    summary.update(n_interactions=int(len(inter["E"])), n_particles=int(len(part["pdg"])), time_s=round(time.time() - t0, 1))
    json.dump(summary, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"{len(inter['E']):,} interactions, {len(part['pdg']):,} particles -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
