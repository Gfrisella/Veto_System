#!/usr/bin/env python3
"""cuda-muons' transport on CPU cores (cpp/cpu_muons, an OpenMP port of faster_muons_torch's field-map kernel): timing
and a statistical check against the GPU, on the same input muons, planes and steps as bin/generate_muon_study.py.

    dump    (farm GPU node, his container) the environment of set_environment on CPU tensors -> one .pt file
            python3 bin/cpu_transport.py dump --out env_TRY6.pt          (VETO_GEOMETRY_CACHE as generate_muon_study)
    run     transport the first N muons of an input h5 to z = 30.6 m then 82 m, on CPU (any machine with torch and the
            built module) or on the GPU (faster_muons_torch, farm / daint), timing each plane
            python3 bin/cpu_transport.py run --env env_TRY6.pt --input train_000.h5 --n 1000000 --device cpu --out cpu.npz
    compare two run outputs (CPU vs GPU, or two GPU seeds for the seed-to-seed spread)
            python3 bin/cpu_transport.py compare gpu_s0.npz cpu_s0.npz [gpu_s1.npz]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_muon_study import MM_DIR, N_STEPS, PLANE_DZ, PLANES  # noqa: E402


def ext_build(mm):
    """The compiled faster_muons_torch for this machine and Python (as generate_muon_study.ext_build)."""
    import platform
    v = sys.version_info
    for d in sorted((mm / "cuda_muons/faster_muons_torch/build").glob("lib.*")):
        if platform.machine() in d.name and (f"{v.major}.{v.minor}" in d.name or f"{v.major}{v.minor}" in d.name):
            return str(d.relative_to(mm))
    return "cuda_muons/faster_muons_torch/build/lib.linux-x86_64-3.10"


def dump(a):
    import torch
    mm = Path(a.mm_dir)
    for sub in ("cuda_muons", "cpp/build", "muons_and_matter", ext_build(mm)):
        sys.path.insert(0, str(mm / sub))
    from cuda_muons import set_environment
    from cuda_muons_ship import load_histogram
    corners, cavern, field = torch.load(os.environ["VETO_GEOMETRY_CACHE"], weights_only=False)
    hd = mm / "cuda_muons" / "data"
    iron, step = load_histogram(str(hd / "alias_histograms_G4_Fe.pkl"))
    concrete, step2 = load_histogram(str(hd / "alias_histograms_G4_CONCRETE.pkl"))
    assert step == step2
    env = set_environment(corners, cavern, {"iron": iron, "concrete": concrete}, field, device="cpu")
    assert env["use_field_map"]
    keep = {k: env[k] for k in ("material_histograms_cuda", "arb8_corners", "cells_arb8", "hashed_arb8", "cavern_params")}
    keep["field_args"] = tuple(t.contiguous() for t in env["field_args"])
    keep["step_length"] = float(step)
    torch.save(keep, a.out)
    print("wrote", a.out, "step", step)


def run(a):
    import h5py
    import torch
    env = torch.load(a.env, weights_only=False)
    if a.device == "cpu":
        sys.path.insert(0, a.build or str(HERE.parent / "cpp" / "cpu_muons"))
        import cpu_muons
        fn = cpu_muons.propagate_muons_with_alias_sampling
        print(f"cpu_muons, {cpu_muons.num_threads()} OpenMP threads", flush=True)
    else:
        mm = Path(a.mm_dir)
        sys.path.insert(0, str(mm / ext_build(mm)))
        import faster_muons_torch
        fn = faster_muons_torch.propagate_muons_with_alias_sampling
        env["material_histograms_cuda"] = {k: [t.cuda() for t in v] for k, v in env["material_histograms_cuda"].items()}
        for k in ("arb8_corners", "cells_arb8", "hashed_arb8", "cavern_params"):
            env[k] = env[k].cuda()
        env["field_args"] = (env["field_args"][0].cuda(), env["field_args"][1])   # the range stays on the host
    with h5py.File(a.input, "r") as f:
        n = a.n or len(f["px"])
        mom = np.stack([f[k][:n] for k in ("px", "py", "pz")], 1).astype(np.float32)
        pos = np.stack([f[k][:n] for k in ("x", "y", "z")], 1).astype(np.float32)
        q = (f["pdg"][:n] / -13).astype(np.float32)
        w = f["weight"][:n]
    P = torch.from_numpy(pos.copy())
    M = torch.from_numpy(mom.copy())
    Q = torch.from_numpy(q)
    if a.device != "cpu":
        P, M, Q = P.cuda(), M.cuda(), Q.cuda()
    out = {"weight": w, "x0": pos[:, 0], "y0": pos[:, 1], "pz0": mom[:, 2]}
    times = []
    planes = tuple(a.planes) if a.planes else PLANES
    for i, z in enumerate(planes):
        t0 = time.time()
        fn(P, M, Q, env["material_histograms_cuda"], env["arb8_corners"], env["cells_arb8"], env["hashed_arb8"],
           *env["field_args"], env["cavern_params"], True, z - PLANE_DZ / 2, 0.18, N_STEPS, env["step_length"], a.seed + i)
        if a.device != "cpu":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        p, m = P.cpu().numpy(), M.cpu().numpy()
        for k, arr, c in (("x", p, 0), ("y", p, 1), ("z", p, 2), ("px", m, 0), ("py", m, 1), ("pz", m, 2)):
            out[f"{k}_{z:g}"] = arr[:, c].copy()
        print(f"plane {z} m: {times[-1]:.2f} s, {n / times[-1]:.3g} muons/s", flush=True)
    if a.timing_only:
        out = {}
    if a.keep_box and out:          # only the muons reaching the first plane inside the box |x| < X, |y| < Y (e.g. the MS7 iron)
        z0 = f"{planes[0]:g}"
        X, Y = a.keep_box
        m = (out[f"z_{z0}"] >= planes[0] - PLANE_DZ) & (np.abs(out[f"x_{z0}"]) < X) & (np.abs(out[f"y_{z0}"]) < Y) & (out[f"pz_{z0}"] > 0)
        out = {k: v[m] for k, v in out.items()}
        out["row"] = np.nonzero(m)[0]; out["q"] = q[m].astype(np.int8)
        print(f"kept {m.sum():,} of {n:,} muons in the box at z = {planes[0]} m ({w[m].sum():.4g} of {w.sum():.4g} weight)", flush=True)
    np.savez(a.out, times=np.array(times), n=n, device=a.device, seed=a.seed, planes=np.array(planes), w_input=float(w.sum()), **out)
    print(f"{n:,} muons, {sum(times):.2f} s total -> {a.out}")


def compare(a):
    """Muons reaching each plane (weighted), and their spectra (p, x, y, pT) as histogram chi2 per bin."""
    runs = [np.load(f) for f in a.files]
    print("files:", ", ".join(f"{Path(f).name} ({str(r['device'])}, seed {int(r['seed'])}, {float(r['times'].sum()):.1f} s)"
                                for f, r in zip(a.files, runs)))
    for z in PLANES:
        tag = f"{z:g}"
        sel, var = [], []
        for r in runs:
            m = r[f"z_{tag}"] >= z - PLANE_DZ
            p = np.sqrt(r[f"px_{tag}"] ** 2 + r[f"py_{tag}"] ** 2 + r[f"pz_{tag}"] ** 2)
            sel.append(m)
            var.append(dict(p=p[m], x=r[f"x_{tag}"][m], y=r[f"y_{tag}"][m], pt=np.hypot(r[f"px_{tag}"], r[f"py_{tag}"])[m],
                            w=r["weight"][m]))
        print(f"\nplane {z} m: reached " + ", ".join(f"{m.sum():,} ({m.mean():.4%})" for m in sel))
        ref = var[0]
        edges = dict(p=np.linspace(0, 300, 31), x=np.linspace(-5, 5, 41), y=np.linspace(-5, 5, 41), pt=np.linspace(0, 5, 26))
        for j, v in enumerate(var[1:], 1):
            line = []
            for k, e in edges.items():
                h0 = np.histogram(ref[k], e)[0].astype(float)
                h1 = np.histogram(v[k], e)[0].astype(float)
                ok = (h0 + h1) > 0
                chi2 = (((h0 - h1) ** 2)[ok] / (h0 + h1)[ok]).sum()
                line.append(f"{k}: chi2/ndf {chi2:.0f}/{ok.sum()}")
            print(f"  {Path(a.files[j]).name} vs {Path(a.files[0]).name} (counts): " + "; ".join(line))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--out", required=True); d.add_argument("--mm-dir", default=str(MM_DIR))
    r = sub.add_parser("run")
    for k, kw in (("--env", dict(required=True)), ("--input", dict(required=True)), ("--out", dict(required=True)),
                  ("--n", dict(type=int, default=0)), ("--device", dict(default="cpu")), ("--seed", dict(type=int, default=0)),
                  ("--build", dict(default=None)), ("--timing-only", dict(action="store_true")), ("--mm-dir", dict(default=str(MM_DIR))),
                  ("--planes", dict(type=float, nargs="+", default=None)), ("--keep-box", dict(type=float, nargs=2, default=None))):
        r.add_argument(k, **kw)
    c = sub.add_parser("compare"); c.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"dump": dump, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
