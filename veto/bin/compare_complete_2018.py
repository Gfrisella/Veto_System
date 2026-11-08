#!/usr/bin/env python3
"""The complete-mode target production (tail_produce.py --mode complete) against the 2018 FairShip files
(bin/build_reference_2018.py) and the 601 M h5, without charm (FairShip's charm class came from a separate
production; ours: muons without a charm / beauty ancestor, hf == 0): min. bias + dimuon / photon classes of the
2018 files against all our sources; the 2018 selection (E > 10 GeV at the 2018 plane, from the production momentum,
bin/old_convention.py) applied to ours.

  --compute ours  (where the slices are, farm or eiger): <out>/compare_ours_<tag>.npz (ours in the momentum at production
                  and after the target, by source)
  --compute refs  (farm, once): <out>/compare_refs.npz (2018 files at production, h5 after the target, by class)
  (local)        : python3 bin/compare_complete_2018.py compare_ours_<tag>.npz [...] --refs compare_refs.npz
                   (one figure per production; with several, also their ratio to the 2018 files side by side)
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

H5 = "/disk/users/gfrise/MuonsAndMatter/data/muons/full_sample_after_target.h5"
REF = "/disk/lhcb_data/sqasim/reference-2018-fairship-506M/reference_2018.h5"
PE = np.geomspace(10, 400, 33)
TE = np.linspace(0, 4, 21)
CLASSES = {"minbias": 768.75, "dimuon": 7.6875}
PROC = {1: "long-lived decays", 2: "prompt decays", 7: "prompt decays (cascade)", 8: "resonances in hadronic models", 3: "gamma -> mu mu", 4: "e+ e- -> mu mu",
        5: "Pythia muons", 6: "other"}


def hist(p, pt, w):
    i, j = np.digitize(p, PE) - 1, np.digitize(pt, TE) - 1
    ok = (i >= 0) & (i < len(PE) - 1) & (j >= 0) & (j < len(TE) - 1)
    k = i[ok] * (len(TE) - 1) + j[ok]
    n, shape = (len(PE) - 1) * (len(TE) - 1), (len(PE) - 1, len(TE) - 1)
    return (np.bincount(k, weights=w[ok], minlength=n).reshape(shape),
            np.bincount(k, weights=w[ok] ** 2, minlength=n).reshape(shape))


def compute_ours(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: F401
    import old_convention as oc
    table = oc.loss_map()
    res = dict(PE=PE, TE=TE)
    events = 0.0
    acc = {}
    files = sorted(glob.glob(f"{a.src}/{a.tag}_[0-9]*.npz"))
    for fn in files:
        d = np.load(fn)
        events += float(d["events"]) * float(d["rep"])
        pp = d["p_prod"].astype(np.float64)
        p = np.sqrt(d["px"].astype(np.float64) ** 2 + d["py"] ** 2 + d["pz"] ** 2)
        pt = np.hypot(d["px"], d["py"]).astype(np.float64)
        m = (d["hf"] == 0) & (pp - oc.draw_loss(table, pp) > 10.0)
        w = d["w_decay"].astype(np.float64) / float(d["rep"])
        for k in list(PROC) + [0]:
            mk = m & ((d["gen"] == k) if k else True)
            for conv, pv, ptv in (("prod", pp, pt * pp / p), ("exit", p, pt)):
                s1, s2 = hist(pv[mk], ptv[mk], w[mk])
                key = f"ours_{conv}_{k}"
                acc[key + "_s1"] = acc.get(key + "_s1", 0) + s1
                acc[key + "_s2"] = acc.get(key + "_s2", 0) + s2
    norm = 5e13 / events
    for k, v in acc.items():
        res[k] = v * (norm if k.endswith("_s1") else norm ** 2)
    res["pot"] = events
    print(f"{len(files)} slices, {events:.4g} p.o.t.", flush=True)
    out = Path(a.out) / f"compare_ours_{a.tag}.npz"
    np.savez(out, tag=a.tag, **res)
    print("wrote", out)


def compute_refs(a):
    import h5py
    res = dict(PE=PE, TE=TE)
    for tag, path, wcol in (("ref", REF, "w_fairship"), ("h5", H5, "weight")):
        with h5py.File(path, "r") as f:
            wc = f[wcol][:]
            for c, v in CLASSES.items():
                m = np.abs(wc - v) < 0.01 * v
                px, py, pz = (f[k][:][m].astype(np.float64) for k in ("px", "py", "pz"))
                w = f["weight"][:][m].astype(np.float64)
                pt = np.hypot(px, py)
                res[f"{tag}_{c}_s1"], res[f"{tag}_{c}_s2"] = hist(np.sqrt(pt ** 2 + pz ** 2), pt, w)
                del px, py, pz, w, pt
    out = Path(a.out) / "compare_refs.npz"
    np.savez(out, **res)
    print("wrote", out)


def plot(npz, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = dict(np.load(npz))
    h.update({k: v for k, v in np.load(refs).items() if k.startswith(("ref_", "h5_"))})
    pc, tc = np.sqrt(PE[1:] * PE[:-1]), 0.5 * (TE[1:] + TE[:-1])
    dp = np.diff(PE)
    fig, axs = plt.subplots(2, 3, figsize=(22, 11), gridspec_kw=dict(height_ratios=[3, 1.5]))
    SRC_COL = {1: "#eb6834", 2: "#1baa7a", 7: "#0f7c5c", 8: "#7a5c0f", 3: "#a35ec4", 4: "#d4a017", 5: "#777777", 6: "#bb3333"}
    for j, (conv, base, blab, xl) in enumerate((("prod", "ref", "2018 FairShip files (67)", "momentum at production [GeV]"),
                                                 ("exit", "h5", "601 M h5", "momentum after the target [GeV]"))):
        ax, bx = axs[0, j], axs[1, j]
        b1 = sum(h[f"{base}_{c}_s1"].sum(1) for c in CLASSES)
        b2 = sum(h[f"{base}_{c}_s2"].sum(1) for c in CLASSES)
        ax.errorbar(pc, np.where(b1 > 0, b1 / dp, np.nan), yerr=np.sqrt(b2) / dp, color="#222222", lw=2.2, ds="steps-mid",
                    label=f"{blab}: min. bias + dimuon/photon")
        ax.step(pc, h[f"{base}_dimuon_s1"].sum(1) / dp, where="mid", color="#222222", ls=":", lw=1.5, label=f"{blab}: dimuon/photon class")
        o1, o2 = h[f"ours_{conv}_0_s1"].sum(1), h[f"ours_{conv}_0_s2"].sum(1)
        ax.errorbar(pc, np.where(o1 > 0, o1 / dp, np.nan), yerr=np.sqrt(o2) / dp, color="#2a78d6", lw=2.2, ds="steps-mid",
                    label="ours, complete: all sources")
        for k, col in SRC_COL.items():
            if f"ours_{conv}_{k}_s1" not in h:                           # older files: no proc 7
                continue
            s = h[f"ours_{conv}_{k}_s1"].sum(1)
            if s.sum() > 0:
                ax.step(pc, np.where(s > 0, s / dp, np.nan), where="mid", color=col, lw=1.2, ls="--", label=f"ours: {PROC[k]}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel("muons per spill per GeV (no charm)")
        ax.set_title(f"{xl.split(' [')[0]}: ours vs the {blab}"); ax.legend(fontsize=7)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((b1 > 0) & (o1 > 0), o1 / b1, np.nan)
            e = r * np.sqrt(o2 / o1 ** 2 + b2 / b1 ** 2)
        bx.errorbar(pc, r, yerr=e, color="#2a78d6", marker="o", ms=4, lw=2)
        bx.axhline(1, color="k", lw=1); bx.set_ylim(0.4, 1.6); bx.set_xscale("log"); bx.set_xlabel(xl)
        bx.set_ylabel(f"ours / {base}"); bx.axvspan(10, 30, color="0.9", zorder=0)
    ax = axs[0, 2]
    m = (pc >= 30) & (pc < 150)
    for lab, s, ls, col in (("2018 files (production)", sum(h[f"ref_{c}_s1"] for c in CLASSES), ":", "#222222"),
                            ("ours (production)", h["ours_prod_0_s1"], "-", "#2a78d6"),
                            ("601 M h5 (after target)", sum(h[f"h5_{c}_s1"] for c in CLASSES), "--", "#eb6834"),
                            ("ours (after target)", h["ours_exit_0_s1"], "-.", "#1baa7a")):
        v = s[m].sum(0)
        ax.step(tc, v / v.sum(), where="mid", ls=ls, color=col, lw=2, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("pT [GeV]"); ax.set_ylabel("fraction"); ax.set_title("pT shape, p = 30-150 GeV")
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    tot = h["ours_prod_0_s1"][pc >= 20].sum()
    fr = [h[f"ours_prod_{k}_s1"][pc >= 20].sum() / tot if f"ours_prod_{k}_s1" in h else 0.0 for k in PROC]
    ax.barh(np.arange(len(PROC)), fr, color=[SRC_COL[k] for k in PROC])
    ax.set_yticks(np.arange(len(PROC)), list(PROC.values())); ax.set_xscale("log"); ax.set_xlim(1e-5, 1)
    ax.set_title("ours: share of the flux above 20 GeV by source"); ax.invert_yaxis()
    for i, f in enumerate(fr):
        ax.text(max(f, 1.2e-5), i, f" {f:.2%}", va="center", fontsize=8)
    for a_ in axs.flat:
        a_.grid(alpha=0.3)
    fig.suptitle(f"Complete target production {h['tag']} ({float(h['pot']):.3g} p.o.t.) vs the 2018 FairShip files and the h5, without charm",
                 fontsize=13)
    fig.tight_layout()
    o = Path(npz).with_suffix(".png")
    fig.savefig(o, dpi=95)
    print(o)


def overview(npzs, refs, labels):
    """Ratio to the 2018 files (production momentum, no charm) of several productions in one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.load(refs)
    pc = np.sqrt(PE[1:] * PE[:-1])
    b1 = sum(r[f"ref_{c}_s1"].sum(1) for c in CLASSES)
    b2 = sum(r[f"ref_{c}_s2"].sum(1) for c in CLASSES)
    e1 = sum(r[f"h5_{c}_s1"].sum(1) for c in CLASSES)
    e2 = sum(r[f"h5_{c}_s2"].sum(1) for c in CLASSES)
    tb = sum(r[f"ref_{c}_s1"] for c in CLASSES)
    fig, axs = plt.subplots(1, 3, figsize=(22, 6.5))
    cols = ["#2a78d6", "#eb6834", "#1baa7a", "#a35ec4", "#777777"]
    for f, lab, col in zip(npzs, labels, cols):
        h = np.load(f)
        for ax, conv, d1, d2, base in ((axs[0], "prod", b1, b2, "2018 files"), (axs[1], "exit", e1, e2, "h5")):
            o1, o2 = h[f"ours_{conv}_0_s1"].sum(1), h[f"ours_{conv}_0_s2"].sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where((d1 > 0) & (o1 > 0), o1 / d1, np.nan)
                e = q * np.sqrt(o2 / o1 ** 2 + d2 / d1 ** 2)
            m = pc >= 25
            ax.errorbar(pc[m], q[m], yerr=e[m], color=col, marker="o", ms=4, lw=2, label=lab)
            ax.set_title(f"ours / {base} ({'production' if conv == 'prod' else 'after-target'} momentum, no charm)")
        m = (pc >= 30) & (pc < 150)
        v, rb = h["ours_prod_0_s1"][m].sum(0), tb[m].sum(0)
        tcen = 0.5 * (TE[1:] + TE[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            axs[2].plot(tcen, (v / v.sum()) / (rb / rb.sum()), color=col, marker="o", ms=4, lw=2, label=lab)
    for ax in axs[:2]:
        ax.axhline(1, color="k", lw=1); ax.set_ylim(0.3, 1.7); ax.set_xlim(0, 400); ax.set_xlabel("p [GeV]")
        ax.set_ylabel("ratio (1 = same flux)"); ax.legend(fontsize=9)
    axs[2].axhline(1, color="k", lw=1); axs[2].set_ylim(0, 2.5); axs[2].set_xlabel("pT [GeV]")
    axs[2].set_ylabel("ratio of the normalised pT distributions")
    axs[2].set_title("pT shape / 2018 files' (p = 30-150 GeV, production)"); axs[2].legend(fontsize=9)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    o = Path(npzs[0]).with_name("compare_complete_overview.png")
    fig.savefig(o, dpi=95)
    print(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="*")
    ap.add_argument("--compute", choices=["ours", "refs"])
    ap.add_argument("--refs")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--src", default="/disk/lhcb_data/sqasim/complete_validation/tail")
    ap.add_argument("--tag", default="cv_complete")
    ap.add_argument("--out", default="/disk/lhcb_data/sqasim/complete_validation")
    a = ap.parse_args()
    if a.compute == "ours":
        compute_ours(a)
    elif a.compute == "refs":
        compute_refs(a)
    else:
        for f in a.npz:
            plot(f, a.refs)
        if len(a.npz) > 1:
            overview(a.npz, a.refs, a.labels or [Path(f).stem for f in a.npz])


if __name__ == "__main__":
    main()
