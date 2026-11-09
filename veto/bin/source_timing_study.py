#!/usr/bin/env python3
"""What would a fresh input sample per seed cost? The source generators, timed, against the enriched input file.

The enriched file (601 M rows) is reused by every seed. Making new muons instead needs the source productions
(bin/tail_produce.py with forced decays, bin/charm_cascade.py), whose muons carry weights (forced decays: BR step /
lambda_dec), so their worth is the EFFECTIVE number per momentum bin, n_eff = (sum w)^2 / sum w^2. Per component
and log p bin at the input plane (z = -0.0045 m):
    ours   n_eff per core-hour (slices of the original productions, times from their logs: 150 - 200 slices in
           parallel on one node, so the per-core speed of a loaded machine)
    file   n_eff of the enriched file's rows of that component
    cost   core-hours for the file's n_eff in that bin = file / ours
In the file's convention: momentum at production, kept if it reached the 2018 plane behind the absorber with
E > 10 GeV (bin/old_convention.py: loss drawn from the 2018 map). Components as the file: tail (p >= 150 GeV: tail2), min. bias (p < 150: mb2), dimuon (dmlow2 below 150, dmhigh2
above), charm (charmc). Also the flux per spill ours / file (sanity: the reproduction agreed at 10 - 20 %).

    bin/remote.sh run bin/source_timing_study.py [--slices 10]            # CPU, minutes
Writes results/timing/source_timing.json (per bin arrays for the thinning study: bin/input_thinning_study.py).
"""

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from veto import paths

import old_convention as oc                                          # bin/: the 2018 loss map, the file's selection

FVS = Path("/disk/users/sqasim/workspace/uzh/fast-veto-sim")          # read only: the original productions
LOGP = np.arange(0.0, 2.81, 0.1)
POT = 5e13
PCUT = 150.0
PRODUCTIONS = {                                                      # tag: (component, p range)
    "tail2": ("tail", (PCUT, np.inf)),
    "mb2": ("min. bias", (0, PCUT)),
    "dmlow2": ("dimuon", (0, PCUT)),
    "dmhigh2": ("dimuon", (PCUT, np.inf)),
    "charmc": ("charm", (0, np.inf)),
}
COMPONENTS = ("tail", "min. bias", "dimuon", "charm")


def slice_seconds(tag):
    t = []
    for f in glob.glob(str(FVS / "log" / f"{tag}_*.log")):
        m = re.findall(r"^slice \d+:.*total (\d+) s", open(f).read(), re.M)
        t += [float(x) for x in m]
    return np.array(t)


def hist(p, w):
    i = np.clip(np.digitize(np.log10(p), LOGP) - 1, 0, len(LOGP) - 1)
    return np.bincount(i, weights=w, minlength=len(LOGP)), np.bincount(i, weights=w * w, minlength=len(LOGP)), np.bincount(i, minlength=len(LOGP))


def file_side():
    import h5py
    with h5py.File(paths.enriched_input_h5(), "r") as f:
        w = f["weight"][:].astype(np.float64)
        src = f["source"][:]
        p = np.sqrt(f["px"][:].astype(np.float64) ** 2 + f["py"][:] ** 2 + f["pz"][:] ** 2)
    comp = np.full(len(w), -1)
    comp[np.abs(w - 768.75) < 0.5] = 1
    comp[np.abs(w - 7.6875) < 0.01] = 2
    comp[np.abs(w - 326.16) < 0.5] = 3
    comp[src == 1] = 0
    out = {}
    for c, name in enumerate(COMPONENTS):
        m = comp == c
        s1, s2, n = hist(p[m], w[m])
        out[name] = dict(s1=s1, s2=s2, rows=n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices", type=int, default=10, help="slices read per production")
    args = ap.parse_args()
    fs = file_side()
    res = dict(logp=LOGP.tolist(), productions={}, components={})
    ours = {c: dict(neff_per_ch=np.zeros(len(LOGP)), flux=np.zeros(len(LOGP))) for c in COMPONENTS}
    table = oc.loss_map()
    data = {}
    for tag in PRODUCTIONS:
        files = sorted(glob.glob(str(FVS / "data" / "tail" / f"{tag}_[0-9]*.npz")))[:args.slices]
        ds = [dict(np.load(f)) for f in files]
        d = dict(files=len(files), events=sum(float(x["events"]) for x in ds),
                 w=np.concatenate([x["w_decay"].astype(np.float64) / float(x["rep"]) for x in ds]),
                 p=np.concatenate([np.sqrt(x["px"].astype(np.float64) ** 2 + x["py"] ** 2 + x["pz"] ** 2) for x in ds]),
                 p_prod=np.concatenate([x["p_prod"].astype(np.float64) for x in ds]))
        data[tag] = d
    # charm stores no creation momentum: plane momentum + our mean loss (from min. bias), as old_convention.py
    mb = data["mb2"]
    pl = np.geomspace(5, 400, 30)
    k = np.digitize(mb["p"], pl)
    ml = np.array([np.median((mb["p_prod"] - mb["p"])[k == i]) if np.sum(k == i) > 50 else np.nan for i in range(1, len(pl))])
    good = np.isfinite(ml)
    data["charmc"]["p_prod"] = data["charmc"]["p"] + np.interp(data["charmc"]["p"], np.sqrt(pl[1:] * pl[:-1])[good], ml[good])
    for tag, (comp, (lo, hi)) in PRODUCTIONS.items():
        secs = slice_seconds(tag)
        d = data[tag]
        files, events = range(d["files"]), d["events"]
        pp = d["p_prod"]
        m = (pp >= lo) & (pp < hi) & (pp - oc.draw_loss(table, pp) > 10.0)     # the file's selection (2018 plane, E > 10)
        s1, s2, n = hist(pp[m], d["w"][m])
        hours = len(files) * secs.mean() / 3600
        with np.errstate(divide="ignore", invalid="ignore"):
            neff = np.where(s2 > 0, s1 ** 2 / s2, 0)
        ours[comp]["neff_per_ch"] += neff / hours
        ours[comp]["flux"] += s1 * POT / events                        # muons per spill
        res["productions"][tag] = dict(component=comp, p_range=[lo, min(hi, 1e9)], slices_read=len(files),
                                       slice_seconds_mean=float(secs.mean()), slices_logged=int(len(secs)),
                                       rows_per_slice=float(n.sum() / len(files)), neff_per_core_hour=float(neff.sum() / hours),
                                       rows_per_core_hour=float(n.sum() / hours))
        print(f"{tag:8s} {comp:10s} {len(files)} slices x {secs.mean():.0f} s: {n.sum() / hours:,.0f} rows / core-h, "
              f"n_eff {neff.sum() / hours:,.0f} / core-h", flush=True)
    print(f"\n{'component':10s}{'file rows':>14}{'file n_eff':>14}{'ours n_eff/core-h':>19}{'flux ours/file':>16}{'core-h (all bins)':>19}{'core-h p>10 GeV':>17}")
    for c in COMPONENTS:
        f = fs[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            fneff = np.where(f["s2"] > 0, f["s1"] ** 2 / f["s2"], 0)
            cost = np.where(fneff > 0, fneff / ours[c]["neff_per_ch"], 0)
        hi10 = LOGP >= 1.0
        res["components"][c] = dict(file_rows=f["rows"].tolist(), file_neff=fneff.tolist(), file_flux=f["s1"].tolist(),
                                    ours_neff_per_core_hour=ours[c]["neff_per_ch"].tolist(), ours_flux=ours[c]["flux"].tolist(),
                                    core_hours_per_bin=np.nan_to_num(cost, posinf=-1).tolist())
        finite = np.isfinite(cost)
        print(f"{c:10s}{f['rows'].sum():>14,}{fneff.sum():>14,.0f}{ours[c]['neff_per_ch'].sum():>19,.0f}"
              f"{ours[c]['flux'].sum() / f['s1'].sum():>16.2f}{cost[finite].max():>19,.0f}{cost[finite & hi10].max():>17,.0f}")
        if not finite.all():
            print(f"   ({(~finite).sum()} bins with file muons but none of ours: {np.round(LOGP[~finite], 1).tolist()})")
    out = paths.results_dir() / "timing"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "source_timing.json", "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
