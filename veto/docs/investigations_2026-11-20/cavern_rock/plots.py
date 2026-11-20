#!/usr/bin/env python3
"""Plots and summary numbers of the cavern-rock investigation from data.npz / cutflow.json (rock_study.py)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{D}/data.npz")
cf = json.load(open(f"{D}/cutflow.json"))
FAC = cf["fac_15y"]
Z_T = -2.14
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})
C_MU, C_RK, C_SEL, C_PAIR = "#1f77b4", "#d62728", "#2ca02c", "#9467bd"
summary = {}


def pnorm(p):
    return np.linalg.norm(p, axis=1)


rp, rw, good, acc = z["r_p"], z["r_w"], z["r_good"], z["r_acc"]
mp, mw = z["mu_p"], z["mu_w"]
P = pnorm(rp)
Pm = pnorm(mp)
pdg = np.abs(z["r_pdg"])
gidx = np.nonzero(good)[0]                      # rock good rows (in rock_study order = rk rows)
mr, mrw, mrmip = z["mr_pairs"], z["mr_w"], z["mr_mip"]
sel = mrmip < 0.10
rk_leg = gidx[mr[:, 1]]                         # rock row of the rock leg of each mu x rock candidate
mu_leg = mr[:, 0]

# ---------------------------------------------------------------- 1. momentum at SST1
bins = np.geomspace(0.1, 500, 61)
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.hist(P[acc], bins, weights=rw[acc], histtype="step", lw=1.5, color="grey", label=f"rock, charged in SST1 aperture ({rw[acc].sum():.3g}/spill)")
ax.hist(P[good], bins, weights=rw[good], histtype="step", lw=2, color=C_RK, label=f"rock, good tracks ({rw[good].sum():.3g}/spill)")
ax.hist(Pm, bins, weights=mw, histtype="step", lw=2, color=C_MU, label=f"shield muons, good ({mw.sum():.3g}/spill)")
s = mrw * FAC
ax.hist(P[rk_leg][sel], bins, weights=s[sel] * (rw[good].sum() / s[sel].sum()), histtype="stepfilled", alpha=0.35, color=C_RK,
        label="rock leg of mu x rock pairs with IP < 10 cm (shape)")
ax.hist(Pm[mu_leg][sel], bins, weights=s[sel] * (mw.sum() / s[sel].sum()), histtype="stepfilled", alpha=0.35, color=C_MU,
        label="muon leg of mu x rock pairs with IP < 10 cm (shape)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1, None)
ax.set_xlabel("momentum at SST1  |p|  [GeV/c]"); ax.set_ylabel("tracks per spill (4e13 PoT) per bin")
ax.set_title("Cavern-rock vs shield-muon tracks at SST1")
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout(); fig.savefig(f"{D}/p_sst1.png"); plt.close(fig)


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [float(x[o][np.searchsorted(c, qq)]) for qq in q]


summary["p_quantiles_10_50_90"] = dict(rock_good=wq(P[good], rw[good], [.1, .5, .9]), muon_good=wq(Pm, mw, [.1, .5, .9]),
                                       rock_leg_sel=wq(P[rk_leg][sel], mrw[sel], [.1, .5, .9]),
                                       muon_leg_sel=wq(Pm[mu_leg][sel], mrw[sel], [.1, .5, .9]))
summary["rock_good_frac_p_lt_3"] = float(rw[good & (P < 3)].sum() / rw[good].sum())
summary["rock_good_frac_p_lt_2"] = float(rw[good & (P < 2)].sum() / rw[good].sum())

# ---------------------------------------------------------------- 2. species
names = {11: "e±", 13: "μ±", 211: "π±", 321: "K±", 2212: "p/p̄"}
spec = {}
for lab, m, ww in (("rock good", good, rw), ):
    tot = ww[m].sum()
    spec[lab] = {names.get(k, "other"): 0.0 for k in list(names) + [0]}
    for k in np.unique(pdg[m]):
        spec[lab][names.get(int(k), "other")] += float(ww[m & (pdg == k)].sum() / tot)
pp = pdg[rk_leg]
for lab, m in (("rock leg, candidates", np.ones(len(mr), bool)), ("rock leg, IP < 10 cm", sel)):
    tot = mrw[m].sum()
    spec[lab] = {names.get(k, "other"): 0.0 for k in list(names) + [0]}
    for k in np.unique(pp[m]):
        spec[lab][names.get(int(k), "other")] += float(mrw[m & (pp == k)].sum() / tot)
summary["species_fractions"] = spec
qg = z["r_q"][good]
summary["rock_good_charge_frac_negative"] = float(rw[good][qg < 0].sum() / rw[good].sum())
summary["rock_leg_sel_charge_frac_negative"] = float(mrw[sel][z["r_q"][rk_leg][sel] < 0].sum() / mrw[sel].sum())
summary["muon_good_charge_frac_positive"] = float(mw[z["mu_q"] > 0].sum() / mw.sum())
summary["muon_leg_sel_charge_frac_positive"] = float(mrw[sel][z["mu_q"][mu_leg][sel] > 0].sum() / mrw[sel].sum())
fig, ax = plt.subplots(figsize=(8, 4.5))
labs = list(spec); cats = [c for c in list(names.values()) + ["other"]]
xx = np.arange(len(cats)); wbar = 0.27
for i, l in enumerate(labs):
    ax.bar(xx + (i - 1) * wbar, [100 * spec[l][c] for c in cats], wbar, label=l)
ax.set_xticks(xx, cats); ax.set_ylabel("weighted fraction [%]"); ax.set_title("Species of the rock tracks (good tracks at SST1)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{D}/species.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. direction at SST1
th_r = np.hypot(rp[:, 0], rp[:, 1]) / rp[:, 2]
th_m = np.hypot(mp[:, 0], mp[:, 1]) / mp[:, 2]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
b = np.geomspace(1e-3, 1.5, 61)
axs[0].hist(th_r[good], b, weights=rw[good], histtype="step", lw=2, color=C_RK, label="rock good")
axs[0].hist(th_m, b, weights=mw, histtype="step", lw=2, color=C_MU, label="shield muons good")
axs[0].hist(th_r[rk_leg][sel], b, weights=mrw[sel] * rw[good].sum() / mrw[sel].sum(), histtype="stepfilled", alpha=.35, color=C_RK, label="rock leg, IP<10 cm (shape)")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_ylim(1, None)
axs[0].set_xlabel("polar angle at SST1  θ = p_T / p_z  [rad]"); axs[0].set_ylabel("tracks / spill / bin"); axs[0].legend(fontsize=9)
tx_r, ty_r = rp[:, 0] / rp[:, 2], rp[:, 1] / rp[:, 2]
h = axs[1].hist2d(tx_r[good], ty_r[good], bins=[np.linspace(-0.2, 1.0, 80), np.linspace(-0.6, 0.6, 80)], weights=rw[good], cmap="Reds",
                  norm=matplotlib.colors.LogNorm())
plt.colorbar(h[3], ax=axs[1], label="rock tracks / spill / bin")
axs[1].set_xlabel("slope tx = px/pz at SST1"); axs[1].set_ylabel("slope ty = py/pz"); axs[1].set_title("rock good tracks: slopes (wall at x = -4.56 m)")
fig.tight_layout(); fig.savefig(f"{D}/direction.png"); plt.close(fig)
summary["theta_median"] = dict(rock_good=wq(th_r[good], rw[good], [.5])[0], muon_good=wq(th_m, mw, [.5])[0])

# ---------------------------------------------------------------- 4. origin: rock exit and vessel entry
v = z["r_v"]; surf = z["r_surf"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
bz = np.linspace(28, 84, 57)
for s_, lab, col in ((0, "wall (x = -4.56 m)", C_RK), (1, "floor (y = -3.36 m)", "orange")):
    m = good & (surf == s_)
    axs[0].hist(v[m, 2], bz, weights=rw[m], histtype="step", lw=2, color=col, label=f"{lab}: {rw[m].sum():.3g}/spill")
m = good
axs[0].hist(z["r_ivz"][m], bz, weights=rw[m], histtype="step", lw=1.2, ls="--", color="k", label="muon interaction z (rock)")
axs[0].set_xlabel("z of the rock exit point [m]"); axs[0].set_ylabel("good tracks / spill / m"); axs[0].legend(fontsize=8.5)
axs[0].set_title("where the good rock tracks leave the rock")
zin, face, ang = z["g_zin"], z["g_face"], z["g_ang"]
wg = rw[good]
for f_, lab, col in ((0, "side wall ±x", C_RK), (1, "top/bottom ±y", "orange"), (2, "entrance window", "k")):
    m = face == f_
    axs[1].hist(zin[m], bz, weights=wg[m], histtype="step", lw=2, color=col, label=f"{lab}: {100 * wg[m].sum() / wg.sum():.1f} %")
axs[1].set_xlabel("z where the line enters the vessel [m]"); axs[1].set_ylabel("good tracks / spill / m"); axs[1].legend(fontsize=9)
axs[1].set_title("vessel entry of the rock tracks")
ba = np.geomspace(1e-3, 1.6, 50)
mm = face >= 0
axs[2].hist(ang[mm], ba, weights=wg[mm], histtype="step", lw=2, color=C_RK, label="all good rock tracks")
selrows = np.unique(mr[sel, 1])
wsel = np.bincount(mr[sel, 1], weights=mrw[sel], minlength=len(wg))
axs[2].hist(ang[mm], ba, weights=wsel[mm] * wg.sum() / wsel.sum(), histtype="stepfilled", alpha=.35, color=C_RK, label="rock leg, IP < 10 cm (shape)")
axs[2].set_xscale("log"); axs[2].set_xlabel("angle of the track to the vessel wall it crosses [rad]"); axs[2].set_ylabel("tracks / spill / bin")
axs[2].legend(fontsize=9); axs[2].set_title("grazing angle at the vessel wall")
fig.tight_layout(); fig.savefig(f"{D}/origin.png"); plt.close(fig)
summary["vessel_entry_face_frac"] = {n: float(wg[face == f_].sum() / wg.sum()) for f_, n in ((0, "side"), (1, "top_bottom"), (2, "entrance"), (-1, "never_inside"))}
summary["good_from_wall_frac"] = float(rw[good & (surf == 0)].sum() / rw[good].sum())
# LS path in the SBT (20 cm) and the radiation / interaction lengths it represents (LS: X0 ~ 0.45 m, lambda_I ~ 0.95 m)
Lls = np.where(mm, 0.20 / np.sin(np.clip(ang, 1e-3, None)), np.nan)
summary["ls_path_median_m"] = wq(Lls[mm], wg[mm], [.5])[0]
summary["ls_path_median_m_selected_rock_legs"] = wq(Lls[mm], wsel[mm], [.5])[0]
summary["p_interact_in_20cm_LS_median"] = float(1 - np.exp(-summary["ls_path_median_m"] / 0.95))

# ---------------------------------------------------------------- 5. cut flow
steps = ["opposite charge", "DOCA < 2 cm", "vertex z in vessel", "vertex inside frustum", "true IP < 3 m", "measured IP < 2.5 m", "measured IP < 10 cm"]
a_ = [cf["mu_x_mu"][s]["per_15y"] for s in steps]
b_ = [cf["mu_x_rock"][s]["per_15y"] for s in steps]
fig, ax = plt.subplots(figsize=(10, 5))
xx = np.arange(len(steps))
ax.bar(xx - 0.2, a_, 0.4, color=C_MU, label="μ × μ")
ax.bar(xx + 0.2, b_, 0.4, color=C_RK, label="μ × rock")
for i in range(len(steps)):
    off = cf["mu_x_mu_official"]
    ref = {"true IP < 3 m": off["cand_15y"], "measured IP < 10 cm": off["sel_15y"]}.get(steps[i])
    if a_[i] > 0:
        txt = f"×{b_[i] / a_[i]:.0f}" + (f"\n(×{b_[i] / ref:.0f} vs ★)" if ref else "")
        ax.text(xx[i], max(a_[i], b_[i]) * 1.6, txt, ha="center", fontsize=9.5)
ax.plot([4 - 0.2, 6 - 0.2], [off["cand_15y"], off["sel_15y"]], "k*", ms=12, label="μ × μ official (reco inputs, full sample)")
ax.set_yscale("log"); ax.set_xticks(xx, steps, rotation=20, ha="right"); ax.set_ylabel("pairs per 15 years (±200 ps)")
ax.set_ylim(max(min(a_), 0.1) / 3, max(b_) * 30)
ax.set_title("Pair cut flow (true states; IP measured), ratio μ×rock / μ×μ above the bars"); ax.legend()
fig.tight_layout(); fig.savefig(f"{D}/cutflow.png"); plt.close(fig)

# ---------------------------------------------------------------- 6. why IP passes: IP of the muon leg alone vs pair IP
def ip_line(b, p):
    dz = Z_T - b[:, 2]
    return np.hypot(b[:, 0] + p[:, 0] / p[:, 2] * dz, b[:, 1] + p[:, 1] / p[:, 2] * dz)


ipmu = ip_line(z["mu_b"], mp)
mm_p, mmw, mmmip = z["mm_pairs"], z["mm_w"], z["mm_mip"]
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
bi = np.geomspace(1e-3, 5, 60)
axs[0].hist(mrmip, bi, weights=mrw * FAC, histtype="step", lw=2, color=C_RK, label="μ × rock candidates: pair IP")
axs[0].hist(ipmu[mu_leg], bi, weights=mrw * FAC, histtype="step", lw=1.5, ls="--", color=C_MU, label="μ × rock: IP of the muon leg alone")
axs[0].hist(mmmip, bi, weights=mmw * FAC, histtype="step", lw=2, color="k", label="μ × μ candidates: pair IP")
axs[0].axvline(0.10, color="grey", ls=":"); axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel("IP at the target plane [m]"); axs[0].set_ylabel("pairs / 15 y / bin"); axs[0].legend(fontsize=8.5)
axs[0].set_title("pair IP vs IP of the muon alone")
# momentum ratio of the legs
pr = P[rk_leg] / Pm[mu_leg]
p1 = pnorm(z["mu2_p"][mm_p[:, 0]]); p2 = pnorm(z["mu2_p"][mm_p[:, 1]])
prm = np.minimum(p1, p2) / np.maximum(p1, p2)
br = np.geomspace(1e-4, 1, 50)
axs[1].hist(pr, br, weights=mrw * FAC, histtype="step", lw=2, color=C_RK, label="μ × rock: p_rock / p_μ (candidates)")
axs[1].hist(pr[sel], br, weights=mrw[sel] * FAC, histtype="stepfilled", alpha=.35, color=C_RK, label="μ × rock, IP < 10 cm")
axs[1].hist(prm, br, weights=mmw * FAC, histtype="step", lw=2, color="k", label="μ × μ: p_soft / p_hard (candidates)")
axs[1].hist(prm[mmmip < .1], br, weights=mmw[mmmip < .1] * FAC, histtype="stepfilled", alpha=.35, color="k", label="μ × μ, IP < 10 cm")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("momentum ratio of the two legs"); axs[1].set_ylabel("pairs / 15 y / bin")
axs[1].legend(fontsize=8.5); axs[1].set_title("soft leg / hard leg")
bm = np.geomspace(1, 500, 50)
axs[2].hist(Pm[mu_leg], bm, weights=mrw * FAC, histtype="step", lw=2, color=C_RK, label="μ × rock: muon leg (candidates)")
axs[2].hist(Pm[mu_leg][sel], bm, weights=mrw[sel] * FAC, histtype="stepfilled", alpha=.35, color=C_RK, label="μ × rock: muon leg, IP < 10 cm")
axs[2].hist(np.r_[p1, p2], bm, weights=np.r_[mmw, mmw] * FAC, histtype="step", lw=2, color="k", label="μ × μ: both legs (candidates)")
axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("muon momentum at SST1 [GeV/c]"); axs[2].set_ylabel("legs / 15 y / bin")
axs[2].legend(fontsize=8.5); axs[2].set_title("momentum of the muon legs")
fig.tight_layout(); fig.savefig(f"{D}/ip_mechanism.png"); plt.close(fig)
summary["mu_leg_ip_lt_10cm_frac_in_sel"] = float(mrw[sel & (ipmu[mu_leg] < 0.1)].sum() / mrw[sel].sum())
summary["mu_leg_ip_lt_30cm_frac_in_sel"] = float(mrw[sel & (ipmu[mu_leg] < 0.3)].sum() / mrw[sel].sum())
summary["muon_good_ip_lt_10cm_frac"] = float(mw[ipmu < 0.1].sum() / mw.sum())
summary["muon_good_ip_lt_30cm_frac"] = float(mw[ipmu < 0.3].sum() / mw.sum())
summary["prock_over_pmu_median_sel"] = wq(pr[sel], mrw[sel], [.5])[0]

# opening angle and vertex z of the candidates
def angle(p1, p2):
    return np.arccos(np.clip((p1 * p2).sum(1) / pnorm(p1) / pnorm(p2), -1, 1))


oa_mr = angle(mp[mu_leg], rp[rk_leg])
oa_mm = angle(z["mu2_p"][mm_p[:, 0]], z["mu2_p"][mm_p[:, 1]])
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
bo = np.geomspace(1e-3, 1.5, 60)
axs[0].hist(oa_mr, bo, weights=mrw * FAC, histtype="step", lw=2, color=C_RK, label="μ × rock candidates")
axs[0].hist(oa_mm, bo, weights=mmw * FAC, histtype="step", lw=2, color="k", label="μ × μ candidates")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("opening angle of the pair [rad]"); axs[0].set_ylabel("pairs / 15 y / bin")
axs[0].legend(); axs[0].set_title("crossing angle: large angles sweep the whole vessel")
bz2 = np.linspace(31, 82, 52)
axs[1].hist(z["mr_vtx"][:, 2], bz2, weights=mrw * FAC, histtype="step", lw=2, color=C_RK, label="μ × rock candidates")
axs[1].hist(z["mr_vtx"][sel, 2], bz2, weights=mrw[sel] * FAC, histtype="stepfilled", alpha=.35, color=C_RK, label="μ × rock, IP < 10 cm")
axs[1].hist(z["mm_vtx"][:, 2], bz2, weights=mmw * FAC, histtype="step", lw=2, color="k", label="μ × μ candidates")
axs[1].set_yscale("log"); axs[1].set_xlabel("pair vertex z [m]"); axs[1].set_ylabel("pairs / 15 y / m"); axs[1].legend(fontsize=9)
axs[1].set_title("vertex position")
fig.tight_layout(); fig.savefig(f"{D}/pairs_geometry.png"); plt.close(fig)
summary["opening_angle_median"] = dict(mu_x_rock=wq(oa_mr, mrw, [.5])[0], mu_x_mu=wq(oa_mm, mmw, [.5])[0])
summary["mr_vertex_z_median_sel"] = wq(z["mr_vtx"][sel, 2], mrw[sel], [.5])[0]


# material the rock leg crosses entering the vessel (SBT shell = vessel wall: 2 x 0.8 cm Al + 20 cm LS + ribs ~ 23 g/cm2
# at normal incidence, lib/veto/sbt_material.py), NOT in the straight-line model. Survival without a nuclear interaction
# exp(-t / sin(a) / lambda) with lambda ~ 115 g/cm2 (pi, K in C/Al), 90 g/cm2 (p); e+- at >= 2 X0 (X0 ~ 0.58 X0 normal) count as lost.
T_NORMAL = 23.0
lam = np.where(pdg == 2212, 90.0, 115.0)
ang_all = np.full(len(P), np.nan); ang_all[gidx] = ang
t_g = T_NORMAL / np.sin(np.clip(ang_all, 1e-3, None))
surv = np.exp(-t_g / lam)
surv = np.where(pdg == 11, np.where(t_g / 40.0 >= 2, 0.0, np.exp(-t_g / 40.0)), surv)
surv = np.where(np.isfinite(surv), surv, 1.0)
summary["material_g_cm2_median_selected_rock_legs"] = wq(t_g[rk_leg][sel], mrw[sel], [.5])[0]
summary["survival_no_interaction_selected_rock_legs"] = float((surv[rk_leg][sel] * mrw[sel]).sum() / mrw[sel].sum())
summary["survival_no_interaction_good_rock"] = float((surv[good] * rw[good]).sum() / rw[good].sum())
summary["ip10_15y_scaled_by_survival"] = float(cf["mu_x_rock"]["measured IP < 10 cm"]["per_15y"] * summary["survival_no_interaction_selected_rock_legs"])


# the systematic kick: rock tracks all come from the -x wall (px > 0); mu+ point back to the target with an x offset that
# the soft negative rock track's momentum compensates
def xy_target(b, p):
    dz = Z_T - b[:, 2]
    return b[:, 0] + p[:, 0] / p[:, 2] * dz, b[:, 1] + p[:, 1] / p[:, 2] * dz


mx, my = xy_target(z["mu_b"], mp)
qm = z["mu_q"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
bx_ = np.linspace(-0.6, 0.6, 61)
axs[0].hist(mx[qm > 0], bx_, weights=mw[qm > 0], histtype="step", lw=2, color="C3", label="μ+ (good, all)")
axs[0].hist(mx[qm < 0], bx_, weights=mw[qm < 0], histtype="step", lw=2, color="C0", label="μ− (good, all)")
axs[0].set_xlabel("x of the muon line at the target plane z = −2.14 m [m]"); axs[0].set_ylabel("muons / spill / bin"); axs[0].legend()
axs[0].set_title("shield muons: where their line points (x)")
# pair: x at target of the summed momentum from the vertex, vs the muon alone, for the selected mu x rock pairs
vt = z["mr_vtx"]
ps = mp[mu_leg] + rp[rk_leg]
dzv = Z_T - vt[:, 2]
px_pair = vt[:, 0] + ps[:, 0] / ps[:, 2] * dzv
axs[1].hist(mx[mu_leg], bx_, weights=mrw * FAC, histtype="step", lw=2, color=C_MU, label="μ×rock candidates: muon line alone")
axs[1].hist(px_pair, bx_, weights=mrw * FAC, histtype="step", lw=2, color=C_RK, label="μ×rock candidates: pair (vertex + p_μ + p_rock)")
axs[1].hist(px_pair[sel], bx_, weights=mrw[sel] * FAC, histtype="stepfilled", alpha=.35, color=C_RK, label="pair, IP < 10 cm")
axs[1].set_xlabel("x at the target plane [m]"); axs[1].set_ylabel("pairs / 15 y / bin"); axs[1].legend(fontsize=8.5)
axs[1].set_title("the rock track's +x momentum shifts the μ+ line onto the target")
fig.tight_layout(); fig.savefig(f"{D}/ip_kick.png"); plt.close(fig)
summary["rock_good_frac_px_positive"] = float(rw[good & (rp[:, 0] > 0)].sum() / rw[good].sum())
summary["mu_plus_x_at_target_median"] = wq(mx[qm > 0], mw[qm > 0], [.5])[0]
summary["mu_minus_x_at_target_median"] = wq(mx[qm < 0], mw[qm < 0], [.5])[0]

# statistics of the rock sample
nd = len(np.unique(z["r_draw"][good]))
wsel_draw = np.bincount(z["r_draw"][rk_leg][sel], weights=mrw[sel])
wsel_draw = wsel_draw[wsel_draw > 0]
summary["rock_mc"] = dict(good_rows=int(good.sum()), draws_with_good=nd, weight_per_draw_per_spill=float(np.median(rw[good])),
                          selected_rock_rows=int(len(np.unique(mr[sel, 1]))),
                          top5_rock_rows_share_of_sel=float(np.sort(np.bincount(mr[sel, 1], weights=mrw[sel]))[::-1][:5].sum() / mrw[sel].sum()),
                          n_eff_rock_rows_sel=float(np.bincount(mr[sel, 1], weights=mrw[sel]).sum() ** 2 / (np.bincount(mr[sel, 1], weights=mrw[sel]) ** 2).sum()))
json.dump(summary, open(f"{D}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
