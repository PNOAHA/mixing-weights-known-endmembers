"""v0.2 local run, K = 3 (2026-09-17).  Replaces the SLSQP moduli of v0.1 by the exact 1-D moduli
(generator.modulus_k3) and adds the A2 predicted lines, the anchor family F2 (Section 4) and the Gaussian
flat limit F3 (Proposition 3.5 / Prop A.3 (numbering of paperA v0.3+: anchor section = 5, compressed kernel = App. A)).
Outputs: results/k3_moduli_v02.csv, results/k3_lambda_theta_v02.csv, results/k3_anchor_v02.csv,
         results/k3_gauss_v02.csv, figs/fig_a_moduli_v02.png, fig_b_lambda_theta_v02.png,
         fig_c_anchor_v02.png, fig_d_gauss_v02.png ; prints all checks.
"""
import os, csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generator import (tangent_basis, family_F1, family_F2, family_F3, point_quantities, modulus_k3, c_tv_k3,
                       delta0_tv_k3, a1_prediction, gram)

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
rng = np.random.default_rng(20260917)
t0 = time.time()
K, m = 3, 6
Z = tangent_basis(K)
OKABE = dict(blue="#0072B2", orange="#D55E00", green="#009E73", pink="#CC79A7", sky="#56B4E9", yellow="#E69F00", black="#000000")

def style(S):
    """scale fonts/lines by the print reduction factor S so that the printed sizes are ~9/7 pt (EJS text width 5.66 in)"""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9 * S,
                         'axes.titlesize': 9 * S, 'axes.labelsize': 9 * S, 'xtick.labelsize': 8 * S, 'ytick.labelsize': 8 * S,
                         'legend.fontsize': 7 * S, 'lines.linewidth': 1.0 * S, 'lines.markersize': 3.5 * S, 'axes.linewidth': 0.7 * S,
                         'xtick.major.width': 0.7 * S, 'ytick.major.width': 0.7 * S, 'xtick.major.size': 3 * S, 'ytick.major.size': 3 * S,
                         'pdf.fonttype': 42})
    return S

checks = {}


def write_csv(rows, name):
    with open(os.path.join(HERE, "results", name), "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)


# ============================================================ (a) moduli, F1, exact 1-D
thetas = [0.0, 0.3, 0.6]
ws = {"centroid": np.array([1, 1, 1]) / 3, "w=(.6,.3,.1)": np.array([0.6, 0.3, 0.1]), "w=(.8,.15,.05)": np.array([0.8, 0.15, 0.05])}
base = rng.dirichlet(np.ones(m) * 0.7, size=K)
fam = {th: (1 - th) * base + th / m for th in thetas}
rows = []; viol = []
S = style(12 / 5.66)
fig, axes = plt.subplots(len(thetas), len(ws), figsize=(12, 9))
for i, th in enumerate(thetas):
    P = fam[th]
    for j, (wname, w) in enumerate(ws.items()):
        q = point_quantities(P, w, Z, rng=rng)
        cTV = c_tv_k3(P, Z); d0TV = delta0_tv_k3(P, w, Z)
        rho_w = 1 / w.min()
        deltas = np.linspace(0.02, q.delta_max * 0.98, 14)
        om = {k: np.array([modulus_k3(P, w, Z, d, k) for d in deltas]) for k in ("TV", "H", "chi")}
        tv_line = deltas * cTV / 2
        h_lo = om["chi"] / (np.sqrt(2) * (1 + np.sqrt(rho_w))); h_hi = om["chi"] / np.sqrt(2)
        h_lo_rho = om["chi"] / (np.sqrt(2) * (1 + np.sqrt(q.rho))) if np.isfinite(q.rho) else np.full_like(deltas, np.nan)
        h_hi_rho = om["chi"] / (np.sqrt(2) * (1 + q.rho ** -0.5)) if np.isfinite(q.rho) else np.full_like(deltas, np.nan)
        for d, tv, h, c, tl, hl, hh, hlr, hhr in zip(deltas, om["TV"], om["H"], om["chi"], tv_line, h_lo, h_hi, h_lo_rho, h_hi_rho):
            rows.append(dict(theta=th, w=wname, delta=d, omega_TV=tv, omega_H=h, omega_chi=c, a1_pred=a1_prediction(q, d),
                             tv_line=tl, h_lo_w=hl, h_hi=hh, h_lo_rho=hlr, h_hi_rho=hhr, lam_min=q.lam_min, lam_max=q.lam_max,
                             delta0=q.delta0, delta0_TV=d0TV, r=q.r, delta_max=q.delta_max, c_TV=cTV, rho=q.rho, rho_w=rho_w, lam_star=q.lam_star))
            # ---- checks: A1 (Thm 3.1), A2(i) (Thm 3.2(i)), Lemma 3.3, Thm 3.2(ii)
            if c < a1_prediction(q, d) - 1e-9: viol.append(("chi below A1", th, wname, d))
            if d <= q.delta0 and abs(c - a1_prediction(q, d)) > 1e-6 * max(1, c): viol.append(("chi != A1 on (0,delta0]", th, wname, d, c, a1_prediction(q, d)))
            if d > q.delta0 and c > d * np.sqrt(q.lam_max) + 1e-9: viol.append(("chi above sqrt(lam_max)", th, wname, d))
            if tv < tl - 1e-9: viol.append(("TV below c_TV line", th, wname, d))
            if d <= d0TV and abs(tv - tl) > 1e-7: viol.append(("TV != line on (0,delta0_TV]", th, wname, d, tv, tl))
            if d > d0TV + 1e-9 and tv <= tl + 1e-9: viol.append(("TV not strictly above line beyond delta0_TV", th, wname, d))
            if not (hl - 1e-9 <= h <= hh + 1e-9): viol.append(("H outside Lemma 3.3 sandwich", th, wname, d, hl, h, hh))
            if np.isfinite(q.rho) and not (hlr - 1e-9 <= h <= hhr + 1e-9): viol.append(("H outside Thm 3.2(ii) sandwich", th, wname, d))
        rat = om["chi"] / deltas
        if any(rat[k + 1] < rat[k] - 1e-9 for k in range(len(rat) - 1)): viol.append(("chi/delta not monotone", th, wname))
        rat = om["TV"] / deltas
        if any(rat[k + 1] < rat[k] - 1e-9 for k in range(len(rat) - 1)): viol.append(("TV/delta not monotone", th, wname))
        ax = axes[i, j]
        ax.plot(deltas, om["chi"], "o", color=OKABE["blue"], ms=4, label=r"$\omega_\chi$")
        ax.plot(deltas, a1_prediction(q, deltas), "-", color=OKABE["blue"], lw=1, label=r"Thm 3.1: $\delta\sqrt{\lambda_{\min}}$")
        ax.plot(deltas, deltas * np.sqrt(q.lam_max), ":", color=OKABE["blue"], lw=1, alpha=0.7, label=r"$\delta\sqrt{\lambda_{\max}}$")
        ax.plot(deltas, om["H"], "s", color=OKABE["green"], ms=4, label=r"$\omega_H$")
        ax.fill_between(deltas, h_lo, h_hi, color=OKABE["green"], alpha=0.15, label="Lemma 3.3 sandwich")
        if np.isfinite(q.rho): ax.fill_between(deltas, h_lo_rho, h_hi_rho, color=OKABE["yellow"], alpha=0.25, label=r"Thm 3.2(ii) sandwich ($\rho$)")
        ax.plot(deltas, om["TV"], "^", color=OKABE["pink"], ms=4, label=r"$\omega_{TV}$")
        ax.plot(deltas, tv_line, "-", color=OKABE["pink"], lw=1, label=r"Thm 3.2(i): $\delta c_{TV}/2$")
        ax.axvline(q.delta0, color=OKABE["blue"], lw=0.8, ls="-."); ax.axvline(d0TV, color=OKABE["pink"], lw=0.8, ls="-.")
        ax.set_title(rf"$\theta$={th}, {wname}" + "\n" + rf"$\lambda_{{\min}}$={q.lam_min:.3g}, $\delta_0$={q.delta0:.3f}, $\delta_0^{{TV}}$={d0TV:.3f}, $\rho$={q.rho:.3g}", fontsize=6.5 * S)
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel("modulus")
        if i == 0 and j == 0: leg_h, leg_l = ax.get_legend_handles_labels()
fig.suptitle(r"$K=3$, F1 ($m=6$) · exact separation moduli vs Theorems 3.1–3.2 and Lemma 3.3", fontsize=10 * S)
fig.legend(leg_h, leg_l, loc="lower center", ncol=4, fontsize=6.5 * S, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(HERE, "figs", "fig_a_moduli_v02.png"), dpi=150); fig.savefig(os.path.join(HERE, "figs", "fig_a_moduli_v02.pdf")); plt.close(fig)
write_csv(rows, "k3_moduli_v02.csv")
checks["(a) moduli vs Thm 3.1 / Thm 3.2(i)(ii) / Lemma 3.3 on 126 points"] = (not viol, viol[:6])

# ============================================================ (b) lambda_min vs theta (F1)
th_grid = np.linspace(0, 0.95, 20); rows_b = []
S = style(6 / 3.5)
MK = {"centroid": "o-", "w=(.6,.3,.1)": "s--", "w=(.8,.15,.05)": "^:"}
fig, ax = plt.subplots(figsize=(6, 4))
for wname, w in ws.items():
    lam = []
    for th in th_grid:
        P = (1 - th) * base + th / m
        q = point_quantities(P, w, Z, rng=rng)
        lam.append(q.lam_min); rows_b.append(dict(theta=th, w=wname, lam_min=q.lam_min, lam_star=q.lam_star, upper=q.lam_star / w.min()))
    ax.semilogy(th_grid, lam, MK.get(wname, "o-"), ms=3 * S, label=wname)
lam_star = [r["lam_star"] for r in rows_b[:len(th_grid)]]
ax.semilogy(th_grid, lam_star, "k--", label=r"$\lambda_*$ (Thm 3.1(3), Lemma 3.4(ii))")
ax.set_xlabel(r"overlap $\theta$"); ax.set_ylabel(r"$\lambda_{\min}(I(w))$"); ax.legend(fontsize=7 * S)
ax.set_title(r"$K=3$, F1: degeneration of the tangent Fisher information", fontsize=9 * S)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "fig_b_lambda_theta_v02.png"), dpi=150); fig.savefig(os.path.join(HERE, "figs", "fig_b_lambda_theta_v02.pdf")); plt.close(fig)
write_csv(rows_b, "k3_lambda_theta_v02.csv")
ok_b = all(r["lam_star"] - 1e-12 <= r["lam_min"] <= r["upper"] + 1e-12 for r in rows_b)
checks["(b) lambda_* <= lambda_min(I(w)) <= lambda_*/min w (Lemma 3.4(ii)) on 60 points"] = (ok_b, [])

# ============================================================ (c) anchor family F2 (Section 4)
a_grid = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]; rows_c = []; c_const = 8 / 9
base2 = rng.dirichlet(np.ones(m) * 0.7, size=K); base2 = 0.7 * base2 + 0.3 / m
n_list = [50, 200, 1000]
for a in a_grid:
    P = np.zeros((K, m + K))
    for k in range(K):
        P[k, :m] = (1 - a) * base2[k]; P[k, m + k] = a
    for wname, w in ws.items():
        q = point_quantities(P, w, Z, rng=rng)
        lb42 = (a / w).min()                                                    # Prop 4.2
        risk41 = (w * (1 - w * a) / a).sum()                                    # Thm 4.1 (times 1/n)
        for n in n_list:
            lam = q.lam_min
            feas = n >= c_const / (lam * q.delta0 ** 2)
            dn = np.sqrt(c_const / (n * lam)); wp = w + dn * q.u_min
            p1, p0 = wp @ P, w @ P
            chi2 = ((p1 - p0) ** 2 / p0).sum(); H2 = 0.5 * ((np.sqrt(p1) - np.sqrt(p0)) ** 2).sum()
            rows_c.append(dict(a=a, w=wname, n=n, lam_min=lam, prop42_lb=lb42, delta0=q.delta0, pair_feasible=feas, delta_n=dn,
                               n_chi2=n * chi2, n_H2=n * H2, risk_anchor=risk41 / n, lower_43=1 / (27 * n * lam),
                               ratio=risk41 / n * 27 * n * lam))
ok_c = all(r["lam_min"] >= r["prop42_lb"] - 1e-12 for r in rows_c) and \
       all((not r["pair_feasible"]) or (abs(r["n_chi2"] - c_const) < 1e-9 and r["n_H2"] <= c_const / 2 + 1e-12) for r in rows_c) and \
       all(r["risk_anchor"] >= r["lower_43"] - 1e-15 for r in rows_c if r["pair_feasible"])
checks["(c) anchor F2: Prop 4.2 bound; Thm 4.3 pair n chi^2 = 8/9, n H^2 <= 4/9; Thm 4.1 risk >= 1/(27 n lambda)"] = (ok_c, [])
write_csv(rows_c, "k3_anchor_v02.csv")
S = style(10 / 5.66)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for wname in ws:
    sub = [r for r in rows_c if r["w"] == wname and r["n"] == 200]
    axes[0].loglog([r["a"] for r in sub], [r["lam_min"] for r in sub], "o-", ms=4 * S, label=rf"$\lambda_{{\min}}(I(w))$, {wname}")
    axes[0].loglog([r["a"] for r in sub], [r["prop42_lb"] for r in sub], "--", lw=1, label=rf"Prop 5.2: $\min_k a_k/w_k$, {wname}")
    axes[1].loglog([r["a"] for r in sub], [r["ratio"] for r in sub], "s-", ms=4 * S, label=wname)
axes[0].set_xlabel(r"anchor mass $a$"); axes[0].set_ylabel(r"$\lambda_{\min}$")
_h, _l = axes[0].get_legend_handles_labels(); fig.legend(_h, _l, loc="lower center", ncol=3, fontsize=6 * S, frameon=False, bbox_to_anchor=(0.5, -0.01))
axes[1].set_xlabel(r"anchor mass $a$"); axes[1].set_ylabel("risk (Thm 5.1) / bound (Thm 5.3)"); axes[1].legend(fontsize=7 * S)
axes[1].set_title("gap grows as $a\\to0$ (Prop 5.4)", fontsize=9 * S)
axes[0].set_title(r"$K=3$, F2 ($m=6$ shared cells, equal masses)", fontsize=9 * S)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(os.path.join(HERE, "figs", "fig_c_anchor_v02.png"), dpi=150); fig.savefig(os.path.join(HERE, "figs", "fig_c_anchor_v02.pdf")); plt.close(fig)

# ============================================================ (d) Gaussian flat limit F3 (Prop 3.5, P2.3)
hs = [2, 4, 8, 16, 32, 64]; rows_d = []
t = np.linspace(-1, 1, K)
V = np.vander(t, K, increasing=True); Q, R = np.linalg.qr(V); R33 = R[-1, -1] ** 2      # = 2/3 for t = (-1,0,1)
pred_const = R33 / np.math.factorial(K - 1) if hasattr(np, "math") else R33 / 2
pred_const = R33 / 2.0
for h in hs:
    L = 6 * h + 3; mm = 400001
    P, X = family_F3(K, h, L, mm, a=1.0)
    Kp = np.exp(-(t[:, None] - t[None, :]) ** 2 / (2 * h * h)); D = np.diag(np.exp(t ** 2 / (2 * h * h)))
    Lam_h = np.linalg.eigvalsh(Z.T @ D @ Kp @ D @ Z)[0]
    lam_Kp = np.linalg.eigvalsh(Z.T @ Kp @ Z)[0]
    for wname, w in {"centroid": ws["centroid"], "w=(.6,.3,.1)": ws["w=(.6,.3,.1)"], "vertex e1": np.array([1.0, 0, 0])}.items():
        G = gram(P, w); lam = np.linalg.eigvalsh(Z.T @ G @ Z)[0]
        rows_d.append(dict(h=h, w=wname, lam_min=lam, Lambda_h=Lam_h, ratio=lam / Lam_h, Lambda_h_h4=Lam_h * h ** 4,
                           pred_h4=pred_const, ratio_Lam_Kp=Lam_h / lam_Kp))
ok_d = all(abs(r["ratio"] - 1) < 0.02 for r in rows_d if r["h"] >= 32) and \
       all(abs(r["Lambda_h_h4"] / r["pred_h4"] - 1) < 0.05 for r in rows_d if r["h"] >= 16) and \
       all(abs(r["ratio_Lam_Kp"] - 2 / 3) < 0.02 for r in rows_d if r["h"] >= 16)
checks["(d) Gaussian F3: lambda_min/Lambda_h -> 1 (Prop 3.5), Lambda_h h^4 -> R33^2/2! (P2.3), Lambda_h/lambda(Z'K'Z) -> 2/3"] = (ok_d, [])
write_csv(rows_d, "k3_gauss_v02.csv")
S = style(6 / 3.5)
fig, ax = plt.subplots(figsize=(6, 4))
for wname in ("centroid", "w=(.6,.3,.1)", "vertex e1"):
    sub = [r for r in rows_d if r["w"] == wname]
    ax.loglog([r["h"] for r in sub], [r["lam_min"] for r in sub], "o-", ms=4 * S, label=rf"$\lambda_{{\min}}(I(w))$, {wname}")
sub = [r for r in rows_d if r["w"] == "centroid"]
ax.loglog(hs, [r["Lambda_h"] for r in sub], "k--", lw=1, label=r"$\Lambda_h$ (Prop 3.5)")
ax.loglog(hs, [pred_const / h ** 4 for h in hs], ":", color=OKABE["orange"], lw=1.5, label=r"Prop A.3: $h^{-4}R_{3,3}^2/2!$")
ax.set_xlabel(r"bandwidth $h$"); ax.set_ylabel(r"$\lambda_{\min}$"); ax.legend(fontsize=7 * S)
ax.set_xticks(hs); ax.set_xticklabels([str(h) for h in hs]); ax.minorticks_off()
ax.set_title(r"$K=3$ Gaussian endmembers at $t=(-1,0,1)$: flat limit", fontsize=9 * S)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "fig_d_gauss_v02.png"), dpi=150); fig.savefig(os.path.join(HERE, "figs", "fig_d_gauss_v02.pdf")); plt.close(fig)

# ============================================================ report
print("=" * 70)
for k_, (ok, detail) in checks.items():
    print(("PASS " if ok else "FAIL ") + k_, "" if ok else detail)
print("=" * 70)
print("ALL PASS" if all(v[0] for v in checks.values()) else "SOME FAIL")
print(f"done in {time.time()-t0:.1f}s; moduli rows={len(rows)}")
# numbers for the manuscript
sub = [r for r in rows if r["theta"] == 0.3 and r["w"] == "w=(.6,.3,.1)"]
print("theta=0.3, w=(.6,.3,.1): lam_min=%.4g lam_max=%.4g delta0=%.3f delta0_TV=%.3f c_TV=%.4f rho=%.3g rho_w=%.3g" %
      (sub[0]["lam_min"], sub[0]["lam_max"], sub[0]["delta0"], sub[0]["delta0_TV"], sub[0]["c_TV"], sub[0]["rho"], sub[0]["rho_w"]))
print("max rel. error chi vs A1 on delta<=delta0: %.2e" % max(abs(r["omega_chi"] / r["a1_pred"] - 1) for r in rows if r["delta"] <= r["delta0"]))
print("max rel. error TV vs line on delta<=delta0_TV: %.2e" % max(abs(r["omega_TV"] / r["tv_line"] - 1) for r in rows if r["delta"] <= r["delta0_TV"]))
print("H/H_hi range: %.3f-%.3f ; H/H_lo_w range: %.3f-%.3f" % (min(r["omega_H"] / r["h_hi"] for r in rows), max(r["omega_H"] / r["h_hi"] for r in rows),
      min(r["omega_H"] / r["h_lo_w"] for r in rows), max(r["omega_H"] / r["h_lo_w"] for r in rows)))
print("anchor ratio (n=200, centroid):", [(r["a"], round(r["ratio"], 2)) for r in rows_c if r["w"] == "centroid" and r["n"] == 200])
print("gauss:", [(r["h"], r["w"], round(r["ratio"], 4), round(r["Lambda_h_h4"], 4), round(r["ratio_Lam_Kp"], 4)) for r in rows_d])
