# -*- coding: utf-8 -*-
"""Queue 16: publication figures from the D107 full grid (results/d107/grid_*.csv, produced by run_grid_d107.py).
Recomputes the theorem checks with relative tolerances (the cluster run used an absolute 1e-9 on TV, which flags 1e-8
relative differences between two Brent refinements), prints every number quoted in the paper, and writes figs/d107/*.pdf|png.
Okabe-Ito palette; Times; vector PDF + 300 dpi PNG."""
import os, csv, json, glob, collections, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import tri as mtri

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'd107'); FIG = os.path.join(HERE, 'figs', 'd107'); os.makedirs(FIG, exist_ok=True)
ONLY_FIG123 = '--fig123-only' in sys.argv
FIG123 = ('figG2_k3_mle_ratio', 'figG3_mle_ratio_vs_minw', 'figG4_k3_anchor_vs_mle')


def save_figure(fig, name):
    """The JMVA refresh writes only its three figures when requested."""
    if ONLY_FIG123 and name not in FIG123:
        return
    for suffix in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, name + '.' + suffix), bbox_inches='tight')
OK = dict(blue='#0072B2', orange='#D55E00', green='#009E73', pink='#CC79A7', sky='#56B4E9', yellow='#E69F00', black='#000000', grey='#7f7f7f')
def style(S):
    """09-20: scale fonts/lines by the print reduction factor S (figure width / EJS text width 5.66 in)"""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9 * S, 'axes.labelsize': 9 * S,
                         'axes.titlesize': 9 * S, 'legend.fontsize': 7 * S, 'xtick.labelsize': 8 * S, 'ytick.labelsize': 8 * S, 'lines.linewidth': 1.0 * S,
                         'lines.markersize': 3.5 * S, 'axes.linewidth': 0.7 * S, 'figure.dpi': 120, 'savefig.dpi': 300, 'pdf.fonttype': 42})
    return S
S = style(1.0)
REL = 1e-6


def load(tag):
    rows = list(csv.DictReader(open(os.path.join(RES, f'grid_{tag}.csv'), encoding='utf-8')))
    for r in rows:
        for k in r:
            if k not in ('w', 'viol'): r[k] = float(r[k])
        r['wv'] = np.array(json.loads(r['w']))
    p = os.path.join(RES, f'grid_{tag}_mle.csv'); mle = []
    if os.path.exists(p):
        mle = list(csv.DictReader(open(p, encoding='utf-8')))
        for r in mle:
            for k in r:
                if k != 'w': r[k] = float(r[k]) if r[k] else np.nan
            r['wv'] = np.array(json.loads(r['w']))
    return rows, mle


def recheck(rows):
    """Theorem checks with relative tolerance; returns Counter of violation labels."""
    c = collections.Counter(); byw = collections.defaultdict(list)
    for r in rows: byw[r['w']].append(r)
    for w, rr in byw.items():
        rr.sort(key=lambda r: r['delta'])
        for r in rr:
            d = r['delta']; tol = REL * max(1.0, r['omega_chi'])
            if r['omega_chi'] < r['a1_pred'] - tol: c['chi<A1'] += 1
            if d <= r['delta0'] and abs(r['omega_chi'] - r['a1_pred']) > tol: c['chi!=A1'] += 1
            if d > r['delta0'] and r['omega_chi'] > d * np.sqrt(r['lam_max']) + tol: c['chi>lam_max'] += 1
            tolt = REL * max(1.0, r['omega_TV'])
            if r['omega_TV'] < r['tv_line'] - tolt: c['TV<line'] += 1
            K = len(r['wv'])
            if K == 3:   # K=3: exact 1-D moduli and exact delta0_TV
                if d <= r['delta0_TV'] and abs(r['omega_TV'] - r['tv_line']) > tolt: c['TV!=line'] += 1
                if d > r['delta0_TV'] * (1 + REL) and r['omega_TV'] <= r['tv_line'] + tolt: c['TV_not_strict'] += 1
            else:        # K=4: TV modulus by grid + Nelder-Mead/Powell (an upper bound), delta0_TV from the grid (a lower bound): report the excess only
                if d <= r['delta0_TV'] and r['omega_TV'] > r['tv_line'] + tolt:
                    c['K4_TV_above_line_on_[0,d0TV]'] += 1; c['K4_TV_max_rel_excess'] = max(c['K4_TV_max_rel_excess'], (r['omega_TV'] - r['tv_line']) / r['tv_line'])
            tolh = REL * max(1.0, r['omega_H'])
            if not (r['h_lo_w'] - tolh <= r['omega_H'] <= r['h_hi'] + tolh): c['H_sandwich'] += 1
        for a, b in zip(rr, rr[1:]):
            if b['omega_chi'] / b['delta'] < a['omega_chi'] / a['delta'] * (1 - REL): c['chi/delta_mono'] += 1
            if b['omega_TV'] / b['delta'] < a['omega_TV'] / a['delta'] * (1 - REL): c['TV/delta_mono'] += 1
    return c


def tern_xy(W):
    """Barycentric (K=3) -> 2-D coordinates of the equilateral triangle."""
    return W[:, 1] + 0.5 * W[:, 2], (np.sqrt(3) / 2) * W[:, 2]


def tern_panel(ax, W, z, title, cmap='viridis', vmin=None, vmax=None, log=False, fmt='%.2g', norm=None):
    x, y = tern_xy(W); T = mtri.Triangulation(x, y)
    zz = np.log10(z) if log else z
    tc = ax.tripcolor(T, zz, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color='k', lw=0.6 * S)
    ax.text(-0.02, -0.02, '$e_1$', ha='right', va='top', fontsize=8 * S); ax.text(1.0, -0.03, '$e_2$', ha='right', va='top', fontsize=8 * S); ax.text(0.5, np.sqrt(3) / 2 + 0.02, '$e_3$', ha='center', va='bottom', fontsize=8 * S)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=8.5 * S, pad=10 * S)
    cb = plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04, format=fmt); cb.ax.tick_params(labelsize=7 * S, width=0.7 * S)
    if log: cb.set_label(r'$\log_{10}$', fontsize=7 * S)
    return tc


def Lambda_h(K, h, a=1.0):
    t = np.linspace(-a, a, K); Z = tangent(K)
    M = np.exp(np.outer(t, t) / h ** 2)          # D_h K'_h D_h
    return float(np.linalg.eigvalsh(Z.T @ M @ Z)[0])


def tangent(K):
    A = np.eye(K) - np.ones((K, K)) / K; U, _, _ = np.linalg.svd(A); return U[:, :K - 1]


def RKK2(K, a=1.0):
    """R_{K,K}^2 = det(V_{<=K-1}^T V_{<=K-1}) / det(V_{<=K-2}^T V_{<=K-2}) for nodes linspace(-a,a,K) (Proposition A.3)."""
    t = np.linspace(-a, a, K); V = np.vander(t, K, increasing=True)
    return float(np.linalg.det(V.T @ V) / np.linalg.det(V[:, :K - 1].T @ V[:, :K - 1]))


out = {}
tags3 = [f'3_F1_{t}' for t in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)] + [f'3_F2_{a}' for a in (0.3, 0.1, 0.03)] + [f'3_F3_{h}' for h in (1.0, 2.0, 4.0, 8.0)]
tags4 = [t.replace('3_', '4_', 1) for t in tags3]
data = {}
for t in tags3 + tags4:
    if os.path.exists(os.path.join(RES, f'grid_{t}.csv')): data[t] = load(t)
print('loaded', len(data), 'tags')

# ---------------------------------------------------------------- theorem checks over the whole grid
tot = collections.Counter(); nrows = 0; nrows_K4 = 0; max_excess = {}; notstrict = []
for t, (rows, _) in data.items():
    c = recheck(rows)
    if 'K4_TV_max_rel_excess' in c: max_excess[t] = c.pop('K4_TV_max_rel_excess')
    tot.update(c); nrows += len(rows)
    if t.startswith('4_'): nrows_K4 += len(rows)
    for r in rows:
        if len(r['wv']) == 3 and r['delta'] > r['delta0_TV'] * (1 + REL) and r['omega_TV'] <= r['tv_line'] + REL * max(1.0, r['omega_TV']):
            notstrict.append(dict(tag=t, w=r['w'], delta_over_d0TV=r['delta'] / r['delta0_TV'] - 1, rel_excess=(r['omega_TV'] - r['tv_line']) / r['tv_line']))
    print(f'{t}: {len(rows)} rows, {len(set(r["w"] for r in rows))} w, checks: {dict(c) if c else "all pass"}')
print('TOTAL modulus rows', nrows, 'violations', dict(tot) if tot else 'none')
out['n_rows'] = nrows; out['n_rows_K4'] = nrows_K4; out['violations'] = dict(tot); out['K4_TV_max_rel_excess_by_tag'] = max_excess; out['notstrict_rows'] = notstrict
print('K4 max excess by tag', max_excess); print('not-strict rows', notstrict)

# ---------------------------------------------------------------- Fig G1: K=3, F1 theta=0.3: lambda_min, delta0 vs r, delta0_TV/delta0, H-sandwich position over the simplex
rows, mle = data['3_F1_0.3']
byw = {}
for r in rows: byw.setdefault(r['w'], []).append(r)
W = np.array([json.loads(w) for w in byw]); first = [byw[w][0] for w in byw]
lam = np.array([r['lam_min'] for r in first]); d0 = np.array([r['delta0'] for r in first]); rr = np.array([r['r'] for r in first])
d0tv = np.array([r['delta0_TV'] for r in first]); rho_w = np.array([r['rho_w'] for r in first])
# Hellinger position inside the Lemma 3.3 sandwich at the smallest delta: (omega_H - lo)/(hi - lo)
pos = np.array([(byw[w][0]['omega_H'] - byw[w][0]['h_lo_w']) / (byw[w][0]['h_hi'] - byw[w][0]['h_lo_w']) for w in byw])
ratioH = np.array([byw[w][0]['omega_H'] / (byw[w][0]['omega_chi'] / np.sqrt(2)) for w in byw])
def _last_feasible(rr):
    ok = [r for r in sorted(rr, key=lambda r: r['delta']) if r['delta'] <= r['delta0']]
    return ok[-1] if ok else sorted(rr, key=lambda r: r['delta'])[0]
ratioH_big = np.array([_last_feasible(byw[w])['omega_H'] / (_last_feasible(byw[w])['omega_chi'] / np.sqrt(2)) for w in byw])
delta_big = np.array([_last_feasible(byw[w])['delta'] for w in byw])
S = style(11 / 5.66)
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
tern_panel(axes[0], W, lam, r'(a) $\lambda_{\min}(I(w))$', log=True)
tern_panel(axes[1], W, d0 / rr, r'(b) $\delta_0(w)/r(w)$', log=True, fmt='%.1f')
tern_panel(axes[2], W, d0tv / d0, r'(c) $\delta_0^{\rm TV}(w)/\delta_0(w)$', fmt='%.2f')
tern_panel(axes[3], W, ratioH_big, r'(d) $\omega_H/(\omega_\chi/\sqrt{2})$ at largest $\delta\leq\delta_0$', cmap='plasma', fmt='%.2f')
fig.suptitle(r'$K=3$, family F1 ($m=6$, $\theta=0.3$), 91 lattice points of $\Delta$', fontsize=9 * S, y=1.02)
fig.tight_layout(); save_figure(fig, 'figG1_k3_landscape'); plt.close(fig)
out['G1'] = dict(lam_min_range=[float(lam.min()), float(lam.max())], d0_over_r=[float((d0 / rr).min()), float((d0 / rr).max())],
                 d0tv_over_d0=[float((d0tv / d0).min()), float((d0tv / d0).max())], frac_d0tv_le_d0=float((d0tv <= d0 + 1e-9).mean()),
                 ratioH=[float(ratioH.min()), float(ratioH.max())], ratioH_big=[float(ratioH_big.min()), float(ratioH_big.max())], delta_big=[float(delta_big.min()), float(delta_big.max())], pos=[float(pos.min()), float(pos.max())])
out['G1']['argmax_d0_over_r'] = json.loads(list(byw)[int(np.argmax(d0 / rr))]); print('G1', out['G1'])

# ---------------------------------------------------------------- Fig G2: MLE risk ratio n*risk/tau over the simplex (K=3, F1 theta=0.3), n = 100..6400 ; plus K=4 scatter vs min w
S = style(11 / 5.66)
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
mle_by = collections.defaultdict(dict)
for r in mle: mle_by[int(r['n'])][r['w']] = r
G2 = {}
for ax, n in zip(axes, (100, 400, 1600, 6400)):
    Wn = np.array([json.loads(w) for w in mle_by[n]]); rat = np.array([mle_by[n][w]['ratio'] for w in mle_by[n]])
    tern_panel(ax, Wn, rat, rf'$n={n}$', cmap='coolwarm', norm=TwoSlopeNorm(vcenter=1.0, vmin=0.5, vmax=1.1), fmt='%.2f')
    inner = Wn.min(1) >= 0.1
    G2[n] = dict(all=[float(rat.min()), float(rat.max())], inner_minw_ge_0p1=[float(rat[inner].min()), float(rat[inner].max())], inner_median=float(np.median(rat[inner])),
                 bfrac_max=float(max(mle_by[n][w]['boundary_frac'] for w in mle_by[n])))
fig.suptitle(r'$K=3$, F1 ($\theta=0.3$): $n\,\mathbb{E}_w\|\hat w_n-w\|^2/\mathrm{tr}\,I(w)^{-1}$ for the maximum likelihood estimator (2000 replications per lattice point)', fontsize=9 * S, y=1.02)
fig.tight_layout(); save_figure(fig, 'figG2_k3_mle_ratio'); plt.close(fig)
out['G2'] = G2; print('G2', G2)

# ---------------------------------------------------------------- Fig G3: K=4 (F1 theta=0.3) ratio vs min_k w_k, all n ; and K=3 all thetas overlay
S = style(8 / 5.66)
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
G3 = {}
for ax, K, tags in ((axes[0], 3, [f'3_F1_{t}' for t in (0.0, 0.3, 0.6)]), (axes[1], 4, [f'4_F1_{t}' for t in (0.0, 0.3, 0.6)])):
    cols = {100: OK['orange'], 400: OK['yellow'], 1600: OK['green'], 6400: OK['blue']}
    for t in tags:
        if t not in data or not data[t][1]: continue
        for n, c in cols.items():
            pts = [r for r in data[t][1] if int(r['n']) == n]
            mw = np.array([r['wv'].min() for r in pts]); rat = np.array([r['ratio'] for r in pts])
            thn = np.array([r['rho_w'] ** 2 / (n * r['lam_min']) for r in pts])
            ax.scatter(thn, rat, s=6, color=c, alpha=0.6, label=f'$n={n}$' if t == tags[0] else None, marker='o' if t.endswith('0.3') else ('^' if t.endswith('0.0') else 's'))
            G3[f'K{K}_{t}_n{n}'] = dict(inner=[float(rat[mw >= 0.1].min()), float(rat[mw >= 0.1].max())] if (mw >= 0.1).any() else None, all=[float(rat.min()), float(rat.max())])
    ax.axhline(1, color='k', lw=0.8); ax.set_xscale('log'); ax.set_xlabel(r'$\theta_n=\rho_w^2/(n\lambda_{\min}(I(w)))$'); ax.set_ylabel(r'$n\,\mathbb{E}\|\hat w_n-w\|^2/\mathrm{tr}\,I(w)^{-1}$')
    ax.set_title(rf'$K={K}$, F1, $\theta=0$ (triangles), $0.3$ (circles), $0.6$ (squares)', fontsize=7.5 * S); ax.set_ylim(0, 1.8)
    if K == 3: ax.legend(loc='lower left', fontsize=6.5 * S)
fig.tight_layout(); save_figure(fig, 'figG3_mle_ratio_vs_minw'); plt.close(fig)
out['G3'] = G3; print('G3', {k: v for k, v in G3.items() if 'n6400' in k or 'n400' in k})
bins = [(0, 0.3), (0.3, 1), (1, 3), (3, 10), (10, 100), (100, 1e9)]; G3b = {}
for K in (3, 4):
    pts = [r for t in (f'{K}_F1_{th}' for th in (0.0, 0.3, 0.6)) if t in data for r in data[t][1]]
    for lo, hi in bins:
        sel = [r for r in pts if lo <= r['rho_w'] ** 2 / (r['n'] * r['lam_min']) < hi]
        if sel: G3b[f'K{K}_[{lo},{hi})'] = dict(n=len(sel), min=float(min(r['ratio'] for r in sel)), max=float(max(r['ratio'] for r in sel)), median=float(np.median([r['ratio'] for r in sel])))
out['G3_theta_bins'] = G3b; print('G3 bins', G3b)
se_rel = [r['se'] / r['risk'] for r in data['3_F1_0.3'][1] if r['risk'] > 0]; out['G2_se_rel'] = [float(min(se_rel)), float(max(se_rel)), float(np.median(se_rel))]; print('se_rel', out['G2_se_rel'])

# ---------------------------------------------------------------- Fig G4: anchor family F2 over the simplex: anchor-count risk / MLE risk at n=1600, a = 0.3, 0.1, 0.03 (K=3)
S = style(9 / 5.66)
fig, axes = plt.subplots(1, 3, figsize=(9, 2.9)); G4 = {}
for ax, a in zip(axes, (0.3, 0.1, 0.03)):
    t = f'3_F2_{a}'
    if t not in data: continue
    rows_a, mle_a = data[t]; n = 1600
    pts = [r for r in mle_a if int(r['n']) == n]; Wa = np.array([r['wv'] for r in pts])
    anchor_risk = np.array([(r['wv'] * (1 - r['wv'] * a) / a).sum() / n for r in pts]); mle_risk = np.array([r['risk'] for r in pts])
    ratio = anchor_risk / mle_risk
    tern_panel(ax, Wa, ratio, rf'$a={a}$', cmap='magma', log=True, vmin=0, vmax=1.5, fmt='%.1f')
    lamr = np.array([r['lam_min'] for r in pts]); lb = 1 / (27 * n * lamr)
    G4[a] = dict(ratio_range=[float(ratio.min()), float(ratio.max())], median=float(np.median(ratio)), mle_over_lb=[float((mle_risk / lb).min()), float((mle_risk / lb).max())])
fig.suptitle(r'$K=3$, anchor family F2 ($m=6$ shared cells, $\theta=0.3$): $\log_{10}$ of risk(anchor count)/risk(MLE) at $n=1600$ over $\Delta$', fontsize=9 * S, y=1.02)
fig.tight_layout(); save_figure(fig, 'figG4_k3_anchor_vs_mle'); plt.close(fig)
out['G4'] = G4; print('G4', G4)

# ---------------------------------------------------------------- Fig G5: Gaussian F3: lambda_min(I(w))/Lambda_h over the simplex, h=1,2,4,8 (K=3) and K=4 vs min w
S = style(11 / 5.66)
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9)); G5 = {}
for ax, h in zip(axes, (1.0, 2.0, 4.0, 8.0)):
    t = f'3_F3_{h}'
    if t not in data: continue
    rows_h = data[t][0]; by = {}
    for r in rows_h: by.setdefault(r['w'], r)
    Wh = np.array([json.loads(w) for w in by]); lamh = np.array([by[w]['lam_min'] for w in by]); Lh = Lambda_h(3, h)
    tern_panel(ax, Wh, lamh / Lh, rf'$h={h:g}$', cmap='cividis', fmt='%.3f' if h >= 8 else '%.2f')
    G5[h] = dict(range=[float((lamh / Lh).min()), float((lamh / Lh).max())], Lambda_h=Lh, Lambda_h_h4=Lh * h ** 4,
                 w_at_min=list(map(float, Wh[int(np.argmin(lamh / Lh))])), w_at_max=list(map(float, Wh[int(np.argmax(lamh / Lh))])))
    if f'4_F3_{h}' in data:
        r4 = {}
        for r in data[f'4_F3_{h}'][0]: r4.setdefault(r['w'], r)
        l4 = np.array([r4[w]['lam_min'] for w in r4]); L4 = Lambda_h(4, h)
        G5[h]['K4_range'] = [float((l4 / L4).min()), float((l4 / L4).max())]; G5[h]['K4_Lambda_h_h6'] = L4 * h ** 6
fig.suptitle(r'$K=3$, Gaussian family F3 (nodes $t=(-1,0,1)$, 4001 cells): $\lambda_{\min}(I(w))/\Lambda_h$, uniformity in $w$ of Proposition 3.5(c)', fontsize=9 * S, y=1.02)
fig.tight_layout(); save_figure(fig, 'figG5_gauss_uniform'); plt.close(fig)
import math
G5['pred_K3'] = RKK2(3) / math.factorial(2); G5['pred_K4'] = RKK2(4) / math.factorial(3)
out['G5'] = G5; print('G5', G5)

# ---------------------------------------------------------------- Fig G6: Hellinger sandwich tightness across all F1 grids (K=3 and K=4): omega_H/(omega_chi/sqrt{2}) vs rho_w, coloured by theta
S = style(8 / 5.66)
fig, axes = plt.subplots(1, 2, figsize=(8, 3)); G6 = {}
for ax, K in ((axes[0], 3), (axes[1], 4)):
    for th, c in zip((0.0, 0.15, 0.3, 0.45, 0.6, 0.75), (OK['black'], OK['blue'], OK['green'], OK['yellow'], OK['orange'], OK['pink'])):
        t = f'{K}_F1_{th}'
        if t not in data: continue
        rr = [r for r in data[t][0] if r['delta'] <= r['delta0']]
        rho = np.array([r['rho_w'] for r in rr]); rat = np.array([r['omega_H'] / (r['omega_chi'] / np.sqrt(2)) for r in rr]); lo = np.array([1 / (1 + np.sqrt(r['rho_w'])) for r in rr])
        oc = np.array([r['omega_chi'] for r in rr])
        ax.scatter(oc, rat, s=4, color=c, alpha=0.5, label=rf'$\theta={th}$')
        G6[f'K{K}_th{th}'] = [float(rat.min()), float(rat.max()), float((rat / lo).min())]
    ax.axhline(1, color='k', lw=0.8, label=r'Lemma 3.3 upper line'); ax.axhline(0.5, color=OK['grey'], lw=0.8, ls='--', label=r'small-$\delta$ limit $1/2$ ($H^2\approx\chi^2/8$)')
    ax.set_xscale('log'); ax.set_xlabel(r'$\omega_\chi(\delta;w)$'); ax.set_ylabel(r'$\omega_H/(\omega_\chi/\sqrt{2})$'); ax.set_title(rf'$K={K}$, F1, all lattice points, $\delta\leq\delta_0(w)$', fontsize=8.5 * S); ax.set_ylim(0, 1.05)
    if K == 3: ax.legend(ncol=2, loc='lower left', fontsize=6 * S)
fig.tight_layout(); save_figure(fig, 'figG6_hellinger_sandwich'); plt.close(fig)
out['G6'] = G6; out['G6_lower_factor_min'] = float(min(v[2] for v in G6.values())); out['G6_ratio_range'] = [float(min(v[0] for v in G6.values())), float(max(v[1] for v in G6.values()))]; print('G6', G6)
# small-delta constant: omega_H^2/omega_chi^2 at the smallest delta (0.02) over every lattice point of every family (expected -> 1/8)
sq = []
for t, (rows_t, _) in data.items():
    for r in rows_t:
        if r['delta'] <= 0.02 + 1e-12: sq.append(r['omega_H'] ** 2 / r['omega_chi'] ** 2)
sq = np.array(sq); out['H2_over_chi2_at_delta0p02'] = dict(n=int(len(sq)), min=float(sq.min()), max=float(sq.max()), median=float(np.median(sq)), target=0.125)
print('H^2/chi^2 at delta=0.02:', out['H2_over_chi2_at_delta0p02'])

if ONLY_FIG123:
    summary = {k: v for k, v in out.items() if k in ('G2', 'G3', 'G3_theta_bins', 'G2_se_rel', 'G4')}
    summary_name = '_summary_fig123_dualgap.json'
else:
    summary, summary_name = out, '_summary_numbers.json'
json.dump(summary, open(os.path.join(RES, summary_name), 'w'), indent=1)
print('figures written to', FIG)
