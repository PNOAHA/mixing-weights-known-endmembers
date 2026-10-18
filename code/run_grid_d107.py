# -*- coding: utf-8 -*-
"""paperA queue 16: full grid on D107 (2026-09-19).
One Slurm array task = one (K, family, parameter) combination. For every interior lattice point w of the simplex:
  (a) point quantities (lambda_min/max, delta0, r, delta_max, c_TV, delta0_TV, rho, lambda_*),
  (b) the three exact separation moduli on a delta grid (K=3: 1-D angle grid + Brent; K=4: Fibonacci grid on S^2 in T + Nelder-Mead),
      with the Theorem 3.1 / 3.2(i)(ii) / Lemma 3.3 checks,
  (c) Monte Carlo of the maximum likelihood estimator (EM + tangent Newton + dual-gap/SLSQP acceptance) at n in N_LIST: risk, tau/n, ratio, P(E1), boundary fraction, dual-gap stats.
Usage: python run_grid_d107.py TASK_ID   (TASK_ID in 0..len(TASKS)-1)  -> results/grid_<K>_<fam>_<par>.csv (+ _mle.csv)
Deterministic: seed = 20260919 + TASK_ID. numpy/scipy only.
"""
import os, sys, csv, time, json
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator import tangent_basis, family_F1, family_F2, family_F3, gram, gram_min, delta0_of_u, _dist_vec
from mle_dualgap import polish_mle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
N_LIST = (100, 400, 1600, 6400); REPS = 2000
TASKS = []
for K in (3, 4):
    for th in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75): TASKS.append((K, 'F1', th))
    for a in (0.3, 0.1, 0.03): TASKS.append((K, 'F2', a))
    for h in (1.0, 2.0, 4.0, 8.0): TASKS.append((K, 'F3', h))


def simplex_lattice(K, n, margin=0.05):
    pts = []
    def rec(prefix, left, depth):
        if depth == K - 1: pts.append(prefix + [left]); return
        for i in range(left + 1): rec(prefix + [i], left - i, depth + 1)
    rec([], n, 0)
    W = np.array(pts, float) / n
    return (1 - margin) * W + margin / K


def fib_sphere(N):
    i = np.arange(N) + 0.5; phi = np.arccos(1 - 2 * i / N); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])   # 3 x N unit vectors


def dirs(K, Z, N):
    if K == 3:
        phi = np.linspace(0, 2 * np.pi, N, endpoint=False); return Z @ np.stack([np.cos(phi), np.sin(phi)])
    return Z @ fib_sphere(N)


def ang2dir(K, Z, x):
    if K == 3: return Z @ np.array([np.cos(x[0]), np.sin(x[0])])
    return Z @ np.array([np.cos(x[0]) * np.sin(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[1])])


def dir2ang(K, Z, u):
    y = Z.T @ u
    if K == 3: return np.array([np.arctan2(y[1], y[0])])
    return np.array([np.arctan2(y[1], y[0]), np.arccos(np.clip(y[2], -1, 1))])


def modulus(P, w, Z, delta, kind, U, n_ref=5):
    """omega_d(delta; w): minimiser is on the sphere ||v||=delta (homogeneity / monotonicity, see v0.2), so minimise over directions."""
    K = len(w)
    # candidate directions: the grid plus the vertex directions (e_k - w) and their feasible-boundary neighbours
    Vx = np.eye(K) - w[:, None].T; Vx = (Vx / np.linalg.norm(Vx, axis=1, keepdims=True)).T   # K unit directions
    U = np.concatenate([U, Vx], axis=1)
    d = _dist_vec(P, w, delta * U, kind)
    if not np.isfinite(d).any(): return np.inf, None
    best = np.inf; ubest = None
    n_ref = max(n_ref, 12)
    if kind == 'chi':   # exact QP on the sphere: min y'Iy s.t. |y|=delta, w+Zy>=0 (SLSQP, multi-start from feasible grid points)
        G = gram(P, w); I = Z.T @ G @ Z
        for i in np.argsort(d)[:8]:
            y0 = Z.T @ U[:, i] * delta
            r = minimize(lambda y: y @ I @ y, y0, jac=lambda y: 2 * I @ y, method='SLSQP',
                         constraints=[{'type': 'eq', 'fun': lambda y: y @ y - delta ** 2, 'jac': lambda y: 2 * y},
                                      {'type': 'ineq', 'fun': lambda y: w + Z @ y, 'jac': lambda y: Z}],
                         options={'ftol': 1e-14, 'maxiter': 500})
            if r.success and (w + Z @ r.x).min() >= -1e-12 and abs(r.x @ r.x - delta ** 2) < 1e-9 * delta ** 2:
                val = float(np.sqrt(max(r.fun, 0.0)))
                if val < best: best = val; ubest = Z @ r.x / delta
    if K == 4 and kind != 'chi': n_ref = 30          # non-smooth TV / H on S^2: more restarts + Powell polish
    for i in np.argsort(d)[:n_ref]:
        f = lambda x: float(v) if np.isfinite(v := _dist_vec(P, w, delta * ang2dir(K, Z, x)[:, None], kind)[0]) else 1e9
        x0 = dir2ang(K, Z, U[:, i])
        if K == 3:
            h = 2 * np.pi / U.shape[1]
            r = minimize_scalar(lambda t: f([t]), bounds=(x0[0] - h, x0[0] + h), method='bounded', options={'xatol': 1e-12}); fx, xx = r.fun, [r.x]
        else:
            r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 600}); fx, xx = r.fun, r.x
            if kind != 'chi':
                r2 = minimize(f, xx, method='Powell', options={'xtol': 1e-10, 'ftol': 1e-13, 'maxiter': 2000})
                if r2.fun < fx: fx, xx = r2.fun, r2.x
        cand = min(d[i], fx if fx < 1e8 else np.inf)
        if cand < best: best = cand; ubest = U[:, i] if d[i] <= fx else ang2dir(K, Z, xx)
    return float(best), ubest


def c_tv_and_delta0tv(P, w, Z, U, tol=1e-9):
    vals = np.abs(U.T @ P).sum(axis=1); K = len(w)
    i = int(np.argmin(vals)); x0 = dir2ang(K, Z, U[:, i])
    g = lambda x: float(np.abs(ang2dir(K, Z, x) @ P).sum())
    if K == 3:
        h = 2 * np.pi / U.shape[1]; r = minimize_scalar(lambda t: g([t]), bounds=(x0[0] - h, x0[0] + h), method='bounded', options={'xatol': 1e-12})
    else:
        r = minimize(g, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-13})
    c = min(vals[i], r.fun)
    idx = np.where(vals <= vals[i] + tol * max(1, vals[i]))[0]   # minimisers among grid directions (refined value c may be smaller)
    d0 = max(delta0_of_u(w, U[:, k]) for k in idx)
    return float(c), float(d0)


def point_q(P, w, Z):
    K = len(w); G = gram(P, w); I = Z.T @ G @ Z; lam, V = np.linalg.eigh(I)
    u = Z @ V[:, 0]; d0 = max(delta0_of_u(w, u), delta0_of_u(w, -u))
    r = np.sqrt(K / (K - 1)) * w.min(); dmax = max(np.linalg.norm(np.eye(K)[k] - w) for k in range(K))
    supp = P.max(axis=0) > 0; Ps = P[:, supp]
    rho = float((Ps.max(axis=0) / Ps.min(axis=0)).max()) if (Ps.min(axis=0) > 0).all() else np.inf
    lam_star = float(np.linalg.eigvalsh(Z.T @ gram_min(P) @ Z)[0])
    tau = float(np.trace(np.linalg.inv(I)))
    return dict(I=I, lam_min=float(lam[0]), lam_max=float(lam[-1]), delta0=float(d0), r=float(r), delta_max=float(dmax), rho=rho, lam_star=lam_star, tau=tau)


def dual_gap(P, N, W):
    """KKT dual gap max_k g_k - w^T g for average multinomial score g = (N/n) @ (P/pw)^T / row? 
    Here g_k = n^{-1} sum_x N_x p_k(x)/p_w(x) = (N * (P_k/pw)).sum / n per row.
    """
    n = N.sum(1, keepdims=True).astype(float)
    pw = np.maximum(W @ P, 1e-300)
    # g: (R, K) with g_rk = sum_m N_rm * P_km / pw_rm / n_r
    g = (np.where(N > 0, N / pw, 0.0) @ P.T) / n
    return (g.max(1) - (W * g).sum(1))


def _proj_simplex(v):
    v = np.asarray(v, float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, len(v) + 1)
    rho = ind[u - cssv / ind > 0][-1]
    theta = cssv[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def refine_mle_slsqp(P, N_row, w0, gap_tol=1e-8):
    """Single-replication constrained MLE on the simplex via SLSQP (fallback).
    Returns best (w, gap, success) among starts; caller should keep prior w if gap does not improve.
    """
    K = P.shape[0]
    def neg_ll(w):
        pw = np.maximum(w @ P, 1e-300)
        return -float((N_row * np.log(pw)).sum())
    cons = {'type': 'eq', 'fun': lambda w: float(w.sum() - 1.0)}
    bounds = [(0.0, 1.0)] * K
    starts = [_proj_simplex(w0)]
    # barycentre and coordinate-near starts help near faces
    starts.append(np.ones(K) / K)
    for k in range(K):
        e = np.full(K, 1e-6); e[k] = 1.0; starts.append(_proj_simplex(e))
    best_w, best_gap, best_ok = _proj_simplex(w0), float('inf'), False
    for s in starts:
        r = minimize(neg_ll, s, method='SLSQP', bounds=bounds, constraints=cons,
                     options={'ftol': 1e-14, 'maxiter': 400, 'disp': False})
        w = _proj_simplex(np.maximum(r.x, 0.0))
        gap = float(dual_gap(P, N_row[None, :], w[None, :])[0])
        if gap < best_gap:
            best_w, best_gap, best_ok = w, gap, bool(getattr(r, 'success', False)) or gap <= gap_tol
        if best_gap <= gap_tol:
            break
    return best_w, best_gap, best_ok


def mle_vectorized(P, N, W0, iters_em=2000, iters_newton=20, gap_tol=1e-8, refine=True, strict=True):
    """EM + tangent Newton with per-replication KKT dual-gap acceptance.
    Returns W, gaps, accepted mask, n_refined. The default strict mode
    includes simplex feasibility and refuses to return unaccepted refined
    solutions. strict=False reproduces the first-pass solver for auditing.
    Dual gap: max_k g_k - w^T g with g_k = n^{-1} sum_x N_x p_k(x)/p_w(x).
    """
    n = N.sum(1, keepdims=True); W = W0.copy(); K = P.shape[0]; Z = tangent_basis(K)
    Rbatch = W.shape[0]
    newton_ok = np.ones(Rbatch, dtype=bool)
    for _ in range(iters_em):
        pw = W @ P
        W = W * ((np.where(N > 0, N / np.maximum(pw, 1e-300), 0.0)) @ P.T) / n   # 0/0 -> 0
    def ll(W): return (N * np.log(np.maximum(W @ P, 1e-300))).sum(1)
    for _ in range(iters_newton):
        active = newton_ok
        if not active.any():
            break
        pw = W @ P
        if strict:
            # Scores from unobserved cells are zero. In particular, a private
            # cell with count zero and weight zero must not produce 0/0.
            R = np.divide(P[None], pw[:, None, :],
                          out=np.zeros((Rbatch, K, P.shape[1])),
                          where=(N[:, None, :] > 0) & (pw[:, None, :] > 0))
        else:
            R = P[None] / pw[:, None, :]
        grad = np.einsum('rm,rkm->rk', N, R) @ Z
        H = np.einsum('rm,rkm,rlm->rkl', N, R, R); H = np.einsum('ki,rkl,lj->rij', Z, H, Z)
        step = np.zeros_like(W)
        # Solve per-row so one LinAlgError does not abort the whole batch
        for i in np.where(active)[0]:
            try:
                step[i] = np.linalg.solve(H[i], grad[i]) @ Z.T
            except np.linalg.LinAlgError:
                newton_ok[i] = False
        alpha = np.ones((Rbatch, 1)); ll0 = ll(W)
        for _bt in range(30):
            Wn = W + alpha * step
            ok = active & (Wn.min(1) > 0) & (ll(Wn) >= ll0 - 1e-12)
            # only shrink failed active rows
            fail = active & ~ok
            alpha[fail] *= 0.5
            if not fail.any():
                break
        Wn = W + alpha * step
        take = active & (Wn.min(1) > 0)
        if strict:
            take &= np.isfinite(Wn).all(1) & (ll(Wn) >= ll0 - 1e-12)
        W = np.where(take[:, None], Wn, W)
    gaps = dual_gap(P, N, W)
    accepted = gaps <= gap_tol
    if strict:
        accepted &= (np.isfinite(gaps) & np.isfinite(W).all(1)
                     & (W.min(1) >= 0) & (np.abs(W.sum(1) - 1) <= 1e-12))
    n_refined = 0
    if refine and (not accepted.all()):
        for i in np.where(~accepted)[0]:
            if strict:
                w_new, gap_new, ok, reason = polish_mle(P, N[i], W[i], gap_tol=gap_tol)
                if not ok:
                    raise RuntimeError(f'MLE acceptance failed for replication {i}: gap={gap_new}, {reason}')
                W[i], gaps[i] = w_new, gap_new
                n_refined += 1
                continue
            w_new, gap_new, _ok = refine_mle_slsqp(P, N[i], W[i], gap_tol=gap_tol)
            # Never worsen the dual gap; keep EM/Newton if refine fails
            if gap_new < gaps[i] - 1e-16:
                W[i] = w_new
                gaps[i] = gap_new
                n_refined += 1
        accepted = gaps <= gap_tol
        if strict:
            # Recompute the certificate independently of the polisher result.
            gaps = dual_gap(P, N, W)
            accepted = (np.isfinite(gaps) & (gaps <= gap_tol) & np.isfinite(W).all(1)
                        & (W.min(1) >= 0) & (np.abs(W.sum(1) - 1) <= 1e-12))
            if not accepted.all():
                raise RuntimeError('MLE feasibility/dual-gap acceptance failed; grid not written')
    return W, gaps, accepted, n_refined


def work_point(args):
    P, w, K, Z, seed, do_mle = args
    rng = np.random.default_rng(seed)
    U = dirs(K, Z, 36000 if K == 3 else 40000)
    q = point_q(P, w, Z); cTV, d0TV = c_tv_and_delta0tv(P, w, Z, U); rho_w = 1 / w.min()
    deltas = np.linspace(0.02, q['delta_max'] * 0.98, 12)
    rows = []; viol = []
    for d in deltas:
        om = {k: modulus(P, w, Z, d, k, U)[0] for k in ('TV', 'H', 'chi')}
        a1 = d * np.sqrt(q['lam_min']); tl = d * cTV / 2
        hlo = om['chi'] / (np.sqrt(2) * (1 + np.sqrt(rho_w))); hhi = om['chi'] / np.sqrt(2)
        row = dict(delta=d, omega_TV=om['TV'], omega_H=om['H'], omega_chi=om['chi'], a1_pred=a1, tv_line=tl, h_lo_w=hlo, h_hi=hhi)
        if om['chi'] < a1 - 1e-9: viol.append('chi<A1')
        if d <= q['delta0'] and abs(om['chi'] - a1) > 1e-6 * max(1, om['chi']): viol.append('chi!=A1')
        if d > q['delta0'] and om['chi'] > d * np.sqrt(q['lam_max']) + 1e-9: viol.append('chi>lam_max')
        if om['TV'] < tl - 1e-9: viol.append('TV<line')
        if d <= d0TV and abs(om['TV'] - tl) > 1e-6 * max(1, om['TV']): viol.append('TV!=line')
        if not (hlo - 1e-9 <= om['H'] <= hhi + 1e-9): viol.append('H_sandwich')
        rows.append(row)
    for i in range(1, len(rows)):
        if rows[i]['omega_chi'] / rows[i]['delta'] < rows[i - 1]['omega_chi'] / rows[i - 1]['delta'] - 1e-9: viol.append('chi/delta_monotone')
        if rows[i]['omega_TV'] / rows[i]['delta'] < rows[i - 1]['omega_TV'] / rows[i - 1]['delta'] - 1e-9: viol.append('TV/delta_monotone')
    base = dict(w=json.dumps([round(x, 6) for x in w]), lam_min=q['lam_min'], lam_max=q['lam_max'], delta0=q['delta0'], delta0_TV=d0TV, r=q['r'],
                delta_max=q['delta_max'], c_TV=cTV, rho=q['rho'], rho_w=rho_w, lam_star=q['lam_star'], tau=q['tau'], n_viol=len(viol), viol=';'.join(sorted(set(viol))))
    for row in rows: row.update(base)
    mle_rows = []
    if do_mle:
        I = q['I']; lam = q['lam_min']; pw = w @ P; ZR = Z.T @ (P / pw)
        for n in N_LIST:
            N = rng.multinomial(n, pw, size=REPS)
            What, gaps, accepted, n_ref = mle_vectorized(P, N, np.tile(np.ones(K) / K, (REPS, 1)))
            if not accepted.all():
                raise RuntimeError('Unaccepted MLE replication; refusing to aggregate risk')
            x2 = ((What - w) ** 2).sum(1); risk = x2.mean(); se = x2.std() / np.sqrt(REPS)
            Ghat = np.einsum('rm,im,jm->rij', N, ZR, ZR); E1 = (np.linalg.norm(Ghat - n * I, 2, axis=(1, 2)) <= n * lam / 2).mean()
            bnd = (What.min(1) < 1e-7).mean()  # numerically on a face: min_k w_hat < 1e-7
            mle_rows.append(dict(w=base['w'], n=n, risk=risk, se=se, tau_over_n=q['tau'] / n, ratio=n * risk / q['tau'], P_E1=E1, boundary_frac=bnd,
                                 lam_min=lam, lam_max=q['lam_max'], rho_w=rho_w, tau=q['tau'],
                                 dual_gap_max=float(gaps.max()), dual_gap_med=float(np.median(gaps)),
                                 dual_gap_accept_frac=float(accepted.mean()), n_refined=int(n_ref)))
    return rows, mle_rows


if __name__ == '__main__':
    tid = int(sys.argv[1]); K, fam, par = TASKS[tid]
    rng = np.random.default_rng(20260919 + tid); Z = tangent_basis(K)
    if fam == 'F1': P = family_F1(K, 2 * K, par, rng)
    elif fam == 'F2': P = family_F2(K, 2 * K, 0.3, np.full(K, par), rng)
    else: P, _ = family_F3(K, par, 6 * par + 3, 4001)
    W = simplex_lattice(K, 12 if K == 3 else 8)
    if os.environ.get('SMOKE'): W = W[::max(1, len(W) // 3)][:3]; REPS = 200
    do_mle = fam != 'F3' or par <= 2.0
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', 32))) as pool:
        res = pool.map(work_point, [(P, w, K, Z, 1000 * tid + i, do_mle) for i, w in enumerate(W)])
    tag = f'{K}_{fam}_{par}'
    rows = [r for rr, _ in res for r in rr]; mle = [r for _, mm in res for r in mm]
    with open(os.path.join(OUT, f'grid_{tag}.csv'), 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    if mle:
        with open(os.path.join(OUT, f'grid_{tag}_mle.csv'), 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(mle[0].keys())); wtr.writeheader(); wtr.writerows(mle)
    np.save(os.path.join(OUT, f'P_{tag}.npy'), P)
    nv = sum(r['n_viol'] for rr, _ in res for r in rr[:1])
    print(f'task {tid} {tag}: {len(W)} lattice points, {len(rows)} modulus rows, {len(mle)} mle rows, violations {nv}, {time.time() - t0:.0f} s')
