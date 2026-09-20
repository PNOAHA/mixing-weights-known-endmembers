"""Machine verification for A4 (general endmembers: non-asymptotic MLE risk bound). Items (a)-(g).
Run: python a4_verify.py   (miniconda python; numpy, sympy, scipy)
All inequalities are checked with explicit tolerances; MC items report the numbers (not pre-filled anywhere).
"""
import itertools
import numpy as np
import sympy as sp

rng = np.random.default_rng(20260918)  # seed unchanged; 09-20: constants revised to q = sqrt(2kappa+1)
results, log = {}, []


def tangent_basis(K):
    A = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(A)
    return U[:, :K - 1]


# ---------------------------------------------------------------- (a) elementary inequalities + Chernoff algebra
ok_a = True
y = np.concatenate([np.linspace(-0.999, 0, 4000), np.linspace(0, 60, 6000)[1:]])
lhs = np.log1p(y); rhs = y - y ** 2 / (2 * (1 + np.maximum(y, 0)))
if np.any(lhs > rhs + 1e-12): ok_a = False; log.append("(a) log(1+y) <= y - y^2/(2(1+y+)) fails")
u = np.linspace(0, 2.999, 5000)
psi = np.expm1(u) - u
if np.any(psi > u ** 2 / (2 * (1 - u / 3)) + 1e-12): ok_a = False; log.append("(a) psi(u) <= u^2/(2(1-u/3)) fails")
D = np.linspace(0, 1, 1001)
if np.any((1 + D) ** 2 > 1 + 3 * D + 1e-12): ok_a = False; log.append("(a) (1+D)^2 <= 1+3D fails")
x = np.linspace(0, 200, 20001)
if np.any(x * np.exp(-x) > (2 / np.e) * np.exp(-x / 2) + 1e-12): ok_a = False; log.append("(a) x e^-x <= (2/e) e^{-x/2} fails")
yy = np.linspace(-0.99, 60, 5000)
if np.any(np.abs(1 / (1 + yy) - (1 - yy + yy ** 2 / (1 + yy))) > 1e-9): ok_a = False; log.append("(a) 1/(1+y) identity fails")
# Chernoff exponent: -s t + n sigma^2 s^2/(2(1 - s M/3)) at s = t/(n sigma^2 + M t/3) equals -t^2/(2(n sigma^2 + M t/3)); and s M < 3
t, n_, sg, M = sp.symbols('t n sigma M', positive=True)
s = t / (n_ * sg ** 2 + M * t / 3)
expr = -s * t + n_ * sg ** 2 * s ** 2 / (2 * (1 - s * M / 3))
target = -t ** 2 / (2 * (n_ * sg ** 2 + M * t / 3))
if sp.simplify(expr - target) != 0: ok_a = False; log.append("(a) Chernoff exponent algebra fails")
# MGF lemma: (e^{sx}-1-sx)/x^2 is non-decreasing in x (grid, several s)
for s_ in (0.1, 1.0, 3.0):
    xx = np.linspace(-30, 30, 60001); xx = xx[np.abs(xx) > 1e-6]
    f = (np.exp(s_ * xx) - 1 - s_ * xx) / xx ** 2
    if np.any(np.diff(f) < -1e-12 * np.maximum(1, np.abs(f[1:]))): ok_a = False; log.append("(a) MGF ratio not monotone")
# s M < 3  <=>  M t < 3 n sigma^2 + M t : always true
if sp.simplify((3 - s * M) - 3 * n_ * sg ** 2 / (n_ * sg ** 2 + M * t / 3)) != 0: ok_a = False; log.append("(a) sM<3 algebra fails")
# h(y) derivative and k(y) derivative used in Prop A4.3 proof
Y = sp.symbols('y', real=True)
h = Y - Y ** 2 / (2 * (1 + Y)) - sp.log(1 + Y); k = Y - Y ** 2 / 2 - sp.log(1 + Y)
if sp.simplify(sp.diff(h, Y) - Y ** 2 / (2 * (1 + Y) ** 2)) != 0: ok_a = False; log.append("(a) h' formula")
if sp.simplify(sp.diff(k, Y) + Y ** 2 / (1 + Y)) != 0: ok_a = False; log.append("(a) k' formula")
# psi bound: k! >= 2*3^(k-2) for k>=2
if any(sp.factorial(kk) < 2 * 3 ** (kk - 2) for kk in range(2, 40)): ok_a = False; log.append("(a) k! >= 2 3^(k-2)")
results["(a) elementary inequalities + Bernstein/Chernoff algebra"] = ok_a

# ---------------------------------------------------------------- (b) score polytope bounds (Lemma A4.2(i))
ok_b = True
for _ in range(300):
    K = int(rng.integers(2, 8)); w = 0.85 * rng.dirichlet(np.ones(K)) + 0.15 / K; rho = 1 / w.min()
    # random points of P = {r>=0, sum w_k r_k = 1}: r = c/w with c in simplex
    C = rng.dirichlet(np.ones(K) * rng.uniform(0.2, 3), size=2000); R = C / w
    if np.any(np.linalg.norm(R, axis=1) > rho + 1e-10): ok_b = False; log.append("(b) ||r|| > rho")
    if np.any(np.linalg.norm(R - 1, axis=1) > rho + 1e-10): ok_b = False; log.append("(b) ||r-1|| > rho")
    verts = np.eye(K) / w[:, None]
    vmax = (np.linalg.norm(verts - 1, axis=1) ** 2).max()
    if abs(vmax - (rho ** 2 - 2 * rho + K)) > 1e-8: ok_b = False; log.append("(b) vertex formula")
    if rho < K - 1e-12: ok_b = False; log.append("(b) rho >= K")
    Zb = tangent_basis(K); yy = rng.normal(size=(500, K - 1)); yy *= (rng.uniform(0, 1, size=(500, 1)) / (2 * rho)) / np.linalg.norm(yy, axis=1, keepdims=True)
    if np.any((w + yy @ Zb.T).min(1) < 0.5 * w.min() - 1e-12): ok_b = False; log.append("(b) A4.2(ii) min coordinate fails")
results["(b) A4.2(i): ||r||, ||r-1|| <= rho_w on the score polytope"] = ok_b

# ---------------------------------------------------------------- (c) vector fourth moment identity (Lemma A4.1(c)), exact enumeration
ok_c = True
for trial in range(6):
    d = int(rng.integers(1, 4)); m = int(rng.integers(2, 5))
    pts = rng.normal(size=(m, d)); pr = rng.dirichlet(np.ones(m)); pts -= (pr[:, None] * pts).sum(0)  # mean zero
    E2 = (pr * (pts ** 2).sum(1)).sum(); E4 = (pr * (pts ** 2).sum(1) ** 2).sum()
    Sig = (pts * pr[:, None]).T @ pts; trS2 = np.trace(Sig @ Sig)
    for n in (2, 3, 4):
        tot = 0.0
        for idx in itertools.product(range(m), repeat=n):
            p = np.prod(pr[list(idx)]); ssum = pts[list(idx)].sum(0); tot += p * (ssum @ ssum) ** 2
        formula = n * E4 + n * (n - 1) * (E2 ** 2 + 2 * trS2)
        if abs(tot - formula) > 1e-9 * max(1, abs(tot)): ok_c = False; log.append(f"(c) exact identity fails n={n}")
        if tot > n * E4 + 3 * n * n * E2 ** 2 + 1e-9: ok_c = False; log.append("(c) inequality fails")
    if trS2 > E2 ** 2 + 1e-12: ok_c = False; log.append("(c) tr(S^2) <= (tr S)^2 fails")
results["(c) A4.1(c): fourth-moment identity and bound (exact enumeration)"] = ok_c

# ---------------------------------------------------------------- (d) 1/4-net (Lemma A4.1(b))
ok_d = True
for d in (2, 3, 4):
    cand = rng.normal(size=(200000, d)); cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    net = []
    for c in cand:  # greedy maximal 1/4-separated subset of the candidates
        if all(np.linalg.norm(c - u) > 0.25 for u in net): net.append(c)
    net = np.array(net)
    if len(net) > 9 ** d: ok_d = False; log.append(f"(d) net size {len(net)} > 9^{d}")
    test = rng.normal(size=(5000, d)); test /= np.linalg.norm(test, axis=1, keepdims=True)
    dist = np.linalg.norm(test[:, None, :] - net[None, :, :], axis=2).min(1)
    if dist.max() > 0.25 + 0.02: ok_d = False; log.append(f"(d) covering radius {dist.max():.3f} (finite candidate set)")
    for _ in range(200):
        A = rng.normal(size=(d, d)); E = A + A.T
        if np.linalg.norm(E, 2) > 2 * np.abs(np.einsum('ij,jk,ik->i', net, E, net)).max() + 1e-9: ok_d = False; log.append("(d) ||E|| <= 2 max |u'Eu| fails")
    log.append(f"(d) d={d}: |net|={len(net)} (bound {9 ** d}), covering radius {dist.max():.3f}")
results["(d) A4.1(b): 1/4-net size and operator-norm bound"] = ok_d


# ---------------------------------------------------------------- (e) pathwise checks + Monte Carlo (Prop A4.3, Thm A4.4)
def family(K, m, theta, rng):
    base = rng.dirichlet(np.ones(m) * 0.7, size=K)
    return (1 - theta) * base + theta / m  # K x m pmfs, all cells positive


def gram(P, w):
    pw = w @ P; return (P / pw) @ P.T


def mle_vectorized(P, N, W0, iters_em=3000, iters_newton=25):
    """P: K x m; N: reps x m counts; returns reps x K MLE. EM then tangent-space Newton with backtracking."""
    n = N.sum(1, keepdims=True); W = W0.copy(); K = P.shape[0]; Z = tangent_basis(K)
    for _ in range(iters_em):
        pw = W @ P
        W = W * ((N / pw) @ P.T) / n
    def loglik(W):
        return (N * np.log(np.maximum(W @ P, 1e-300))).sum(1)
    for _ in range(iters_newton):
        pw = W @ P                                  # reps x m
        R = P[None, :, :] / pw[:, None, :]          # reps x K x m
        grad = np.einsum('rm,rkm->rk', N, R) @ Z    # reps x (K-1)
        H = np.einsum('rm,rkm,rlm->rkl', N, R, R); H = np.einsum('ki,rkl,lj->rij', Z, H, Z)
        step = np.linalg.solve(H, grad[:, :, None])[:, :, 0] @ Z.T   # reps x K, direction in T
        alpha = np.ones((W.shape[0], 1)); ll0 = loglik(W)
        for _bt in range(30):
            Wn = W + alpha * step
            ok = (Wn.min(1) > 0) & (loglik(Wn) >= ll0 - 1e-12)
            alpha[~ok] *= 0.5
            if ok.all(): break
        Wn = W + alpha * step; W = np.where((Wn.min(1) > 0)[:, None], Wn, W)
    return W


ok_e = True
mc_table = []
for (K, m, w) in [(2, 2, np.array([0.5, 0.5])), (3, 6, np.array([0.5, 0.3, 0.2])), (4, 8, np.array([0.4, 0.3, 0.2, 0.1]))]:
    P = np.array([[0.9, 0.1], [0.1, 0.9]]) if K == 2 else family(K, m, 0.3, rng); Z = tangent_basis(K)
    G = gram(P, w); I = Z.T @ G @ Z; ev = np.linalg.eigvalsh(I); lam, Lam = ev[0], ev[-1]; kap = Lam / lam
    tau = np.trace(np.linalg.inv(I)); rho = 1 / w.min(); T1 = np.trace(I)
    if abs((G @ w - 1).max()) > 1e-10: ok_e = False; log.append("(e) G w = 1 fails")
    q = np.sqrt(2 * kap + 1)  # 09-20: constants in q = sqrt(2 kappa + 1) instead of 2 kappa + 1 (remainder estimated in the Ghat-norm)
    r1 = 1 / (8 * rho * q); eta1 = rho * r1; c1 = 1.45 * rho * q
    pw = w @ P; Rmat = P / pw  # K x m: r(x_j)
    n_list = (400, 4_000_000, 10_000_000) if K == 2 else (100, 400, 1600, 6400, 20_000_000)
    for n in n_list:
        reps = 4000 if K == 2 else (3000 if n < 10 ** 7 else 1500)
        N = rng.multinomial(n, pw, size=reps)
        W0 = np.tile(np.ones(K) / K, (reps, 1))
        What = mle_vectorized(P, N, W0)
        yhat = (What - w) @ Z; x = np.linalg.norm(yhat, axis=1)
        ZR = Z.T @ Rmat                                   # (K-1) x m
        xi = N @ ZR.T                                     # reps x (K-1)
        Ghat = np.einsum('rm,im,jm->rij', N, ZR, ZR)      # reps x (K-1) x (K-1)
        En = Ghat - n * I; En_op = np.linalg.norm(En, 2, axis=(1, 2))
        E1 = En_op <= n * lam / 2
        g0 = np.linalg.norm(xi @ np.linalg.inv(n * I).T, axis=1)
        # E||E_n||_F^2 <= n rho^2 T1 ; E g0^4 <= m4 ; ||zeta_i|| <= rho/(n lam)
        EnF2 = (np.linalg.norm(En, axis=(1, 2)) ** 2)
        if EnF2.mean() - 3 * EnF2.std() / np.sqrt(reps) > n * rho ** 2 * T1: ok_e = False; log.append(f"(e) E||E_n||_F^2 bound fails K={K} n={n}")
        m4 = 3 * tau ** 2 / n ** 2 + rho ** 2 * tau / (n ** 3 * lam ** 2)
        if (g0 ** 4).mean() - 3 * (g0 ** 4).std() / np.sqrt(reps) > m4: ok_e = False; log.append(f"(e) E g0^4 <= m4 fails K={K} n={n}")
        zeta = (np.linalg.inv(n * I) @ (ZR - 0)).T  # m x (K-1): (nI)^{-1} Z^T r(x_j); Z^T 1 = 0 so r-1 and r agree
        if np.linalg.norm(zeta, axis=1).max() > rho / (n * lam) + 1e-12: ok_e = False; log.append(f"(e) ||zeta|| bound fails K={K} n={n}")
        # tail one: |X_i| <= rho^2 and E X_i^2 <= rho^2 Lam for net directions; P(E1^c) MC <= Bernstein bound
        Uc = rng.normal(size=(50, K - 1)); Uc /= np.linalg.norm(Uc, axis=1, keepdims=True)
        proj = (Uc @ ZR) ** 2                          # 50 x m : (u'Z'r(x_j))^2
        uIu = np.einsum('ui,ij,uj->u', Uc, I, Uc)
        Xc = proj - uIu[:, None]
        if np.abs(Xc).max() > rho ** 2 + 1e-9: ok_e = False; log.append(f"(e) |X_i| <= rho^2 fails K={K}")
        if np.any((Xc ** 2 * pw[None, :]).sum(1) > rho ** 2 * Lam + 1e-9): ok_e = False; log.append(f"(e) E X_i^2 <= rho^2 Lam fails K={K}")
        P1 = 2 * 9 ** (K - 1) * np.exp(-n * lam ** 2 / (32 * rho ** 2 * (Lam + lam / 12)))
        if (1 - E1.mean()) - 3 * np.sqrt(E1.mean() * (1 - E1.mean()) / reps + 1e-12) > P1: ok_e = False; log.append(f"(e) P(E1^c) exceeds Bernstein bound K={K} n={n}")
        # exact moment identities (MC)
        xi_cov = xi.T @ xi / reps
        se_cov = np.sqrt(((xi[:, :, None] * xi[:, None, :]) ** 2).mean(0) / reps)
        if np.any(np.abs(xi_cov - n * I) > 5 * se_cov + 1e-9): ok_e = False; log.append(f"(e) E xi xi' = nI fails K={K} n={n}")
        m2 = (g0 ** 2).mean(); se2 = (g0 ** 2).std() / np.sqrt(reps)
        if abs(m2 - tau / n) > 5 * se2: ok_e = False; log.append(f"(e) E g0^2 = tau/n fails K={K} n={n}: {m2:.3e} vs {tau / n:.3e}")
        # pathwise, interior maximisers only: gradient ~ 0 check
        pwh = What @ P; Rh = P[None] / pwh[:, None, :]
        grad = np.einsum('rm,rkm->rk', N, Rh) @ Z
        interior = What.min(1) > 1e-7
        gn = np.linalg.norm(grad, axis=1) / n
        conv = interior & (gn < 1e-8)
        # FOC identity Ghat yhat = xi + rhohat, and the bound on rhohat, on conv & E1 & x<r1
        Yj = yhat @ ZR                                    # reps x m : Y_j(yhat)
        rhohat = np.einsum('rm,im->ri', N * Yj ** 2 / (1 + Yj), ZR)
        resid = np.linalg.norm(np.einsum('rij,rj->ri', Ghat, yhat) - xi - rhohat, axis=1) / np.maximum(np.linalg.norm(xi, axis=1), 1)
        sel = conv & E1 & (x < r1)
        lmax = np.linalg.eigvalsh(Ghat)[:, -1]; lmin = np.linalg.eigvalsh(Ghat)[:, 0]
        if sel.any():
            if resid[sel].max() > 1e-7: ok_e = False; log.append(f"(e) FOC residual {resid[sel].max():.2e} K={K} n={n}")
            if np.any(np.linalg.norm(rhohat[sel], axis=1) > rho * lmax[sel] * x[sel] ** 2 / (1 - eta1) + 1e-9): ok_e = False; log.append(f"(e) rhohat bound fails K={K} n={n}")
            # 09-20: Ghat-norm estimate  ||Ghat^{-1} rhohat|| <= rho sqrt(kappa(Ghat)) ||yhat||^2/(1-eta1)   (the source of (2kappa+1) -> sqrt(2kappa+1))
            Ginv_rho = np.linalg.solve(Ghat[sel], rhohat[sel][:, :, None])[:, :, 0]
            if np.any(np.linalg.norm(Ginv_rho, axis=1) > rho * np.sqrt(lmax[sel] / lmin[sel]) * x[sel] ** 2 / (1 - eta1) + 1e-9): ok_e = False; log.append(f"(e) Ghat^-1 rhohat bound fails K={K} n={n}")
            g = np.linalg.norm(np.linalg.solve(Ghat[sel], xi[sel][:, :, None])[:, :, 0], axis=1)
            if np.any(x[sel] > g * (1 + c1 * g) + 1e-12): ok_e = False; log.append(f"(e) x <= g(1+c1 g) fails K={K} n={n}")
            Dd = En_op[sel] / lmin[sel]
            if np.any(g > g0[sel] * (1 + Dd) + 1e-12): ok_e = False; log.append(f"(e) g <= g0(1+D) fails K={K} n={n}")
            if np.any(x[sel] ** 2 > g0[sel] ** 2 + 3 * g0[sel] ** 2 * Dd + 16 * c1 * g0[sel] ** 3 + 16 * c1 ** 2 * g0[sel] ** 4 + 1e-12): ok_e = False; log.append(f"(e) squared bound fails K={K} n={n}")
        # localisation implication on E1 & x>=r1 (all maximisers, boundary included)
        sel2 = E1 & (x >= r1)
        t2 = n * lam / (35 * rho * q)
        if sel2.any() and np.any(np.linalg.norm(xi[sel2], axis=1) < t2 - 1e-9): ok_e = False; log.append(f"(e) localisation implication fails K={K} n={n}")
        # risk vs bound (A4.1)
        risk = (x ** 2).mean(); se_r = (x ** 2).std() / np.sqrt(reps)
        th = rho ** 2 / (n * lam); un = q * np.sqrt((K - 1) * th)
        Mw = 2500 * (K - 1) * kap * (2 * kap + 1) * rho ** 2; Nw = Mw / lam
        bound = tau / n * (1 + 30 * np.sqrt(3 + th) * un + 34 * (3 + th) * un ** 2) + 8 * 9 ** (K - 1) * np.exp(-n / Nw)
        if risk > bound + 3 * se_r: ok_e = False; log.append(f"(e) risk exceeds bound K={K} n={n}")
        n0 = 2 * Mw / lam * np.log(12 * 9 ** (K - 1) * Mw)
        mc_table.append((K, n, risk, se_r, tau / n, n * risk / tau, bound, E1.mean(), int(sel.sum()), conv.mean(), n0))
        log.append(f"(e) K={K} n={n}: risk={risk:.3e}±{se_r:.1e} tau/n={tau / n:.3e} ratio n*risk/tau={n * risk / tau:.3f} bound(A4.1)={bound:.3e} P(E1)={E1.mean():.3f} branch(ii) samples={int(sel.sum())}/{reps} conv={conv.mean():.3f} n0={n0:.2e} nontrivial={bound < 2 * tau / n}")
    log.append(f"(e) K={K}: lam={lam:.4f} Lam={Lam:.4f} kappa={kap:.3f} q={q:.3f} tau={tau:.4f} rho_w={rho:.2f} r1={r1:.2e} c1={c1:.1f}")
results["(e) A4.3/A4.4 pathwise identities, moment identities, risk <= (A4.1) (MC, K=3,4)"] = ok_e

# ---------------------------------------------------------------- (f) numerical constants in A4.3, A4.4, A4.5 (09-20 version: q = sqrt(2kappa+1))
ok_f = True
R = sp.Rational
# eta1 = 1/(8 q) <= 1/(8 sqrt3) < 1/13 (kappa >= 1);  1/(1-1/13) = 13/12
if not (1 / (8 * sp.sqrt(3)) < R(1, 13)): ok_f = False; log.append("(f) eta1 < 1/13")
if not (1 / (1 - R(1, 13)) == R(13, 12)): ok_f = False; log.append("(f) 1/(1-eta1)")
# Prop A4.3(ii): C r1 <= (13/12)/8 = 13/96 ; 1/(1-13/96) = 96/83 ; (96/83)^2 * 13/12 <= 1.45
if not (R(13, 12) / 8 == R(13, 96)): ok_f = False; log.append("(f) C r1")
if not (1 / (1 - R(13, 96)) == R(96, 83)): ok_f = False; log.append("(f) 96/83")
c1_exact = R(96, 83) ** 2 * R(13, 12)
if not (c1_exact <= R(145, 100)): ok_f = False; log.append("(f) c1 constant")
log.append(f"(f) c1 exact constant (96/83)^2*13/12 = {float(c1_exact):.4f} <= 1.45")
# Prop A4.3(i): n lam r1/(4(1+eta1)) >= (13/14) * n lam/(32 rho q) >= n lam/(35 rho q)
if not (1 / (1 + R(1, 13)) == R(13, 14) and R(13, 14) / 32 >= R(1, 35)): ok_f = False; log.append("(f) t2 constant")
# Thm A4.4 main-term constants: 6 + 16*1.45 = 29.2 <= 30 ; 16*1.45^2 = 33.64 <= 34 ; sqrt(kappa) <= q trivially
if not (6 + 16 * 1.45 <= 30 and 16 * 1.45 ** 2 <= 34): ok_f = False; log.append("(f) 30/34")
# P_1: 35 kappa >= 32 (kappa + 1/12) for kappa >= 1 ; 35 kappa rho^2/lam <= N_w = 2500 (K-1) kappa (2kappa+1) rho^2/lam
kk = np.linspace(1, 1000, 100000)
if np.any(35 * kk < 32 * (kk + 1 / 12)): ok_f = False; log.append("(f) P1 constant")
if np.any(35 * kk > 2500 * 1 * kk * (2 * kk + 1)): ok_f = False; log.append("(f) 35 kappa <= N_w/rho^2 lam")
# P_2: rho t2/(3 sqrt(K-1)) = n lam/(105 q sqrt(K-1)) <= n Lam/181 (105 sqrt3 > 181) ; 2*35^2*182/181 <= 2500
if not (105 * sp.sqrt(3) > 181 and R(2 * 35 ** 2 * 182, 181) <= 2500): ok_f = False; log.append("(f) P2 algebra")
log.append(f"(f) P2 exact constant 2*35^2*182/181 = {float(R(2 * 35 ** 2 * 182, 181)):.1f} <= 2500")
# P_2 exponent on a grid: t2^2/(2(K-1)(n Lam + rho t2/(3 sqrt(K-1)))) >= n lam/(2500 (K-1) kappa (2kappa+1) rho^2); scale lam=1, n=1
for K in range(2, 12):
    for kap in np.geomspace(1, 1e4, 200):
        for rho in np.geomspace(K, 1e5, 30):
            q_ = np.sqrt(2 * kap + 1); t2 = 1 / (35 * rho * q_)
            lhs_ = t2 ** 2 / (2 * (K - 1) * (kap + rho * t2 / (3 * np.sqrt(K - 1))))
            rhs_ = 1 / (2500 * (K - 1) * kap * (2 * kap + 1) * rho ** 2)
            if lhs_ < rhs_ * (1 - 1e-12): ok_f = False; log.append(f"(f) P2 exponent fails K={K} kap={kap:.2f} rho={rho:.1f}"); break
# 4(9^{K-1} + K-1) <= 8 * 9^{K-1}
if any(4 * (9 ** (K - 1) + K - 1) > 8 * 9 ** (K - 1) for K in range(2, 30)): ok_f = False; log.append("(f) 8*9^(K-1)")
# Cor A4.5 (i): u <= 1/120 => correction < 1/2  (theta = u^2/((2kappa+1)(K-1)) <= u^2/3)
u = 1 / 120; th = u ** 2 / 3
corr = 30 * np.sqrt(3 + th) * u + 34 * (3 + th) * u ** 2
if not corr < 0.5: ok_f = False; log.append(f"(f) correction at u=1/120 is {corr:.4f}")
log.append(f"(f) correction term at u=1/120: {corr:.4f} (bracket {1 + corr:.4f})")
# monotone: correction increasing in u, so u <= 1/120 suffices
uu = np.linspace(0, 1 / 120, 1000); thh = uu ** 2 / 3
if np.any(np.diff(30 * np.sqrt(3 + thh) * uu + 34 * (3 + thh) * uu ** 2) < 0): ok_f = False; log.append("(f) correction not monotone")
# Cor A4.5: n0 >= 14400 (K-1)(2kappa+1) rho^2/lam  (scale lam = 1) ; 2 log(12*9*2500*9) > 29 so n0 > 29 M_w/lam >= 32 rho^2/(9 lam)
if not (2 * np.log(12 * 9 * 2500 * 9) > 29 and 32 / np.e < 12): ok_f = False; log.append("(f) 32/e, 29 M_w")
for K in range(2, 12):
    for kap in np.geomspace(1, 1e4, 100):
        for rho in np.geomspace(K, 1e5, 20):
            Mw = 2500 * (K - 1) * kap * (2 * kap + 1) * rho ** 2
            n0 = 2 * Mw * np.log(12 * 9 ** (K - 1) * Mw)
            if n0 < 29 * Mw: ok_f = False; log.append("(f) n0 > 29 M_w fails"); break
            if n0 < 14400 * (K - 1) * (2 * kap + 1) * rho ** 2: ok_f = False; log.append(f"(f) n0 does not imply (i) K={K}"); break
            if n0 <= 32 * rho ** 2 / 9: ok_f = False; log.append("(f) n0 does not imply 32 rho^2/(9 lam)"); break
# Cor A4.5 (ii): n >= 2 N_w log(32 9^{K-1} N_w/(e tau)) => 8 9^{K-1} e^{-n/N_w} <= tau/(2n); and with tau >= 1/lam, N_w/tau <= M_w
for _ in range(2000):
    K = int(rng.integers(2, 8)); Nw = 10 ** rng.uniform(1, 8); tau = 10 ** rng.uniform(-4, 2)
    nmin = 2 * Nw * np.log(32 * 9 ** (K - 1) * Nw / (np.e * tau))
    if nmin <= 0: continue
    for n in (nmin, 1.3 * nmin, 5 * nmin):
        if 8 * 9 ** (K - 1) * np.exp(-n / Nw) > tau / (2 * n) * (1 + 1e-9): ok_f = False; log.append("(f) tail threshold fails"); break
# Cor A4.6: n >= 32 rho^2/(9 lam) => delta_n <= 1/(2 rho); r(w) >= 1/rho ; 2*(3/2) = 3
rr = np.geomspace(2, 1e4, 200); lam_ = np.geomspace(1e-4, 10, 50)
for rho in rr:
    for lam in lam_:
        n = 32 * rho ** 2 / (9 * lam)
        if np.sqrt(8 / (9 * n * lam)) > 1 / (2 * rho) + 1e-12: ok_f = False; log.append("(f) delta_n <= 1/(2rho) fails"); break
for K in range(2, 30):
    for wmin in np.linspace(1e-4, 1 / K, 50):
        if np.sqrt(K / (K - 1)) * wmin < 1 / (1 / wmin) - 1e-15: ok_f = False; log.append("(f) r(w) >= 1/rho")
results["(f) numerical constants of A4.3-A4.6"] = ok_f

# ---------------------------------------------------------------- (g) Loewner continuity of G (Lemma A4.2(iii))
ok_g = True
for _ in range(300):
    K = int(rng.integers(2, 7)); m = int(rng.integers(K + 1, K + 8)); P = family(K, m, rng.uniform(0, 0.9), rng)
    w = 0.85 * rng.dirichlet(np.ones(K)) + 0.15 / K; rho = 1 / w.min(); Z = tangent_basis(K)
    r = rng.uniform(0, 0.99) / rho
    v = rng.normal(size=K); v -= v.mean(); v *= r * rng.uniform(0, 1) / np.linalg.norm(v)
    w2 = w + v
    if w2.min() <= 0: continue
    G, G2 = gram(P, w), gram(P, w2)
    if np.linalg.eigvalsh(G2 - G / (1 + rho * r)).min() < -1e-9: ok_g = False; log.append("(g) lower Loewner fails")
    if np.linalg.eigvalsh(G / (1 - rho * r) - G2).min() < -1e-9: ok_g = False; log.append("(g) upper Loewner fails")
    if 1 / w2.min() > rho / (1 - rho * r) + 1e-9: ok_g = False; log.append("(g) rho_w'' bound fails")
    # consequences at r = 1/(2 rho): lam(w'') >= 2/3 lam etc. (check when the sampled r <= 1/(2rho))
    if rho * r <= 0.5:
        I1, I2 = Z.T @ G @ Z, Z.T @ G2 @ Z
        e1, e2 = np.linalg.eigvalsh(I1), np.linalg.eigvalsh(I2)
        if e2[0] < 2 / 3 * e1[0] - 1e-9 or e2[-1] > 2 * e1[-1] + 1e-9: ok_g = False; log.append("(g) eigen consequences fail")
        if np.trace(np.linalg.inv(I2)) > 1.5 * np.trace(np.linalg.inv(I1)) + 1e-9: ok_g = False; log.append("(g) tau consequence fails")
results["(g) A4.2(iii): Loewner sandwich of G(w'') and its consequences"] = ok_g

# ---------------------------------------------------------------- report
print("=== A4 machine verification ===")
for k, v in results.items(): print(("PASS " if v else "FAIL ") + k)
print(f"{sum(results.values())}/{len(results)} passed")
print("--- log ---")
for line in log: print(line)
print("--- MC table: K, n, risk, se, tau/n, n*risk/tau, bound(A4.1), P(E1), branch(ii) samples, conv, n0 ---")
for row in mc_table: print(", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in row))
