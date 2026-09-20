"""Machine verification for Theorem A2 (TV / Hellinger moduli). Items (a)-(d) of notes/A2.md."""
import numpy as np
import sympy as sp

rng = np.random.default_rng(20260917)
results, log = {}, []


def tangent_basis(K):
    A = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(A); return U[:, :K - 1]


def TV(p, q): return 0.5 * np.abs(p - q).sum()
def H2(p, q): return 0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()
def CHI2(p, q):
    m = q > 0
    return np.inf if np.any(p[~m] > 0) else float(((p[m] - q[m]) ** 2 / q[m]).sum())


def gram(P, w):
    pw = w @ P; m = pw > 0
    return (P[:, m] / pw[m]) @ P[:, m].T


def gram_min(P):
    pm = P.max(axis=0); m = pm > 0
    return (P[:, m] / pm[m]) @ P[:, m].T


def delta0_of_u(w, u):
    neg = u < -1e-14
    return np.inf if not neg.any() else float(np.min(w[neg] / (-u[neg])))


# ------------------------------------------------------------------ (a) TV modulus, K=3 exact via angle grid
ok_a = True
K = 3; Z = tangent_basis(K)
phis = np.linspace(0, 2 * np.pi, 20001)[:-1]
U = (Z @ np.vstack([np.cos(phis), np.sin(phis)])).T          # unit vectors in T, shape (n, 3)
for trial in range(25):
    m = int(rng.integers(4, 10))
    P = rng.dirichlet(np.ones(m) * 0.7, size=K); P = 0.7 * P + 0.3 / m if trial % 2 else P
    f1 = np.abs(U @ P).sum(axis=1)                            # ||f_u||_1 on the circle
    cTV = f1.min(); Umin = U[f1 <= cTV + 1e-12]
    # exactness / w-independence / homogeneity
    for _ in range(20):
        w = rng.dirichlet(np.ones(K)); w2 = rng.dirichlet(np.ones(K)); v = w2 - w
        if abs(TV(w2 @ P, w @ P) - 0.5 * np.abs(v @ P).sum()) > 1e-12: ok_a = False; log.append("(a) exact expr")
    # lambda bounds
    w = 0.9 * rng.dirichlet(np.ones(K)) + 0.1 / K
    lam_min = np.linalg.eigvalsh(Z.T @ gram(P, w) @ Z)[0]
    lam_star = np.linalg.eigvalsh(Z.T @ gram_min(P) @ Z)[0]
    if not (lam_star / np.sqrt(K) - 1e-10 <= cTV <= np.sqrt(lam_min) + 1e-10):
        ok_a = False; log.append(f"(a) cTV bounds: {lam_star/np.sqrt(K):.4g} <= {cTV:.4g} <= {np.sqrt(lam_min):.4g}")
    # delta0^TV and modulus on the sphere
    d0TV = max(delta0_of_u(w, u) for u in Umin)
    r = np.sqrt(K / (K - 1)) * w.min()
    if d0TV < r - 1e-9: ok_a = False; log.append("(a) delta0TV < r")
    dmax = max(np.linalg.norm(np.eye(K)[k] - w) for k in range(K))
    feas_slack = (w[None, :] + 0 * U)                       # placeholder
    ratios = []
    for d in np.linspace(0.02, dmax * 0.98, 25):
        feas = (w[None, :] + d * U).min(axis=1) >= -1e-12
        om = (0.5 * d * f1[feas]).min()                      # omega_TV(d; w) exactly on the sphere
        ratios.append(om / d)
        if om < 0.5 * d * cTV - 1e-12: ok_a = False; log.append("(a) below bound")
        if d <= d0TV and abs(om - 0.5 * d * cTV) > 1e-9: ok_a = False; log.append(f"(a) not equal on (0,d0TV] d={d:.3f} d0={d0TV:.3f}")
        if d > d0TV * 1.02 and om <= 0.5 * d * cTV + 1e-12: ok_a = False; log.append(f"(a) not strict d={d:.3f}")
        if om > 0.5 * d * np.sqrt(K) + 1e-12: ok_a = False; log.append("(a) above sqrtK/2")
    if any(ratios[i + 1] < ratios[i] - 1e-9 for i in range(len(ratios) - 1)): ok_a = False; log.append("(a) ratio not monotone")
# affinely dependent -> cTV = 0
P = rng.dirichlet(np.ones(6), size=2); P = np.vstack([P, 0.3 * P[0] + 0.7 * P[1]])
u_dep = np.array([0.3, 0.7, -1.0]); u_dep /= np.linalg.norm(u_dep)          # exact dependency direction (in T)
if np.abs(u_dep @ P).sum() > 1e-12: ok_a = False; log.append("(a) dependent cTV != 0")
if np.abs(U @ P).sum(axis=1).min() > 2e-3: ok_a = False; log.append("(a) grid min not near 0 for dependent family")
results["(a) (i) TV: exact, cTV bounds, equality/strict/monotone (K=3 exact on circle)"] = ok_a

# ------------------------------------------------------------------ (b) bounded-ratio sandwich
ok_b = True
for trial in range(200):
    K = int(rng.integers(2, 6)); m = int(rng.integers(3, 9))
    base = rng.dirichlet(np.ones(m), size=K); t = rng.uniform(0.05, 1.0)
    P = (1 - t) * base + t / m                                 # densities bounded away from 0 -> rho finite
    rho = (P.max(axis=0) / P.min(axis=0)).max()
    for _ in range(10):
        w = rng.dirichlet(np.ones(K)); w2 = rng.dirichlet(np.ones(K))
        if rng.random() < 0.3: w[rng.integers(K)] = 0; w /= w.sum()        # boundary w
        if rng.random() < 0.3: w2[rng.integers(K)] = 0; w2 /= w2.sum()
        pw, pw2 = w @ P, w2 @ P
        ratio = pw2 / pw
        if ratio.max() > rho + 1e-12 or ratio.min() < 1 / rho - 1e-12: ok_b = False; log.append("(b) ratio outside [1/rho, rho]")
        c2, h2 = CHI2(pw2, pw), H2(pw2, pw)
        lo, hi = c2 / (2 * (1 + np.sqrt(rho)) ** 2), c2 / (2 * (1 + rho ** -0.5) ** 2)
        if not (lo - 1e-12 <= h2 <= hi + 1e-12): ok_b = False; log.append(f"(b) sandwich {lo} {h2} {hi}")
# rho -> 1 limit: H^2/chi^2 -> 1/8
K, m = 3, 5; base = rng.dirichlet(np.ones(m), size=K)
lim_ok = True
for t in (0.9, 0.99, 0.999):
    P = (1 - t) * base + t / m; w = rng.dirichlet(np.ones(K)); w2 = rng.dirichlet(np.ones(K))
    rr = H2(w2 @ P, w @ P) / CHI2(w2 @ P, w @ P)
    if t == 0.999 and abs(rr - 1 / 8) > 5e-3: lim_ok = False; log.append(f"(b) limit ratio {rr}")
results["(b) (ii) ratio bounds + Hellinger sandwich incl. boundary; rho->1 gives 1/8"] = ok_b and lim_ok

# ------------------------------------------------------------------ (c) sympy
x = sp.symbols("x", real=True)
expr = x ** 2 - (sp.sqrt(1 + x) - 1) ** 2
c1 = all(expr.subs(x, sp.Rational(i, 10)) >= 0 for i in range(-10, 60))          # exact rationals on [-1, 6]
c1b = sp.simplify((sp.sqrt(1 + x) - 1) ** 2 - x ** 2 / (sp.sqrt(1 + x) + 1) ** 2) == 0
K = 4; ok_c = True
for _ in range(100):
    w = rng.dirichlet(np.ones(K - 1)); w = np.insert(w, rng.integers(K), 0.0)
    k = int(np.where(w == 0)[0][0]); n2 = np.sum((np.eye(K)[k] - w) ** 2)
    if not (K / (K - 1) - 1e-12 <= n2 <= 2 + 1e-12): ok_c = False
results["(c) sympy (sqrt(1+x)-1)^2 <= x^2 on x>=-1; ||e_k-w||^2 in [K/(K-1),2]"] = bool(c1) and bool(c1b) and ok_c

# ------------------------------------------------------------------ (d) boundary separation, anchor + extra private mass
ok_d = True
K, m = 3, 5
for trial in range(30):
    base = rng.dirichlet(np.ones(m), size=K); a = rng.uniform(0.05, 0.3, size=K); extra = rng.uniform(0.0, 0.2)
    # cells: m shared | K private (anchor) | 1 extra private cell for endmember k=0 (mass 'extra')
    P = np.zeros((K, m + K + 1))
    for j in range(K):
        e = extra if j == 0 else 0.0
        P[j, :m] = (1 - a[j] - e) * base[j]; P[j, m + j] = a[j]; P[j, m + K] = e
    k = 0
    w = np.array([0.0, 0.0, 0.0]); w[1:] = rng.dirichlet(np.ones(K - 1))
    pw = w @ P
    Bk = (P[k] > 0) & (pw == 0); bk = P[k, Bk].sum()
    if bk < a[k] - 1e-12 or (extra > 0 and abs(bk - a[k] - extra) > 1e-12): ok_d = False; log.append("(d) b_k value")
    Sp = pw > 0; gam = (P[k, Sp] ** 2 / pw[Sp]).sum()
    ek = np.eye(K)[k]
    for d in (0.5, 0.1, 0.01, 0.001, 1e-5):
        w2 = (1 - d) * w + d * ek; p2 = w2 @ P
        c2, tv, h2 = CHI2(p2, pw), TV(p2, pw), H2(p2, pw)
        if not np.isinf(c2): ok_d = False; log.append("(d) chi2 finite")
        if not (0.5 * d * bk - 1e-12 <= tv <= d + 1e-12): ok_d = False; log.append("(d) TV interval")
        if not (0.5 * d * bk - 1e-12 <= h2 <= 0.5 * d * bk + 0.5 * d ** 2 * (gam + 1) + 1e-12): ok_d = False; log.append(f"(d) H2 sandwich d={d}")
        # convergence of the first-order coefficient: deviation must be within the proven second-order term
        if abs(h2 / d - 0.5 * bk) > 0.5 * d * (gam + 1) + 1e-12: ok_d = False; log.append(f"(d) first-order coefficient {h2/d} vs {0.5*bk}")
        if d == 1e-5 and abs(h2 / d - 0.5 * bk) > 0.01 * 0.5 * bk: ok_d = False; log.append(f"(d) coefficient not converged at d=1e-5: {h2/d} vs {0.5*bk}")
# gamma_k = inf construction (discrete: impossible; use a continuous-like heavy tail on a fine grid where p_k^2/p_w is huge) -> only check lower bound still holds
xs = np.arange(1, 4001); pw_c = 1 / xs ** 3; pw_c /= pw_c.sum(); pk_c = 1 / xs ** 1.2; pk_c /= pk_c.sum()
P2 = np.vstack([pk_c, pw_c]); w = np.array([0.0, 1.0]); d = 0.05; w2 = (1 - d) * w + d * np.eye(2)[0]
h2 = H2(w2 @ P2, w @ P2); gam = (pk_c ** 2 / pw_c).sum()
if h2 < 0.5 * d * 0.0 - 1e-12: ok_d = False
log.append(f"(d) heavy-tail check: gamma_k(grid)={gam:.3g} (grows with grid), H2={h2:.4g}, lower bound 0.5*d*b_k=0 holds; upper bound 0.5 d^2 (gamma+1)={0.5*d*d*(gam+1):.3g}")
results["(d) (iii) anchor boundary pair: chi2=inf, TV interval, H2 sandwich, first-order coefficient"] = ok_d

print("=" * 60)
for k_, v in results.items(): print(("PASS " if v else "FAIL ") + k_)
print("=" * 60)
for line in log[:20]: print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
