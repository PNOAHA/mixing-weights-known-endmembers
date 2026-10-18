"""Machine verification for 学习/01_下界_Tsybakov/定理重述.md (items a-g)."""
import itertools
import numpy as np
import sympy as sp

rng = np.random.default_rng(20260917)
results, log = {}, []


def rand_pq(m):
    p = rng.dirichlet(np.ones(m) * 0.7); q = rng.dirichlet(np.ones(m) * 0.7)
    return p, q


def TV(p, q): return 0.5 * np.abs(p - q).sum()
def H2(p, q): return 0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()     # project convention (with 1/2)
def KL(p, q): m = p > 0; return float((p[m] * np.log(p[m] / q[m])).sum())
def CHI2(p, q): m = q > 0; return float(((p[m] - q[m]) ** 2 / q[m]).sum()) if np.all(p[~m] == 0) else np.inf


def tangent_basis(K):
    A = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(A); return U[:, :K - 1]


# (a) conversion table
ok = True
for _ in range(300):
    p, q = rand_pq(int(rng.integers(2, 9)))
    tv, h2, kl, c2 = TV(p, q), H2(p, q), KL(p, q), CHI2(p, q)
    mn = np.minimum(p, q).sum()
    checks = [
        abs(tv - (1 - mn)) < 1e-12,                       # Scheffe
        mn >= 0.5 * (1 - h2) ** 2 - 1e-12,                # Le Cam (2.16) converted
        h2 <= tv + 1e-12 and tv <= np.sqrt(h2 * (2 - h2)) + 1e-12,   # (2.17) converted
        2 * h2 <= kl + 1e-12,                             # Lemma 2.4
        tv <= np.sqrt(kl / 2) + 1e-12,                    # Pinsker
        mn >= 0.5 * np.exp(-kl) - 1e-12,                  # Lemma 2.6
        kl <= np.log(1 + c2) + 1e-12 and np.log(1 + c2) <= c2 + 1e-12,   # Lemma 2.7
        tv <= np.sqrt(2 * h2) + 1e-12 and np.sqrt(2 * h2) <= np.sqrt(kl) + 1e-12,  # (2.20)
        h2 <= 0.5 * c2 + 1e-12 and tv <= 0.5 * np.sqrt(c2) + 1e-12,   # notation §2 extra two
    ]
    if not all(checks):
        ok = False; log.append(f"(a) fail flags {checks}")
results["(a) distance inequalities in project units"] = ok

# (b) product formulas by enumeration, n=2,3
ok = True
for n in (2, 3):
    for _ in range(20):
        p, q = rand_pq(int(rng.integers(2, 5)))
        idx = list(itertools.product(range(len(p)), repeat=n))
        pn = np.array([np.prod([p[i] for i in t]) for t in idx]); qn = np.array([np.prod([q[i] for i in t]) for t in idx])
        if abs(H2(pn, qn) - (1 - (1 - H2(p, q)) ** n)) > 1e-12: ok = False; log.append("(b) H2 product")
        if abs(KL(pn, qn) - n * KL(p, q)) > 1e-10: ok = False; log.append("(b) KL additive")
        if abs(CHI2(pn, qn) - ((1 + CHI2(p, q)) ** n - 1)) > 1e-9: ok = False; log.append("(b) chi2 product")
results["(b) product formulas n=2,3"] = ok

# (c) Theorem 2.2 bounds <= (1-TV)/2; converted Hellinger form; monotonicity
ok = True
b_tv = lambda a: (1 - a) / 2
b_H_T = lambda a: 0.5 * (1 - np.sqrt(a * (1 - a / 4)))            # Tsybakov units, a = H_T^2
b_H = lambda a1: 0.5 * (1 - np.sqrt(a1 * (2 - a1)))               # project units, a1 = H^2
b_K = lambda a: max(0.25 * np.exp(-a), 0.5 * (1 - np.sqrt(a / 2)))
for _ in range(300):
    p, q = rand_pq(int(rng.integers(2, 9)))
    tv, h2, kl, c2 = TV(p, q), H2(p, q), KL(p, q), CHI2(p, q)
    half = (1 - tv) / 2
    if b_tv(tv) > half + 1e-12 or b_H(h2) > half + 1e-12 or b_K(kl) > half + 1e-12 or b_K(c2) > half + 1e-12:
        ok = False; log.append("(c) bound > (1-TV)/2")
    if abs(b_H(h2) - b_H_T(2 * h2)) > 1e-12: ok = False; log.append("(c) conversion")
a = np.linspace(0, 0.999, 500)
if np.any(np.diff([b_H(x) for x in a]) > 1e-12) or np.any(np.diff([b_K(x) for x in np.linspace(0, 20, 500)]) > 1e-12):
    ok = False; log.append("(c) not monotone")
results["(c) Thm 2.2 bounds vs (1-TV)/2, conversion, monotone"] = ok

# (d) kappa values quoted in the notes
kH = lambda c: (1 - np.sqrt(c * (2 - c))) / 8
kC = lambda c: 0.25 * max(0.25 * np.exp(-c), 0.5 * (1 - np.sqrt(c / 2)))
vals = (kH(0.1), kH(0.5), kC(0.1), kC(1.0))
quoted = (0.0705, 0.0167, 0.0970, 0.0366)   # corrected after first run (LN had mis-evaluated kappa_chi)
results["(d) kappa numerics as quoted"] = all(abs(v - qv) < 6e-4 for v, qv in zip(vals, quoted))
log.append(f"(d) kappa_H(0.1)={vals[0]:.4f} kappa_H(0.5)={vals[1]:.4f} kappa_chi(0.1)={vals[2]:.4f} kappa_chi(1)={vals[3]:.4f}")

# (e) J(mu0)=pi^2 ; Fisher info of one-parameter subfamily
t = sp.symbols("t", real=True)
mu0 = sp.cos(sp.pi * t / 2) ** 2
J = sp.integrate(sp.diff(mu0, t) ** 2 / mu0, (t, -1, 1))
e1 = sp.simplify(J - sp.pi ** 2) == 0
K, m = 4, 7
P = rng.dirichlet(np.ones(m), size=K); Z = tangent_basis(K)
w = 0.8 * rng.dirichlet(np.ones(K)) + 0.2 / K
u = Z @ rng.normal(size=K - 1); u /= np.linalg.norm(u)
def fisher_fd(t0, n=1, eps=1e-5):
    pw = lambda tt: (w + tt * u) @ P
    score = (pw(t0 + eps) - pw(t0 - eps)) / (2 * eps) / pw(t0)
    return n * (score ** 2 * pw(t0)).sum()
G = (P / (w @ P)) @ P.T
e2 = abs(fisher_fd(0.0) - u @ G @ u) < 1e-7
results["(e) J(mu0)=pi^2 ; I(t)=n u'G(w_t)u"] = bool(e1) and bool(e2)

# (f) Assouad reduction and cube feasibility
ok = True
for _ in range(200):
    K = int(rng.integers(3, 7)); mm = int(rng.integers(1, K))
    Z = tangent_basis(K); w = 0.8 * rng.dirichlet(np.ones(K)) + 0.2 / K
    Q, _ = np.linalg.qr(rng.normal(size=(K - 1, mm))); U = Z @ Q      # m orthonormal directions in T
    r = np.sqrt(K / (K - 1)) * w.min(); eps = r / np.sqrt(mm) * rng.uniform(0.2, 1.0)
    om = rng.integers(0, 2, size=mm); w_om = w + eps * U @ om
    if w_om.min() < -1e-12 or abs(w_om.sum() - 1) > 1e-12: ok = False; log.append("(f) cube leaves simplex")
    what = w + rng.normal(size=K) * 0.3; what -= (what.sum() - 1) / K   # arbitrary estimate on affine hull
    om_hat = (U.T @ (what - w) >= eps / 2).astype(int)
    if np.sum((what - w_om) ** 2) < eps ** 2 / 4 * np.abs(om_hat - om).sum() - 1e-12: ok = False; log.append("(f) reduction fails")
results["(f) Assouad cube feasibility + reduction inequality"] = ok

# (g) notation §6 two facts via direct n-fold products
ok = True
for n in (2, 3, 4):
    for _ in range(20):
        p, q = rand_pq(3)
        idx = list(itertools.product(range(3), repeat=n))
        pn = np.array([np.prod([p[i] for i in tt]) for tt in idx]); qn = np.array([np.prod([q[i] for i in tt]) for tt in idx])
        tvn, h2 = TV(pn, qn), H2(p, q)
        if tvn > np.sqrt(2 * n * h2) + 1e-12: ok = False; log.append("(g) TV_n <= sqrt(2 n H^2) fails")
        if 1 - tvn < 0.5 * (1 - h2) ** (2 * n) - 1e-12: ok = False; log.append("(g) 1-TV_n >= (1-H^2)^{2n}/2 fails")
results["(g) notation §6 product facts"] = ok

print("=" * 60)
for k, v in results.items(): print(("PASS " if v else "FAIL ") + k)
print("=" * 60)
for line in log[:20]: print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
