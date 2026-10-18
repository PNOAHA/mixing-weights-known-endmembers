"""Machine verification for 学习/01_下界_Tsybakov/考题答卷.md (items a-f)."""
import itertools
import numpy as np
import sympy as sp

rng = np.random.default_rng(20260917)
results, log = {}, []


def tangent_basis(K):
    A = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(A); return U[:, :K - 1]


def H2(p, q): return 0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()


def gram(P, w):
    pw = w @ P; m = pw > 0
    return (P[:, m] / pw[m]) @ P[:, m].T


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u); rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    th = (css[rho] - 1) / (rho + 1); return np.maximum(v - th, 0)


# (a) joint convexity + scaling law
ok_a = True
K, m = 3, 5
grid = np.array([[i, j, 30 - i - j] for i in range(31) for j in range(31 - i)]) / 30.0     # 496 simplex points
for trial in range(6):
    P = rng.dirichlet(np.ones(m), size=K); P = 0.8 * P + 0.2 / m
    theta = rng.normal(size=K)
    D = np.array([[H2(g1 @ P, g0 @ P) for g1 in grid] for g0 in grid])                     # H^2 between all pairs
    Tdiff = np.abs(grid @ theta[:, None] - (grid @ theta)[None, :])
    # joint convexity in first argument, second fixed
    for _ in range(200):
        i0, i1 = rng.integers(len(grid), size=2); t = rng.random()
        wt = (1 - t) * grid[i0] + t * grid[i1]
        if H2(wt @ P, grid[i0] @ P) > t * D[i0, i1] + 1e-12: ok_a = False; log.append("(a) convexity")
    def omega(eps):
        return Tdiff[D <= eps ** 2].max()
    for eps in (0.05, 0.1, 0.2):
        for s in (0.3, 0.5, 0.8):
            if omega(s * eps) < s ** 2 * omega(eps) - 1e-9:
                ok_a = False; log.append(f"(a) scaling eps={eps} s={s}: {omega(s*eps):.4f} < {s**2*omega(eps):.4f}")
results["(a) H2 joint convexity; omega(s eps) >= s^2 omega(eps) on grid"] = ok_a

# (b) E1 intermediate: LR-test average error >= (1-sqrt(c(2-c)))/2 with c = n H^2
ok_b = True
for trial in range(5):
    P = rng.dirichlet(np.ones(m), size=K); P = 0.8 * P + 0.2 / m
    w0 = rng.dirichlet(np.ones(K)); w1 = rng.dirichlet(np.ones(K)); h2 = H2(w1 @ P, w0 @ P)
    n = max(1, int(0.5 / h2)); c = n * h2
    if c >= 1: continue
    p0, p1 = w0 @ P, w1 @ P; llr = np.log(p1 / p0); reps = 20000
    X0 = rng.choice(m, size=(reps, n), p=p0); X1 = rng.choice(m, size=(reps, n), p=p1)
    err = 0.5 * ((llr[X0].sum(1) > 0).mean() + (llr[X1].sum(1) <= 0).mean())
    bound = 0.5 * (1 - np.sqrt(c * (2 - c)))
    if err < bound - 0.01: ok_b = False; log.append(f"(b) err {err:.3f} < bound {bound:.3f}")
results["(b) E1 two-point average error >= Hellinger bound (MC)"] = ok_b

# (c) max of c^2 kappa_H(c)
cs = np.linspace(1e-4, 0.9999, 200001); v = cs ** 2 / 8 * (1 - np.sqrt(cs * (2 - cs)))
cmax, vmax = cs[v.argmax()], v.max()
results["(c) max c^2 kappa_H = 0.0042 at c~0.48"] = abs(vmax - 0.0042) < 1e-4 and abs(cmax - 0.48) < 0.01   # corrected after first run
log.append(f"(c) max c^2 kappa_H(c) = {vmax:.5f} at c = {cmax:.3f}")

# (d) sympy
t, s = sp.symbols("t s", positive=True)
mu = sp.cos(sp.pi * t / (2 * s)) ** 2 / s
J1 = sp.simplify(sp.integrate(sp.diff(mu, t) ** 2 / mu, (t, -s, s)))
d1 = sp.simplify(J1 - sp.pi ** 2 / s ** 2) == 0            # per coordinate; product prior adds: 2 pi^2/s^2
w1, w2, w3 = sp.symbols("w1 w2 w3", positive=True)
Zs = sp.Matrix(tangent_basis(3))                             # numeric orthonormal basis, exact enough for a trace identity check
Dm = sp.diag(1 / w1, 1 / w2, 1 / w3)
tr_num = (Zs.T * Dm * Zs).trace()
d2 = abs(float((tr_num - sp.Rational(2, 3) * (1 / w1 + 1 / w2 + 1 / w3)).subs({w1: 0.2, w2: 0.3, w3: 0.5}))) < 1e-12
a = sp.symbols("a", positive=True)
Ic = a * Zs.T * sp.diag(3, 3, 3) * Zs
d3 = all(abs(float((Ic[i, j] - (3 * a if i == j else 0)).subs(a, 1))) < 1e-12 for i in range(2) for j in range(2))
results["(d) sympy: cos^2 prior J = pi^2/s^2 per coord; tr(Z'diag(1/w)Z) = (2/3) sum 1/w; centroid I = 3a Id"] = bool(d1 and d2 and d3)

# (e) table 2.3: MC anchor estimator vs van Trees bounds, K=3 common background, a=0.2, n=400
a_ = 0.2; n = 400; reps = 20000; K = 3
q = rng.dirichlet(np.ones(5)); P = np.zeros((K, 5 + K))
for k in range(K): P[k, :5] = (1 - a_) * q; P[k, 5 + k] = a_
wc = np.ones(K) / K; Z = tangent_basis(K); I = Z.T @ gram(P, wc) @ Z
lam = np.linalg.eigvalsh(I); vt_trace = 4 / (n * lam.sum()); vt_sharp = (1 / lam).sum() / n
probs = np.r_[wc * a_, 1 - (wc * a_).sum()]
cnt = rng.multinomial(n, probs, size=reps)[:, :K]; what = cnt / (n * a_)
risk_raw = ((what - wc) ** 2).sum(1).mean(); risk_proj = np.mean([((proj_simplex(x) - wc) ** 2).sum() for x in what])
risk_th = (wc * (1 - wc * a_) / a_).sum() / n
# the sharp bound is asymptotic (s_n -> 0); at finite n the projected risk may sit within MC error of it, so test closeness, not ordering
se_proj = np.std([((proj_simplex(x) - wc) ** 2).sum() for x in what[:5000]]) / np.sqrt(reps)
ok_e = abs(lam[0] - 3 * a_) < 1e-10 and abs(vt_trace - vt_sharp) < 1e-12 and risk_raw >= vt_sharp \
    and abs(risk_proj / vt_sharp - 1) < 0.03 and risk_proj >= vt_sharp - 4 * se_proj \
    and abs(risk_raw - risk_th) < 3 * ((what - wc) ** 2).sum(1).std() / np.sqrt(reps)
log.append(f"(e) x n: vanTrees trace={n*vt_trace:.4f} sharp={n*vt_sharp:.4f} anchor raw MC={n*risk_raw:.4f} (th {n*risk_th:.4f}) proj MC={n*risk_proj:.4f}; "
           f"ratio proj/sharp={risk_proj/vt_sharp:.3f}; two-point 1/(81a)={1/(81*a_):.4f}")
results["(e) 2.3 table: centroid, a=0.2: bounds ordered, anchor risk matches formula"] = bool(ok_e)

# (f) harmonic-arithmetic: E5 >= E4 on random SPD 2x2
ok_f = True
for _ in range(500):
    l1, l2 = rng.uniform(0.1, 5, size=2)
    if 1 / l1 + 1 / l2 < 4 / (l1 + l2) - 1e-12: ok_f = False
results["(f) tr I^{-1} >= 4/tr I (E5 >= E4)"] = ok_f

print("=" * 60)
for k_, v_ in results.items(): print(("PASS " if v_ else "FAIL ") + k_)
print("=" * 60)
for line in log: print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
