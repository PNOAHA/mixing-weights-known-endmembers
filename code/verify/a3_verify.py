"""Machine verification for A3 (anchor class: upper bound, two-point lower bound, pairing). Items (a)-(f)."""
import numpy as np
import sympy as sp

rng = np.random.default_rng(20260917)
results, log = {}, []


def tangent_basis(K):
    A = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(A); return U[:, :K - 1]


def gram(P, w):
    pw = w @ P; m = pw > 0
    return (P[:, m] / pw[m]) @ P[:, m].T


def H2(p, q): return 0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()
def CHI2(p, q):
    m = q > 0
    return np.inf if np.any(p[~m] > 0) else float(((p[m] - q[m]) ** 2 / q[m]).sum())


def anchor_family(K, m, a, rng, theta=0.3):
    base = rng.dirichlet(np.ones(m) * 0.7, size=K); base = (1 - theta) * base + theta / m
    P = np.zeros((K, m + K))
    for k in range(K):
        P[k, :m] = (1 - a[k]) * base[k]; P[k, m + k] = a[k]
    return P


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u); rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    th = (css[rho] - 1) / (rho + 1); return np.maximum(v - th, 0)


# (a) A3.2 Monte Carlo
K, m = 3, 5; a = np.array([0.15, 0.3, 0.1]); P = anchor_family(K, m, a, rng)
Z = tangent_basis(K); w = np.array([0.5, 0.3, 0.2]); n, reps = 200, 20000
probs = np.r_[w * a, 1 - (w * a).sum()]
counts = rng.multinomial(n, probs, size=reps)[:, :K]
what = counts / (n * a)
risk_mc = ((what - w) ** 2).sum(axis=1).mean()
risk_th = (w * (1 - w * a) / a).sum() / n
risk_proj = np.mean([((proj_simplex(x) - w) ** 2).sum() for x in what])
kstar = 0; comp = what.copy(); comp[:, kstar] = 1 - what[:, [1, 2]].sum(axis=1)
risk_comp_mc = ((comp - w) ** 2).sum(axis=1).mean()
idx = [1, 2]; risk_comp_th = 2 * (w[idx] * (1 - w[idx] * a[idx]) / a[idx]).sum() / n - (w[idx].sum()) ** 2 / n + (w[idx] ** 2).sum() / n
se = ((what - w) ** 2).sum(axis=1).std() / np.sqrt(reps)
ok_a = abs(risk_mc - risk_th) < 4 * se and risk_proj <= risk_mc + 1e-12 and abs(risk_comp_mc - risk_comp_th) < 4 * se * 3 \
    and risk_th <= (w / a).sum() / n + 1e-15 and risk_comp_th <= 2 * (w[idx] / a[idx]).sum() / n + 1e-15
log.append(f"(a) risk MC={risk_mc:.5f} th={risk_th:.5f} proj={risk_proj:.5f} comp MC={risk_comp_mc:.5f} th={risk_comp_th:.5f}")
results["(a) A3.2 exact risk, projection, complement estimator (MC)"] = bool(ok_a)

# (b) A3.3.1
ok_b = True
for _ in range(100):
    K = int(rng.integers(2, 6)); m = int(rng.integers(3, 8)); a = rng.uniform(0.02, 0.5, size=K)
    P = anchor_family(K, m, a, rng, theta=rng.uniform(0, 0.8)); Z = tangent_basis(K)
    w = 0.9 * rng.dirichlet(np.ones(K)) + 0.1 / K
    G = gram(P, w); D = np.diag(a / w)
    if np.linalg.eigvalsh(G - D).min() < -1e-10: ok_b = False; log.append("(b) G - D not PSD")
    if np.linalg.eigvalsh(Z.T @ G @ Z)[0] < (a / w).min() - 1e-10: ok_b = False; log.append("(b) lam_min < min a/w")
results["(b) A3.3.1 G >= diag(a/w), lam_min >= min a_k/w_k"] = ok_b

# (c) A3.0 interior sandwich incl. anchor families (rho = inf)
ok_c = True
for _ in range(200):
    K = int(rng.integers(2, 6)); m = int(rng.integers(3, 8))
    P = anchor_family(K, m, rng.uniform(0.05, 0.4, size=K), rng) if rng.random() < 0.5 else rng.dirichlet(np.ones(m), size=K)
    w = 0.9 * rng.dirichlet(np.ones(K)) + 0.1 / K; rho_w = 1 / w.min()
    for _ in range(5):
        w2 = rng.dirichlet(np.ones(K))
        if rng.random() < 0.3: w2[rng.integers(K)] = 0; w2 /= w2.sum()
        p2, p = w2 @ P, w @ P
        if (p2[p > 0] / p[p > 0]).max() > rho_w + 1e-12: ok_c = False; log.append("(c) ratio > rho_w")
        c2, h2 = CHI2(p2, p), H2(p2, p)
        if not (c2 / (2 * (1 + np.sqrt(rho_w)) ** 2) - 1e-12 <= h2 <= 0.5 * c2 + 1e-12): ok_c = False; log.append("(c) sandwich")
results["(c) A3.0 interior Hellinger-chi sandwich with rho_w = 1/min w"] = ok_c

# (d) optimal c and two-point MC
c = sp.symbols("c", positive=True)
f2 = c / 8 * (1 - sp.sqrt(c / 2)); crit = sp.solve(sp.diff(f2, c), c)
d1 = any(sp.simplify(cc - sp.Rational(8, 9)) == 0 for cc in crit) and sp.simplify(f2.subs(c, sp.Rational(8, 9)) - sp.Rational(1, 27)) == 0
f1max = sp.nsimplify(1 / (16 * sp.E)); d2 = float(f1max) < 1 / 27
# also numeric global max of c*kappa(c)
cs = np.linspace(1e-4, 10, 200001); vals = cs / 4 * np.maximum(0.25 * np.exp(-cs), 0.5 * (1 - np.sqrt(cs / 2)))
d3 = abs(vals.max() - 1 / 27) < 1e-6 and abs(cs[vals.argmax()] - 8 / 9) < 1e-3
# MC: LR test average error for the constructed pair >= Thm 2.2(iii) bound
K, m = 3, 6; P = anchor_family(K, m, np.array([0.2, 0.2, 0.2]), rng); Z = tangent_basis(K); w = np.array([0.4, 0.35, 0.25])
G = gram(P, w); I = Z.T @ G @ Z; lam, V = np.linalg.eigh(I); u = Z @ V[:, 0]
d0 = max(np.min(w[u < 0] / (-u[u < 0])), np.min(w[u > 0] / (u[u > 0])))
u = u if np.min(w[u < 0] / (-u[u < 0])) >= np.min(w[u > 0] / (u[u > 0])) else -u
cc = 8 / 9; n = int(np.ceil(cc / (lam[0] * d0 ** 2))) + 5; dn = np.sqrt(cc / (n * lam[0])); w2 = w + dn * u
p, p2 = w @ P, w2 @ P
d4 = abs(CHI2(p2, p) - cc / n) < 1e-9 and (w2.min() >= -1e-12)
reps = 40000; X0 = rng.choice(len(p), size=(reps, n), p=p); X1 = rng.choice(len(p), size=(reps, n), p=p2)
llr = np.log(p2 / p)
err0 = (llr[X0].sum(axis=1) > 0).mean(); err1 = (llr[X1].sum(axis=1) <= 0).mean()
avg_err = 0.5 * (err0 + err1); bound = max(0.25 * np.exp(-cc), 0.5 * (1 - np.sqrt(cc / 2)))
d5 = avg_err >= bound - 0.01
log.append(f"(d) n={n} avg LR-test error={avg_err:.4f} >= bound {bound:.4f}; n H^2 = {n*H2(p2,p):.4f} <= c/2 = {cc/2:.4f}")
results["(d) c*kappa max = 1/27 at c=8/9 (sympy+numeric); MC two-point pair"] = bool(d1 and d2 and d3 and d4 and d5)

# (e) A3.5 common-background family
ok_e = True
for _ in range(50):
    K = int(rng.integers(2, 6)); m = int(rng.integers(3, 8)); Z = tangent_basis(K)
    q = rng.dirichlet(np.ones(m)); eq = rng.random() < 0.5
    a = np.full(K, rng.uniform(0.05, 0.5)) if eq else rng.uniform(0.05, 0.5, size=K)
    P = np.zeros((K, m + K))
    for k in range(K): P[k, :m] = (1 - a[k]) * q; P[k, m + k] = a[k]
    w = 0.9 * rng.dirichlet(np.ones(K)) + 0.1 / K
    G = gram(P, w); s = (w * (1 - a)).sum(); cvec = 1 - a
    Gth = np.diag(a / w) + np.outer(cvec, cvec) / s
    if np.abs(G - Gth).max() > 1e-10: ok_e = False; log.append("(e) closed form G")
    if abs(np.sum((Z.T @ cvec) ** 2) - np.sum((a - a.mean()) ** 2)) > 1e-10: ok_e = False; log.append("(e) |Z'c|^2")
    if eq:
        I = Z.T @ G @ Z; lm = np.linalg.eigvalsh(I)[0]; ws = np.sort(w)[::-1]
        if np.abs(I - a[0] * Z.T @ np.diag(1 / w) @ Z).max() > 1e-10: ok_e = False; log.append("(e) I = a Z'diag(1/w)Z")
        if not (a[0] / ws[0] - 1e-10 <= lm <= a[0] / ws[1] + 1e-10): ok_e = False; log.append("(e) interlacing")
        wc = np.ones(K) / K; Ic = Z.T @ gram(P, wc) @ Z
        if abs(np.linalg.eigvalsh(Ic)[0] - a[0] * K) > 1e-10: ok_e = False; log.append("(e) centroid lam = aK")
        # pairing ratio: upper 1/(n a) vs lower 1/(27 K n a) -> 27K by construction; check lower <= MLE-ish risk numerically not needed
results["(e) A3.5 closed forms, interlacing, centroid lam_min = aK"] = ok_e

# (f) A3.4 counterexample
ok_f = True; ratios = []
for a in (0.5, 0.1, 0.01, 0.001):
    m = 6; q1 = rng.dirichlet(np.ones(m)); q2 = rng.dirichlet(np.ones(m))
    P = np.zeros((2, 2 * m)); P[0, :m] = q1; P[1, m:] = q2          # disjoint supports; anchors = any subset
    Z = tangent_basis(2); w = np.array([0.6, 0.4])
    lm = np.linalg.eigvalsh(Z.T @ gram(P, w) @ Z)[0]
    if abs(lm - 1 / (2 * w[0] * w[1])) > 1e-10: ok_f = False; log.append("(f) lam_min formula")
    anchor_risk = (w * (1 - w * a) / a).sum(); minimax_upper = 2 * w[0] * w[1]     # both times 1/n
    ratios.append(anchor_risk / minimax_upper)
if not all(ratios[i + 1] > ratios[i] for i in range(len(ratios) - 1)) or ratios[-1] < 100: ok_f = False
log.append(f"(f) anchor risk / minimax upper (x n) for a=0.5,0.1,0.01,0.001: {np.round(ratios,1)}")
results["(f) A3.4 counterexample: pairing ratio diverges as a -> 0"] = ok_f

print("=" * 60)
for k_, v in results.items(): print(("PASS " if v else "FAIL ") + k_)
print("=" * 60)
for line in log: print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
