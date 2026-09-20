"""Machine verification for Lemma C2 (density-ratio Gram vs L2 Gram; Gaussian reduction).

Items (a)-(f) as listed in notes/C2.md.  Run: python c2_verify.py
"""
import numpy as np
import sympy as sp
from scipy.stats import norm

rng = np.random.default_rng(20260916)
results, log = {}, []


def tangent_basis(K):
    A = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(A)
    return U[:, : K - 1]


def rand_simplex(K, interior=True):
    w = rng.dirichlet(np.ones(K))
    return 0.9 * w + 0.1 / K if interior else w


def lmin(A):
    return np.linalg.eigvalsh(A)[0]


def psd(A, tol=1e-10):
    return lmin((A + A.T) / 2) >= -tol


# ------------------------------------------------------------------ (a),(b),(f): discrete
ok_a = ok_b = ok_f = True
for trial in range(60):
    K = int(rng.integers(2, 6)); m_ = int(rng.integers(K + 1, 14))
    P = rng.dirichlet(np.ones(m_) * 0.8, size=K)
    P = 0.6 * P + 0.4 / m_                       # bounded away from zero
    Z = tangent_basis(K)
    L = P @ P.T
    Gmin = (P / P.max(axis=0)) @ P.T
    M, m = P.max(), P.min()
    for interior in (True, False):
        w = rand_simplex(K, interior)
        if not interior:
            w = w.copy(); w[rng.integers(K)] = 0; w /= w.sum()
        pw = w @ P
        G = (P / pw) @ P.T
        I = Z.T @ G @ Z
        if not (psd(G - L / M) and psd(L / m - G)):
            ok_a = False; log.append(f"(a) Loewner fail trial{trial} interior={interior}")
        lL = lmin(Z.T @ L @ Z)
        if not (lL / M - 1e-10 <= lmin(I) <= lL / m + 1e-10):
            ok_a = False; log.append(f"(a) eig fail trial{trial}")
        if interior:
            if not (psd(G - Gmin) and psd(Gmin / w.min() - G)):
                ok_b = False; log.append(f"(b) fail trial{trial}")
            # (f) truncation: random subset A
            A = rng.random(m_) < 0.6
            if A.sum() == 0 or (~A).sum() == 0:
                continue
            LA = P[:, A] @ P[:, A].T
            MA, mA = P[:, A].max(), P[:, A].min()
            tail = sum(P[k, ~A].sum() / w[k] for k in range(K))
            lo, hi = lmin(Z.T @ LA @ Z) / MA, lmin(Z.T @ LA @ Z) / mA + tail
            if not (lo - 1e-10 <= lmin(I) <= hi + 1e-10):
                ok_f = False; log.append(f"(f) fail trial{trial}: {lo} {lmin(I)} {hi}")
results["(a) (i) two-sided sandwich, interior+boundary w"] = ok_a
results["(b) (ii) Gmin <= G(w) <= Gmin/min w"] = ok_b
results["(f) (iii) truncated sandwich"] = ok_f

# ------------------------------------------------------------------ (c): sympy closed forms
x, h, tj, tk, a, s = sp.symbols("x h t_j t_k a s", real=True)
hp = sp.symbols("h", positive=True)
phi = lambda u: sp.exp(-u**2 / (2 * hp**2)) / sp.sqrt(2 * sp.pi * hp**2)
Gt = sp.integrate(phi(x - tj) * phi(x - tk) / phi(x), (x, -sp.oo, sp.oo))
c1 = sp.simplify(Gt - sp.exp(tj * tk / hp**2)) == 0
Ljk = sp.integrate(phi(x - tj) * phi(x - tk), (x, -sp.oo, sp.oo))
c2 = sp.simplify(Ljk - sp.exp(-(tj - tk)**2 / (4 * hp**2)) / sp.sqrt(4 * sp.pi * hp**2)) == 0
c3 = sp.simplify(phi(x) * sp.exp(3 * a * x / hp**2) - phi(x - 3 * a) * sp.exp(9 * a**2 / (2 * hp**2))) == 0
mgf = sp.integrate(phi(x) * sp.exp(s * x / hp**2), (x, -sp.oo, sp.oo))
c4 = sp.simplify(mgf - sp.exp(s**2 / (2 * hp**2))) == 0
zs = np.linspace(0, 8, 801)
c5 = np.all(norm.sf(zs) <= 0.5 * np.exp(-zs**2 / 2) + 1e-15)
results["(c) sympy: G~=exp(t_j t_k/h^2), L closed form, shift identity, MGF, tail bound"] = c1 and c2 and c3 and c4 and bool(c5)
if not (c1 and c2 and c3 and c4 and c5):
    log.append(f"(c) flags {c1,c2,c3,c4,c5}")

# ------------------------------------------------------------------ Gaussian numerics
def gauss_family(t, hh):
    a_ = np.abs(t).max()
    X = np.linspace(-(a_ + 14 * hh), a_ + 14 * hh, 400001)
    dx = X[1] - X[0]
    P = np.exp(-(X[None, :] - t[:, None])**2 / (2 * hh**2)) / np.sqrt(2 * np.pi * hh**2)
    return X, dx, P


def fisher_gauss(P, dx, w):
    pw = w @ P
    mask = pw > 1e-300
    return (P[:, mask] / pw[mask]) @ P[:, mask].T * dx


def matrices(t, hh):
    Kp = np.exp(-(t[:, None] - t[None, :])**2 / (2 * hh**2))
    D = np.diag(np.exp(t**2 / (2 * hh**2)))
    L = np.exp(-(t[:, None] - t[None, :])**2 / (4 * hh**2)) / np.sqrt(4 * np.pi * hh**2)
    return Kp, D, L


# (d) finite-h explicit bounds
ok_d = True
for K, hh in [(3, 4.0), (3, 8.0), (4, 6.0), (2, 3.0)]:
    t = np.linspace(-1, 1, K); a_ = 1.0
    Z = tangent_basis(K)
    X, dx, P = gauss_family(t, hh)
    Kp, D, L = matrices(t, hh)
    Lam = lmin(Z.T @ D @ Kp @ D @ Z)
    for w in [rand_simplex(K), rand_simplex(K), np.eye(K)[0], np.r_[0.5, 0.5, np.zeros(K - 2)]]:
        lam = lmin(Z.T @ fisher_gauss(P, dx, w) @ Z)
        for R in 3 * a_ / hh + np.array([1.0, 2.0, 4.0, 6.0]):
            eps = K * np.exp(5 * a_**2 / hh**2) * np.exp(-(R - 3 * a_ / hh)**2 / 2)
            lo = np.exp(-a_ * R / hh) * (Lam - eps)
            hi = np.exp(a_ * R / hh + a_**2 / (2 * hh**2)) * Lam + eps
            if not (lo - 1e-12 <= lam <= hi + 1e-12):
                ok_d = False
                log.append(f"(d) K={K} h={hh} R={R:.2f} w={np.round(w,2)}: lo={lo:.3e} lam={lam:.3e} hi={hi:.3e}")
    # also (b) lower bound via M
    lamw = lmin(Z.T @ fisher_gauss(P, dx, rand_simplex(K)) @ Z)
    if lamw < np.sqrt(2 * np.pi) * hh * lmin(Z.T @ L @ Z) - 1e-12:
        ok_d = False; log.append(f"(d) M-lower bound fail K={K} h={hh}")
results["(d) (iv)(d) explicit bounds at finite h, incl. boundary w; (iv)(b) M-bound"] = ok_d

# (e) flat limit ratios (observational)
print("(e) flat-limit ratios  [lam_min(I(w))/Lambda_h,  Lambda_h/lmin(Z'K'Z),  Lambda_h/(sqrt(2pi) h lmin(Z'LZ))]")
ratios_to_one = True
for K in (3, 4):
    t = np.linspace(-1, 1, K)
    Z = tangent_basis(K)
    hs = [2, 4, 8, 16, 32, 64] if K == 3 else [2, 4, 8, 16]
    for hh in hs:
        X, dx, P = gauss_family(t, float(hh))
        Kp, D, L = matrices(t, float(hh))
        Lam = lmin(Z.T @ D @ Kp @ D @ Z)
        lamI = [lmin(Z.T @ fisher_gauss(P, dx, w) @ Z) for w in (np.ones(K) / K, rand_simplex(K), np.eye(K)[0])]
        r1 = [v / Lam for v in lamI]
        r2 = Lam / lmin(Z.T @ Kp @ Z)
        r3 = Lam / (np.sqrt(2 * np.pi) * hh * lmin(Z.T @ L @ Z))
        print(f"  K={K} h={hh:3d}: Lambda={Lam:.3e}  lam/Lambda (centroid, rand, vertex)={np.round(r1,4)}  "
              f"Lambda/lmin(Z'K'Z)={r2:.4f}  Lambda/(M-bound)={r3:.4f}")
        if hh >= 16 and any(abs(r - 1) > 0.05 for r in r1):
            ratios_to_one = False
results["(e) lam_min(I(w))/Lambda_h within 5% of 1 for h>=16 (all w tested)"] = ratios_to_one

print("=" * 60)
for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k)
print("=" * 60)
for line in log[:30]:
    print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
