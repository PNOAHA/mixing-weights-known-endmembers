"""Machine verification for Theorem A1 (chi separation modulus).

Checks (a)-(g) listed in notes/A1.md, on random discrete endmember families.
Run:  python a1_verify.py   -> prints PASS/FAIL per item and a summary.
"""
import numpy as np
from scipy.optimize import minimize
import sympy as sp

rng = np.random.default_rng(20260916)
TOL = 1e-9
results = {}


def tangent_basis(K):
    """K x (K-1) matrix Z with orthonormal columns spanning T = {sum v = 0}."""
    A = np.eye(K) - np.ones((K, K)) / K
    U, s, _ = np.linalg.svd(A)
    return U[:, : K - 1]


def random_endmembers(K, m, theta=0.3):
    """Discrete densities on {1..m}: Dirichlet draws mixed with uniform."""
    P = rng.dirichlet(np.ones(m) * 0.7, size=K)
    P = (1 - theta) * P + theta / m
    return P  # shape K x m, rows sum to 1


def gram(P, w):
    pw = w @ P
    mask = pw > 0
    return (P[:, mask] / pw[mask]) @ P[:, mask].T


def gram_min(P):
    pmax = P.max(axis=0)
    mask = pmax > 0
    return (P[:, mask] / pmax[mask]) @ P[:, mask].T


def chi2_direct(P, w, w2):
    pw, pw2 = w @ P, w2 @ P
    out = 0.0
    for a, b in zip(pw2, pw):
        if b > 0:
            out += (a - b) ** 2 / b
        elif a > 0:
            return np.inf
    return out


def random_simplex_point(K, interior=True):
    w = rng.dirichlet(np.ones(K))
    if interior:
        w = 0.9 * w + 0.1 / K
    return w


def delta0_of_u(w, u):
    neg = u < -1e-14
    if not neg.any():
        return np.inf
    return np.min(w[neg] / (-u[neg]))


def delta0(w, Z, I):
    lam, V = np.linalg.eigh(I)
    lmin = lam[0]
    idx = np.where(lam - lmin < 1e-10)[0]
    if len(idx) == 1:
        u = Z @ V[:, 0]
        cands = [delta0_of_u(w, u), delta0_of_u(w, -u)]
        return max(cands), (u if cands[0] >= cands[1] else -u)
    # repeated eigenvalue: sample the unit sphere of E_min densely
    B = Z @ V[:, idx]
    best, ubest = -1, None
    for _ in range(20000):
        y = rng.normal(size=len(idx))
        y /= np.linalg.norm(y)
        u = B @ y
        d = delta0_of_u(w, u)
        if d > best:
            best, ubest = d, u
    return best, ubest


def constrained_min(G, w, Z, delta, lmin, n_starts=30):
    """min v^T G v s.t. v in T, ||v|| = delta, w + v >= 0.  Multi-start SLSQP."""
    K = len(w)
    best = np.inf
    for _ in range(n_starts):
        y0 = rng.normal(size=K - 1)
        y0 = y0 / np.linalg.norm(y0) * delta
        cons = [
            {"type": "eq", "fun": lambda y: y @ y - delta**2},
            {"type": "ineq", "fun": lambda y: w + Z @ y},
        ]
        r = minimize(lambda y: y @ (Z.T @ G @ Z) @ y, y0, constraints=cons,
                     method="SLSQP", options={"ftol": 1e-13, "maxiter": 500})
        if r.success:
            y = r.x
            if abs(y @ y - delta**2) < 1e-7 and (w + Z @ y).min() > -1e-8:
                best = min(best, r.fun)
    return best


# ---------------------------------------------------------------- (a)-(e)
ok_a = ok_b = ok_c = ok_d = ok_e = True
log = []
for trial in range(40):
    K = int(rng.integers(2, 6))
    m = int(rng.integers(K + 1, 12))
    P = random_endmembers(K, m)
    Z = tangent_basis(K)
    w = random_simplex_point(K)
    G = gram(P, w)
    I = Z.T @ G @ Z
    lam = np.linalg.eigvalsh(I)
    lmin, lmax = lam[0], lam[-1]

    # (a) exact quadratic form
    for _ in range(20):
        w2 = random_simplex_point(K, interior=False)
        v = w2 - w
        lhs = chi2_direct(P, w, w2)
        rhs = v @ G @ v
        if abs(lhs - rhs) > TOL * max(1, abs(lhs)):
            ok_a = False
            log.append(f"(a) trial{trial}: {lhs} vs {rhs}")

    # (b) lower bound chi >= delta sqrt(lmin) for random w'
    for _ in range(50):
        w2 = random_simplex_point(K, interior=False)
        v = w2 - w
        d = np.linalg.norm(v)
        if d < 1e-12:
            continue
        if np.sqrt(v @ G @ v) < d * np.sqrt(lmin) - 1e-10:
            ok_b = False
            log.append(f"(b) trial{trial}")

    # (c) equality for delta <= delta0, delta0 >= r(w), boundary attained
    d0, u = delta0(w, Z, I)
    r = np.sqrt(K / (K - 1)) * w.min()
    if d0 < r - 1e-12:
        ok_c = False
        log.append(f"(c) trial{trial}: delta0={d0} < r={r}")
    wb = w + d0 * u
    if wb.min() < -1e-12 or abs(wb.min()) > 1e-9:
        ok_c = False
        log.append(f"(c) trial{trial}: w+delta0 u not on boundary, min={wb.min()}")
    for frac in [0.1, 0.5, 0.999, 1.0]:
        d = frac * d0
        v = d * u
        val = v @ G @ v
        if abs(val - d**2 * lmin) > 1e-9 * max(1, val):
            ok_c = False
            log.append(f"(c) trial{trial}: frac{frac} {val} vs {d**2*lmin}")
        # also the constrained min should equal lmin d^2
        cm = constrained_min(G, w, Z, d, lmin, n_starts=10)
        if cm < d**2 * lmin - 1e-7 or cm > d**2 * lmin + 1e-6:
            ok_c = False
            log.append(f"(c) trial{trial}: QP at frac{frac}: {cm} vs {d**2*lmin}")

    # (d) beyond delta0: strict >, <= lmax, monotone f/delta^2
    dmax = max(np.linalg.norm(np.eye(K)[k] - w) for k in range(K))
    if dmax > d0 * 1.01:
        deltas = np.linspace(d0 * 1.02, dmax * 0.98, 5)
        ratios = []
        for d in deltas:
            f = constrained_min(G, w, Z, d, lmin)
            if not np.isfinite(f):
                continue
            ratios.append(f / d**2)
            if f <= d**2 * lmin + 1e-9:
                ok_d = False
                log.append(f"(d) trial{trial}: not strict at d={d}: {f} vs {d**2*lmin}")
            if f > d**2 * lmax + 1e-7:
                ok_d = False
                log.append(f"(d) trial{trial}: > lmax at d={d}")
        if any(ratios[i + 1] < ratios[i] - 1e-6 for i in range(len(ratios) - 1)):
            ok_d = False
            log.append(f"(d) trial{trial}: ratio not monotone {ratios}")

    # (e) global: G(w) >= Gmin, lam* <= lmin, boundary w
    Gm = gram_min(P)
    lstar = np.linalg.eigvalsh(Z.T @ Gm @ Z)[0]
    if np.linalg.eigvalsh(G - Gm).min() < -1e-9:
        ok_e = False
        log.append(f"(e) trial{trial}: G - Gmin not PSD")
    if lstar > lmin + 1e-9:
        ok_e = False
        log.append(f"(e) trial{trial}: lam*={lstar} > lmin={lmin}")
    # boundary w: put one coordinate to zero
    wbd = w.copy(); wbd[rng.integers(K)] = 0; wbd /= wbd.sum()
    for _ in range(30):
        w2 = random_simplex_point(K, interior=False)
        v = w2 - wbd
        d = np.linalg.norm(v)
        c2 = chi2_direct(P, wbd, w2)
        if c2 < d**2 * lstar - 1e-10:
            ok_e = False
            log.append(f"(e) trial{trial}: boundary w violates")

results["(a) exact quadratic form"] = ok_a
results["(b) local lower bound"] = ok_b
results["(c) equality on (0,delta0], delta0>=r(w)"] = ok_c
results["(d) beyond delta0: strict, <=lmax, monotone"] = ok_d
results["(e) global: Loewner, lam*<=lmin, boundary w"] = ok_e

# ---------------------------------------------------------------- (f)
K, m = 3, 6
P = random_endmembers(2, m)
P = np.vstack([P, 0.4 * P[0] + 0.6 * P[1]])  # affinely dependent
Z = tangent_basis(K)
lstar = np.linalg.eigvalsh(Z.T @ gram_min(P) @ Z)[0]
w = random_simplex_point(K)
lmin_dep = np.linalg.eigvalsh(Z.T @ gram(P, w) @ Z)[0]
results["(f) affinely dependent => lam*=0 and lmin=0"] = abs(lstar) < 1e-10 and abs(lmin_dep) < 1e-10

# ---------------------------------------------------------------- (g) sympy
# K=2: T spanned by u=(1,-1)/sqrt2; I(w) = 1/2 * sum (p1-p2)^2 / pw
p1 = sp.symbols("p1_1:4", positive=True)
p2 = sp.symbols("p2_1:4", positive=True)
w1 = sp.symbols("w1", positive=True)
pw = [w1 * a + (1 - w1) * b for a, b in zip(p1, p2)]
Gsym = sp.Matrix(2, 2, lambda i, j: sum(
    [p1, p2][i][t] * [p1, p2][j][t] / pw[t] for t in range(3)))
u = sp.Matrix([1, -1]) / sp.sqrt(2)
I_sym = (u.T * Gsym * u)[0]
closed = sp.Rational(1, 2) * sum((a - b) ** 2 / c for a, b, c in zip(p1, p2, pw))
g1 = sp.simplify(I_sym - closed) == 0
# delta0 for K=2 (E_min = T): u=(1,-1)/sqrt2 gives delta0(u)=w2/(1/sqrt2), -u gives w1/(1/sqrt2);
# delta0(w) = max of the two = sqrt2*max(w1,w2) = delta_max(w).  (adversarial review item 1:
# NOT sqrt2*min; that is only the lower bound r(w).)
w2s = 1 - w1
d0_u = w2s / (1 / sp.sqrt(2))
d0_mu = w1 / (1 / sp.sqrt(2))
# sympy does not fold Max symbolically; check exactly on rationals covering both branches
expr = sp.Max(d0_u, d0_mu) - sp.sqrt(2) * sp.Max(w1, w2s)
g2 = all(sp.simplify(expr.subs(w1, sp.Rational(i, 20))) == 0 for i in range(1, 20))
# numeric cross-check against the generic delta0() routine
Z2 = tangent_basis(2)
for wv in ([0.9, 0.1], [0.3, 0.7], [0.5, 0.5]):
    wv = np.array(wv)
    P2 = random_endmembers(2, 5)
    d0n, _ = delta0(wv, Z2, Z2.T @ gram(P2, wv) @ Z2)
    dmax2 = max(np.linalg.norm(np.eye(2)[k] - wv) for k in range(2))
    g2 = g2 and abs(d0n - np.sqrt(2) * wv.max()) < 1e-12 and abs(d0n - dmax2) < 1e-12
# max over unit u in T of (-u_k) = sqrt((K-1)/K): Lagrange check for symbolic K
Ksym = sp.symbols("K", positive=True, integer=True)
# u = -(e_k - 1/K) / ||e_k - 1/K||, -u_k = (1-1/K)/sqrt(1-1/K) = sqrt(1-1/K)
g3 = sp.simplify((1 - 1 / Ksym) / sp.sqrt(1 - 1 / Ksym) - sp.sqrt((Ksym - 1) / Ksym)) == 0
# numeric confirmation of the max via optimisation, K=4
Z4 = tangent_basis(4)
best = -1
for _ in range(5000):
    y = rng.normal(size=3); y /= np.linalg.norm(y)
    best = max(best, -(Z4 @ y)[0])
g3n = abs(best - np.sqrt(3 / 4)) < 1e-3
results["(g) sympy K=2 closed form, delta0, sqrt((K-1)/K)"] = g1 and g2 and g3 and g3n

# ---------------------------------------------------------------- report
print("=" * 60)
for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k)
print("=" * 60)
if log:
    print("details:")
    for line in log[:30]:
        print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
