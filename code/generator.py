"""paperA generator v0.1 — endmember families and separation-modulus numerics.

Notation follows paperA/00_标准型与记号.md (v0.1.1):
  endmembers p_k (rows of P, K x m, discrete X = {1..m}, nu = counting measure)
  Phi(w) = w @ P ;  T = {sum v = 0} ;  Z orthonormal basis of T (K x (K-1))
  G(w)_jk = sum_x p_j p_k / p_w ;  I(w) = Z' G(w) Z ;  lambda_min(w) = lambda_min(I(w))
  TV, H (with 1/2), chi (second argument = truth w)
  omega_d(delta; w) = inf { d(Phi(w'), Phi(w)) : w' in Delta, ||w' - w|| >= delta }
Families (skeleton §6): F1 discrete Dirichlet+uniform overlap theta; F2 anchor (private cells, masses a);
F3 truncated Gaussian on [-L, L] discretised (for C2 illustration only).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass


# ----------------------------------------------------------------------------- basics
def tangent_basis(K: int) -> np.ndarray:
    A = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(A)
    return U[:, : K - 1]


def simplex_grid(K: int, n_per_axis: int, margin: float = 0.05) -> np.ndarray:
    """Interior grid points of Delta (K=3 only for now): barycentric lattice pulled toward centroid."""
    if K != 3:
        raise NotImplementedError("v0.1 grids only K=3")
    pts = []
    for i in range(n_per_axis + 1):
        for j in range(n_per_axis + 1 - i):
            w = np.array([i, j, n_per_axis - i - j], float) / n_per_axis
            w = (1 - margin) * w + margin / K
            pts.append(w)
    return np.array(pts)


# ----------------------------------------------------------------------------- families
def family_F1(K: int, m: int, theta: float, rng: np.random.Generator, conc: float = 0.7) -> np.ndarray:
    """Discrete endmembers: Dirichlet(conc) rows mixed with uniform at overlap theta in [0,1]."""
    P = rng.dirichlet(np.ones(m) * conc, size=K)
    return (1 - theta) * P + theta / m


def family_F2(K: int, m: int, theta: float, a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Anchor type: F1 on m shared cells plus one private cell per endmember with mass a_k."""
    base = family_F1(K, m, theta, rng)
    P = np.zeros((K, m + K))
    for k in range(K):
        P[k, :m] = (1 - a[k]) * base[k]
        P[k, m + k] = a[k]
    return P


def family_F3(K: int, h: float, L: float, m: int, a: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Truncated, renormalised Gaussian kernels at equispaced nodes in [-a, a], discretised on m cells of [-L, L]."""
    t = np.linspace(-a, a, K)
    X = np.linspace(-L, L, m)
    P = np.exp(-(X[None, :] - t[:, None]) ** 2 / (2 * h ** 2))
    P /= P.sum(axis=1, keepdims=True)
    return P, X


# ----------------------------------------------------------------------------- quantities
@dataclass
class PointQuantities:
    G: np.ndarray
    I: np.ndarray
    lam_min: float
    lam_max: float
    u_min: np.ndarray          # unit vector in T attaining lambda_min
    delta0: float              # A1 (2)
    r: float                   # sqrt(K/(K-1)) min w
    delta_max: float
    c_TV: float                # A2(i): min_{u in T, |u|=1} ||sum u_k p_k||_1
    rho: float                 # A2(ii): sup_x max_jk p_j/p_k on support
    lam_star: float            # A1 (3)


def gram(P, w):
    pw = w @ P
    mask = pw > 0
    return (P[:, mask] / pw[mask]) @ P[:, mask].T


def gram_min(P):
    pmax = P.max(axis=0)
    mask = pmax > 0
    return (P[:, mask] / pmax[mask]) @ P[:, mask].T


def delta0_of_u(w, u):
    neg = u < -1e-14
    return np.inf if not neg.any() else float(np.min(w[neg] / (-u[neg])))


def c_tv(P, Z, n_starts=40, rng=None):
    """min over unit u in T of ||sum_k u_k p_k||_1 (nonconvex on the sphere; multi-start)."""
    rng = rng or np.random.default_rng(0)
    best = np.inf
    for _ in range(n_starts):
        y0 = rng.normal(size=Z.shape[1]); y0 /= np.linalg.norm(y0)
        r = minimize(lambda y: np.abs((Z @ y) @ P).sum(), y0,
                     constraints=[{"type": "eq", "fun": lambda y: y @ y - 1}], method="SLSQP",
                     options={"ftol": 1e-12, "maxiter": 400})
        if r.success:
            best = min(best, r.fun)
    return float(best)


def point_quantities(P, w, Z, rng=None) -> PointQuantities:
    K = len(w)
    G = gram(P, w); I = Z.T @ G @ Z
    lam, V = np.linalg.eigh(I)
    u = Z @ V[:, 0]
    d0p, d0m = delta0_of_u(w, u), delta0_of_u(w, -u)
    u_best = u if d0p >= d0m else -u
    delta0 = max(d0p, d0m)
    r = np.sqrt(K / (K - 1)) * w.min()
    dmax = max(np.linalg.norm(np.eye(K)[k] - w) for k in range(K))
    supp = P.max(axis=0) > 0
    Ps = P[:, supp]
    rho = float((Ps.max(axis=0) / np.where(Ps.min(axis=0) > 0, Ps.min(axis=0), np.nan)).max()) if (Ps.min(axis=0) > 0).all() else np.inf
    lam_star = float(np.linalg.eigvalsh(Z.T @ gram_min(P) @ Z)[0])
    return PointQuantities(G, I, float(lam[0]), float(lam[-1]), u_best, float(delta0), float(r), float(dmax),
                           c_tv(P, Z, rng=rng), rho, lam_star)


# ----------------------------------------------------------------------------- moduli
def dist(P, w, w2, kind):
    p, q = np.clip(w2 @ P, 0, None), w @ P   # p = candidate (clipped: SLSQP may step slightly outside Delta), q = truth
    if kind == "TV":
        return 0.5 * np.abs(p - q).sum()
    if kind == "H":
        return np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum())
    if kind == "chi":
        m = q > 0
        if np.any(p[~m] > 0):
            return np.inf
        return np.sqrt(((p[m] - q[m]) ** 2 / q[m]).sum())
    raise ValueError(kind)


def modulus(P, w, Z, delta, kind, n_starts=30, rng=None):
    """omega_d(delta; w) = min over v in T, ||v|| >= delta, w+v in Delta of d(Phi(w+v), Phi(w)).
    By the scaling argument (A1 step 3') the minimiser has ||v|| = delta for chi; for TV/H we keep the
    inequality constraint and let the optimiser decide.  Multi-start SLSQP; returns inf if infeasible."""
    rng = rng or np.random.default_rng(1)
    K = len(w)
    dmax = max(np.linalg.norm(np.eye(K)[k] - w) for k in range(K))
    if delta > dmax + 1e-12:
        return np.inf
    best = np.inf
    cons = [{"type": "ineq", "fun": lambda y: y @ y - delta ** 2},
            {"type": "ineq", "fun": lambda y: w + Z @ y}]
    for _ in range(n_starts):
        y0 = rng.normal(size=K - 1); y0 = y0 / np.linalg.norm(y0) * delta * rng.uniform(1.0, 1.3)
        y0 = np.clip(y0, -1, 1)
        r = minimize(lambda y: dist(P, w, w + Z @ y, kind), y0, constraints=cons, method="SLSQP",
                     options={"ftol": 1e-12, "maxiter": 500})
        if r.success and (w + Z @ r.x).min() > -1e-9 and r.x @ r.x >= delta ** 2 - 1e-9:
            best = min(best, r.fun)
    return float(best)


def a1_prediction(q: PointQuantities, delta):
    """A1: omega_chi = delta sqrt(lambda_min) for delta <= delta0; only a lower bound beyond."""
    return delta * np.sqrt(q.lam_min)


# ----------------------------------------------------------------------------- v0.2: exact 1-D moduli for K = 3
def _dir_k3(Z, phi):
    return Z @ np.stack([np.cos(phi), np.sin(phi)])          # (3, len(phi)) unit vectors in T


def _dist_vec(P, w, V, kind):
    """d(Phi(w+v), Phi(w)) for the columns v of V (3 x N); infeasible columns -> inf."""
    Wp = w[:, None] + V                                        # candidate weights, 3 x N
    feas = (Wp >= -1e-15).all(axis=0)
    p = np.clip(Wp.T @ P, 0, None); q = w @ P
    if kind == "TV":
        d = 0.5 * np.abs(p - q[None, :]).sum(axis=1)
    elif kind == "H":
        d = np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)[None, :]) ** 2).sum(axis=1))
    elif kind == "chi":
        m = q > 0
        d = np.sqrt(((p[:, m] - q[m][None, :]) ** 2 / q[m][None, :]).sum(axis=1))
        d = np.where((p[:, ~m] > 1e-15).any(axis=1), np.inf, d)
    else:
        raise ValueError(kind)
    return np.where(feas, d, np.inf)


def modulus_k3(P, w, Z, delta, kind, n_grid=36000):
    """omega_d(delta; w) for K = 3 by exact 1-D minimisation over the direction angle.
    The minimiser lies on the sphere ||v|| = delta (TV: degree-1 homogeneous; chi: degree-2; H: for t >= 0 the
    pointwise map |sqrt(p_w) - sqrt(p_w + t f_u)| is non-decreasing in t at every x, hence so is H), so the problem
    is one-dimensional in phi.
    Dense grid + Brent refinement (piecewise-smooth objective; feasible set = union of arcs)."""
    from scipy.optimize import minimize_scalar
    K = len(w)
    if K != 3:
        raise NotImplementedError("modulus_k3 is K=3 only")
    dmax = max(np.linalg.norm(np.eye(K)[k] - w) for k in range(K))
    if delta > dmax + 1e-12:
        return np.inf
    phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    d = _dist_vec(P, w, delta * _dir_k3(Z, phi), kind)
    if not np.isfinite(d).any():
        return np.inf
    best = np.inf
    order = np.argsort(d)[:5]                                  # refine the 5 best grid points
    h = 2 * np.pi / n_grid
    f = lambda t: float(_dist_vec(P, w, delta * _dir_k3(Z, np.array([t])), kind)[0])
    for i in order:
        r = minimize_scalar(lambda t: (f(t) if np.isfinite(f(t)) else 1e9), bounds=(phi[i] - h, phi[i] + h),
                            method="bounded", options={"xatol": 1e-12})
        best = min(best, d[i], r.fun if r.fun < 1e8 else np.inf)
    return float(best)


def c_tv_k3(P, Z, n_grid=36000):
    """c_TV = min_{|u|=1, u in T} ||sum_k u_k p_k||_1 for K = 3 (exact 1-D)."""
    from scipy.optimize import minimize_scalar
    phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    U = _dir_k3(Z, phi)
    vals = np.abs(U.T @ P).sum(axis=1)
    i = int(np.argmin(vals)); h = 2 * np.pi / n_grid
    g = lambda t: float(np.abs(_dir_k3(Z, np.array([t])).T @ P).sum())
    r = minimize_scalar(g, bounds=(phi[i] - h, phi[i] + h), method="bounded", options={"xatol": 1e-12})
    return float(min(vals[i], r.fun))


def delta0_tv_k3(P, w, Z, n_grid=36000, tol=1e-9):
    """delta_0^TV(w) = max delta_0(u) over minimisers u of ||f_u||_1 (Theorem 3.2(i)); K = 3."""
    phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    U = _dir_k3(Z, phi)
    vals = np.abs(U.T @ P).sum(axis=1)
    c = vals.min()
    idx = np.where(vals <= c + tol)[0]
    return float(max(delta0_of_u(w, U[:, i]) for i in idx))
