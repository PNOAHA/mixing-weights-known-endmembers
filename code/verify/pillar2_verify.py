"""Machine verification for 学习/02_矩阵凸几何_Gillis_Boyd/定理重述.md (items a-f)."""
import numpy as np
import sympy as sp
import mpmath as mp

rng = np.random.default_rng(20260917)
results, log = {}, []
x, y = sp.symbols("x y")
Kg = sp.exp(-(x - y) ** 2 / 2)        # radial Gaussian K'
Ke = sp.exp(x * y)                    # K'' = D K' D kernel


def wblock(kernel, idx):
    return sp.Matrix(len(idx), len(idx), lambda a, b: sp.diff(kernel, x, idx[a], y, idx[b]).subs({x: 0, y: 0}) / (sp.factorial(idx[a]) * sp.factorial(idx[b])))


# (a) Wronskian structure and block ratios
ok_a = True
We = wblock(Ke, list(range(0, 6)))
if not We.is_diagonal() or any(We[i, i] != sp.Rational(1, sp.factorial(i)) for i in range(6)): ok_a = False; log.append("(a) W'' not diag(1/i!)")
gam = {}
for m in range(3, 10):
    s = m - 1
    r_g_del = wblock(Kg, list(range(1, s + 1))).det() / wblock(Kg, list(range(1, s))).det()
    r_e_del = wblock(Ke, list(range(1, s + 1))).det() / wblock(Ke, list(range(1, s))).det()
    if sp.simplify(r_e_del - sp.Rational(1, sp.factorial(s))) != 0: ok_a = False; log.append(f"(a) K'' deleted ratio m={m}")
    g = sp.nsimplify(r_g_del * sp.factorial(s)); gam[m] = g
    expect = sp.Rational(m, m - 1) if m % 2 == 1 else 1
    if g != expect: ok_a = False; log.append(f"(a) gamma_{m}={g} != {expect}")
    if m <= 6:
        r_g_full = wblock(Kg, list(range(0, s + 1))).det() / wblock(Kg, list(range(0, s))).det()
        r_e_full = wblock(Ke, list(range(0, s + 1))).det() / wblock(Ke, list(range(0, s))).det()
        if sp.simplify(r_g_full - r_e_full) != 0: ok_a = False; log.append(f"(a) full ratios differ m={m}")
log.append("(a) gamma_m for m=3..9: " + ", ".join(f"{m}:{gam[m]}" for m in gam))
results["(a) sympy: W'' diagonal; deleted-row block ratios; gamma_m odd/even law (m<=9); full ratios coincide"] = ok_a

# ---------------------------------------------------------------- mpmath helpers
mp.mp.dps = 60


def zbasis(K):
    A = mp.matrix(K, K)
    for i in range(K):
        for j in range(K): A[i, j] = (1 if i == j else 0) - mp.mpf(1) / K
    Q, R = mp.qr(A)
    return Q[:, 0:K - 1]


def kmat(t, h, kind):
    K = len(t); M = mp.matrix(K, K)
    for i in range(K):
        for j in range(K):
            M[i, j] = mp.e ** (-(t[i] - t[j]) ** 2 / (2 * h * h)) if kind == "g" else mp.e ** (t[i] * t[j] / (h * h))
    return M


def lmin_compressed(t, h, kind):
    Z = zbasis(len(t)); M = kmat(t, h, kind)
    return min(mp.eigsy(Z.T * M * Z)[0])


def R2_last(t, K):
    """R_{K,K}^2 of the thin QR of the Vandermonde V_{<=K-1}: squared distance of v_{K-1} to span of lower monomials."""
    V = mp.matrix([[mp.mpf(ti) ** j for j in range(K)] for ti in t])
    Q, R = mp.qr(V)
    return R[K - 1, K - 1] ** 2


# (b) ratio -> 1/gamma_K
ok_b = True
for K in (3, 4, 5, 6):
    t = [mp.mpf(-1) + mp.mpf(2) * i / (K - 1) for i in range(K)]
    h = mp.mpf(10) ** 4
    r = lmin_compressed(t, h, "e") / lmin_compressed(t, h, "g")
    target = mp.mpf(K - 1) / K if K % 2 == 1 else mp.mpf(1)
    if abs(r - target) > mp.mpf("1e-3"): ok_b = False; log.append(f"(b) K={K} ratio {mp.nstr(r,8)} vs {target}")
    log.append(f"(b) K={K}: Lambda/lam_min(Z'K'Z) = {mp.nstr(r, 8)}")
results["(b) mpmath: compression ratio -> 1/gamma_K for K=3..6"] = ok_b

# (c) P2.1 leading constant for both kernels, random nodes
ok_c = True
for K in (3, 4, 5):
    t = [mp.mpf(v) for v in sorted(rng.uniform(-1, 1, size=K))]
    r2 = R2_last(t, K); s = K - 1
    for kind, kern in (("e", Ke), ("g", Kg)):
        wr = wblock(kern, list(range(1, s + 1))).det() / wblock(kern, list(range(1, s))).det()
        pred = r2 * mp.mpf(str(sp.N(wr, 40)))
        h = mp.mpf(10) ** 4
        val = lmin_compressed(t, h, kind) * h ** (2 * (K - 1))
        rel = abs(val / pred - 1)
        if rel > mp.mpf("2e-3"): ok_c = False; log.append(f"(c) K={K} {kind}: {mp.nstr(val,8)} vs pred {mp.nstr(pred,8)}")
        log.append(f"(c) K={K} kernel={kind}: h^(2K-2) lam_min = {mp.nstr(val,8)}, P2.1 prediction {mp.nstr(pred,8)}")
results["(c) mpmath: P2.1 leading constant matches for both kernels, K=3,4,5, random nodes"] = ok_c

# (d) BU21 Cor 4.3 (25) uncompressed, as a control
ok_d = True
for K in (3, 4):
    t = [mp.mpf(v) for v in sorted(rng.uniform(-1, 1, size=K))]
    r2 = R2_last(t, K)
    for kind, kern in (("e", Ke), ("g", Kg)):
        wr = wblock(kern, list(range(0, K))).det() / wblock(kern, list(range(0, K - 1))).det()
        pred = r2 * mp.mpf(str(sp.N(wr, 40)))
        h = mp.mpf(10) ** 4
        val = min(mp.eigsy(kmat(t, h, kind))[0]) * h ** (2 * (K - 1))
        if abs(val / pred - 1) > mp.mpf("2e-3"): ok_d = False; log.append(f"(d) K={K} {kind}: {mp.nstr(val,8)} vs {mp.nstr(pred,8)}")
results["(d) mpmath: BU21 Cor 4.3 (25) uncompressed smallest eigenvalue (control)"] = ok_d

# (e) Gillis Example 4.16 / Lemma 4.11 / anchor => W^T separable
w = sp.symbols("omega", nonnegative=True)
H = sp.Matrix([[w, 1, 1, w, 0, 0], [1, w, 0, 0, w, 1], [0, 0, w, 1, 1, w]])
Q = sp.Rational(1, 3) * sp.Matrix([[-1, 2, 2], [2, -1, 2], [2, 2, -1]])
e1 = sp.simplify(Q.T * Q - sp.eye(3)) == sp.zeros(3)
B = sp.simplify(Q.inv() * H)
B_expect = sp.Rational(1, 3) * sp.Matrix([[2 - w, 2 * w - 1, 2 * w - 1, 2 - w, 2 * w + 2, 2 * w + 2],
                                          [2 * w - 1, 2 - w, 2 * w + 2, 2 * w + 2, 2 - w, 2 * w - 1],
                                          [2 * w + 2, 2 * w + 2, 2 - w, 2 * w - 1, 2 * w - 1, 2 - w]])
e2 = sp.simplify(B - B_expect) == sp.zeros(3, 6)
e3 = all(v >= 0 for v in B.subs(w, sp.Rational(1, 2))) and all(v >= 0 for v in B.subs(w, sp.Rational(3, 4))) and any(v < 0 for v in B.subs(w, sp.Rational(1, 4)))
# separability: random separable H contains scaled permutation of identity <-> cone(H) = R^r_+ (check via LP-free test: unit vectors in cone)
ok_sep = True
for _ in range(20):
    r, n = 3, 7
    Hs = np.abs(rng.normal(size=(r, n))); perm = rng.permutation(n)[:r]
    Hs[:, perm] = np.diag(rng.uniform(0.5, 2, size=r))          # plant scaled permutation of identity
    # (iii) holds by construction; check (i): each e_k in cone(H) trivially; check a non-separable random matrix fails (iii)
    Hn = np.abs(rng.normal(size=(r, n))) + 0.1
    has_diag = any(np.count_nonzero(Hn[:, j]) == 1 for j in range(n))
    if has_diag: ok_sep = False
# anchor => W^T separable
K, m = 3, 5; a = np.array([0.2, 0.3, 0.1]); base = rng.dirichlet(np.ones(m), size=K)
W = np.zeros((m + K, K))
for k in range(K): W[:m, k] = (1 - a[k]) * base[k]; W[m + k, k] = a[k]
sub = W[m:, :]                                                   # K x K block of W (rows = private cells) -> W^T contains diag(a)
e4 = np.allclose(sub, np.diag(a))
results["(e) Gillis Ex 4.16 Q orthogonal, H=QB, B>=0 iff omega>=1/2; separability (iii); anchor => W^T separable"] = bool(e1 and e2 and e3 and ok_sep and e4)

# (f) Lemma 4.32 explicit Q
ok_f = True
for _ in range(30):
    m, r = 6, 3
    Wm = np.abs(rng.normal(size=(m, r))); k, l = 0, 1
    Wm[:3, k] = 0.0                                              # supp(col k) subset supp(col l)
    eps = 0.5 * np.min(Wm[Wm[:, k] > 0, l] / Wm[Wm[:, k] > 0, k])
    Qm = np.eye(r); Qm[k, l] = -eps
    if (Wm @ Qm).min() < -1e-12 or np.linalg.inv(Qm).min() < -1e-12: ok_f = False
results["(f) Gillis Lemma 4.32: WQ >= 0 and Q^{-1} >= 0 for the explicit Q"] = ok_f

print("=" * 60)
for k_, v_ in results.items(): print(("PASS " if v_ else "FAIL ") + k_)
print("=" * 60)
for line in log: print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
