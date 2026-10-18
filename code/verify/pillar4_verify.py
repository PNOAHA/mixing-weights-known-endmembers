"""Machine verification for 学习/04_熵OT_PeyreCuturi/定理重述.md (pillar 4, narrowed scope).

Discrete ground space X = {x_1..x_m} in [0,1], endmembers mu_k = p_k (histograms),
alpha = Phi(w) = sum_k w_k p_k, nu = a fixed histogram. Entropic OT in Feydy's convention
    OT_eps(a,b) = min_{pi in U(a,b)} <C,pi> + eps KL(pi | a (x) b)
solved by log-domain Sinkhorn to 1e-13; Sinkhorn divergence S_eps = OT(a,b) - OT(a,a)/2 - OT(b,b)/2.
Items (a)-(i) correspond to the '机器核验' section of the note.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linprog

rng = np.random.default_rng(20260917)
results, log = {}, []


# ------------------------------------------------------------------ entropic OT machinery
def sinkhorn(a, b, C, eps, tol=1e-13, maxit=20000):
    """Feydy (20)-(21) fixed point in the log domain. Returns extended potentials (f, g) on the
    whole grid (Sinkhorn mapping), the plan pi and OT_eps(a,b) = <a,f> + <b,g> (Feydy (13))."""
    la = np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), -np.inf)
    lb = np.where(b > 0, np.log(np.where(b > 0, b, 1.0)), -np.inf)
    f = np.zeros(len(a)); g = np.zeros(len(b))
    for it in range(maxit):
        f_new = -eps * logsumexp(lb[None, :] + (g[None, :] - C) / eps, axis=1)
        g_new = -eps * logsumexp(la[:, None] + (f_new[:, None] - C) / eps, axis=0)
        err = np.max(np.abs(f_new - f)) + np.max(np.abs(g_new - g))
        f, g = f_new, g_new
        if err < tol:
            break
    pi = np.exp((f[:, None] + g[None, :] - C) / eps) * a[:, None] * b[None, :]
    return f, g, pi, a @ f + b @ g


def sym_potential(a, C, eps):
    """Unique symmetric potential p of OT_eps(a,a): p = (f+g)/2 (optimal pair unique up to (f+K,g-K))."""
    f, g, _, val = sinkhorn(a, a, C, eps)
    return 0.5 * (f + g), val


def S_eps(a, b, C, eps):
    """Feydy (3): OT(a,b) - OT(a,a)/2 - OT(b,b)/2."""
    return sinkhorn(a, b, C, eps)[3] - 0.5 * sinkhorn(a, a, C, eps)[3] - 0.5 * sinkhorn(b, b, C, eps)[3]


def L_PC(a, b, C, eps):
    """Peyre-Cuturi (4.2): <P,C> - eps H(P), H(P) = -sum P (log P - 1), at the optimal plan."""
    pi = sinkhorn(a, b, C, eps)[2]
    P = pi[pi > 0]
    return np.sum(pi * C) + eps * np.sum(P * (np.log(P) - 1.0))


def shannon(a):
    a = a[a > 0]; return -np.sum(a * np.log(a))


def mmd_sq(a, b, k):
    """||a-b||_k^2 = xi^T k xi (Feydy (2) without the 1/2)."""
    xi = a - b; return xi @ k @ xi


# ------------------------------------------------------------------ ground space and endmembers
m = 7
x = np.sort(rng.uniform(0, 1, m))                       # 1-D points in [0,1]
C2 = (x[:, None] - x[None, :]) ** 2                     # quadratic cost (Gaussian Gibbs kernel)
C1 = np.abs(x[:, None] - x[None, :])                    # L1 cost (Laplacian Gibbs kernel)
K = 3
P = rng.dirichlet(np.ones(m), size=K); P = 0.8 * P + 0.2 / m   # endmembers (rows), full support
Z = np.linalg.qr(np.vstack([np.ones(K), rng.normal(size=(K - 1, K))]).T)[0][:, 1:]  # basis of T
Phi = lambda w: w @ P
w0 = np.array([0.5, 0.3, 0.2]); nu = Phi(w0)
wa = np.array([0.2, 0.5, 0.3])


# (a) entropy convention: S_eps identical from OT_eps (KL) or from L_PC; offset = eps(1+H(a)+H(b))
eps = 0.05
ok_a = True
for (a, b) in [(Phi(wa), nu), (rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m)))]:
    otF = sinkhorn(a, b, C2, eps)[3]; L = L_PC(a, b, C2, eps)
    off = otF - L - eps * (1 + shannon(a) + shannon(b))
    S_F = S_eps(a, b, C2, eps)
    S_L = L - 0.5 * L_PC(a, a, C2, eps) - 0.5 * L_PC(b, b, C2, eps)
    ok_a &= abs(off) < 1e-10 and abs(S_F - S_L) < 1e-10
    log.append(f"(a) eps={eps}: OT_KL - L_PC - eps(1+H(a)+H(b)) = {off:.2e};  S_eps(KL) - S_eps(PC) = {S_F - S_L:.2e}")
results["(a) OT_KL = L_PC + eps(1+H(a)+H(b)); debiased S_eps identical under both entropy conventions"] = bool(ok_a)

# (b) Genevay-2018 sharp version (2<C,pi> - ...) differs from 2 S_eps; eps->inf: S_eps -> MMD_{-C}/2, sharp -> MMD_{-C}; eps->0: S_eps -> OT_0
a, b = Phi(wa), nu
def sharp(a, b, e):
    return 2 * np.sum(sinkhorn(a, b, C2, e)[2] * C2) - np.sum(sinkhorn(a, a, C2, e)[2] * C2) - np.sum(sinkhorn(b, b, C2, e)[2] * C2)
d_mid = sharp(a, b, 0.05) - 2 * S_eps(a, b, C2, 0.05)
lim_F = [abs(S_eps(a, b, C2, e) - 0.5 * mmd_sq(a, b, -C2)) for e in (1e2, 1e3, 1e4)]
lim_G = [abs(sharp(a, b, e) - mmd_sq(a, b, -C2)) for e in (1e2, 1e3, 1e4)]
Aeq = np.zeros((2 * m, m * m))
for i in range(m):
    Aeq[i, i * m:(i + 1) * m] = 1; Aeq[m + i, i::m] = 1
ot0 = linprog(C2.ravel(), A_eq=Aeq, b_eq=np.r_[a, b], bounds=(0, None), method="highs").fun
lim0 = [abs(S_eps(a, b, C2, e) - ot0) for e in (1e-2, 3e-3, 1e-3)]
ok_b = abs(d_mid) > 1e-4 and lim_F[-1] < 1e-3 and lim_G[-1] < 1e-3 and lim_F[-1] < lim_F[0] and lim0[-1] < lim0[0] and lim0[-1] < 2e-3
log.append(f"(b) sharp(G18) - 2 S_eps at eps=.05: {d_mid:.4f} (nonzero => definitions differ);  |S_eps - MMD/2| eps=1e2,1e3,1e4: {np.round(lim_F,6)};  |sharp - MMD|: {np.round(lim_G,6)};  |S_eps - OT_0| eps=1e-2,3e-3,1e-3: {np.round(lim0,5)}")
results["(b) [25] sharp 2W-... != 2 S_eps; both eps->inf limits of Feydy (4) hold (MMD/2 vs MMD); eps->0 -> OT_0"] = bool(ok_b)

# (c) joint convexity: L_PC jointly convex (PC Prop 4.6); OT_KL NOT jointly convex (Feydy p.5) but convex in each variable
ntest = 60; ok_c = True
for eps in (1.0, 0.05):
    viol_L = 0; viol_F = 0; viol_sep = 0
    F = lambda a, b: sinkhorn(a, b, C2, eps)[3]
    for _ in range(ntest):
        a1, b1, a2, b2 = (rng.dirichlet(np.ones(m)) for _ in range(4))
        am, bm = 0.5 * (a1 + a2), 0.5 * (b1 + b2)
        if L_PC(am, bm, C2, eps) > 0.5 * (L_PC(a1, b1, C2, eps) + L_PC(a2, b2, C2, eps)) + 1e-9: viol_L += 1
        if F(am, bm) > 0.5 * (F(a1, b1) + F(a2, b2)) + 1e-9: viol_F += 1
        if F(am, b1) > 0.5 * (F(a1, b1) + F(a2, b1)) + 1e-9: viol_sep += 1
    ok_c &= viol_L == 0 and viol_F > 0 and viol_sep == 0
    log.append(f"(c) eps={eps}, {ntest} random midpoint tests: joint-convexity violations L_PC={viol_L}, OT_KL={viol_F}; separate-convexity violations OT_KL={viol_sep}")
results["(c) L_PC jointly convex; OT_KL convex in each variable but NOT jointly (convention-dependent)"] = bool(ok_c)

# (d) positivity, definiteness, Hausdorff sandwich H_eps <= S_eps (Feydy Thm 1 (5)-(6), Def 2, Sec 2.3), Gaussian and Laplacian kernels
ok_d = True
for C in (C2, C1):
    for eps in (0.02, 0.2, 2.0):
        for _ in range(8):
            a, b = rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m))
            S = S_eps(a, b, C, eps)
            pa, _ = sym_potential(a, C, eps); pb, _ = sym_potential(b, C, eps)
            H = 0.5 * (a - b) @ (pb - pa)
            ok_d &= S > 0 and H >= -1e-12 and H <= S + 1e-12
        ok_d &= abs(S_eps(a, a, C, eps)) < 1e-11
log.append(f"(d) S_eps>0 for a!=b, S_eps(a,a)=0, 0<=H_eps<=S_eps on 48 random pairs x 2 costs: {'all hold' if ok_d else 'VIOLATION'}")
results["(d) S_eps > 0 for a != b, S_eps(a,a) = 0, 0 <= H_eps <= S_eps"] = bool(ok_d)

# (e) J_eps(w) = S_eps(Phi(w), nu): gradient formula <mu_j, f - p> (Feydy (26)) vs finite differences; Hessian on T PD;
#     gradient on T vanishes at w0 (nu = Phi(w0)) and Hessian PD there
eps = 0.05
J = lambda w: S_eps(Phi(w), nu, C2, eps)
def grad_formula(w):
    a = Phi(w); f = sinkhorn(a, nu, C2, eps)[0]; p = sym_potential(a, C2, eps)[0]
    return P @ (f - p)                      # component j = <mu_j, f - p>
def hess_T(w, Jf=J, h=1e-3):
    Hm = np.zeros((K - 1, K - 1))
    for i in range(K - 1):
        for j in range(K - 1):
            vi, vj = Z[:, i], Z[:, j]
            Hm[i, j] = (Jf(w + h * vi + h * vj) - Jf(w + h * vi - h * vj) - Jf(w - h * vi + h * vj) + Jf(w - h * vi - h * vj)) / (4 * h * h)
    return 0.5 * (Hm + Hm.T), Hm
ok_e = True
for w in (wa, np.array([0.1, 0.1, 0.8])):
    gT_formula = Z.T @ grad_formula(w)
    gT_fd = np.array([(J(w + 1e-5 * Z[:, i]) - J(w - 1e-5 * Z[:, i])) / 2e-5 for i in range(K - 1)])
    Hs, Hraw = hess_T(w)
    ev = np.linalg.eigvalsh(Hs)
    ok_e &= np.max(np.abs(gT_formula - gT_fd)) < 1e-7 and ev.min() > 0 and np.max(np.abs(Hraw - Hraw.T)) < 1e-6
    log.append(f"(e) w={w}: |grad_T formula - FD| = {np.max(np.abs(gT_formula - gT_fd)):.2e}; Hessian_T eigenvalues {np.round(ev,6)}; asymmetry of raw FD Hessian {np.max(np.abs(Hraw - Hraw.T)):.1e}")
g0 = Z.T @ grad_formula(w0); ev0 = np.linalg.eigvalsh(hess_T(w0)[0])
ok_e &= np.max(np.abs(g0)) < 1e-9 and ev0.min() > 0
log.append(f"(e) at w0 (nu = Phi(w0)): |grad_T| = {np.max(np.abs(g0)):.1e}, Hessian_T eigenvalues {np.round(ev0,6)}")
results["(e) grad J = <mu_j, f - p> on T; Hessian_T > 0 (strict convexity); stationary + PD at w0"] = bool(ok_e)

# (f) second directional derivative = <xi_v, d/dt (f^{alpha_t,nu} - p^{alpha_t})> (potential derivative), v in T
ok_f = True
for v in (Z[:, 0], Z[:, 1], (Z[:, 0] + Z[:, 1]) / np.sqrt(2)):
    w = wa; xi = v @ P; h = 1e-4
    def fp(t):
        a = Phi(w + t * v); f = sinkhorn(a, nu, C2, eps)[0]; p = sym_potential(a, C2, eps)[0]
        return f - p                                       # <xi,.> is shift-invariant (xi has mass 0)
    Q_pot = xi @ (fp(h) - fp(-h)) / (2 * h)
    Q_fd = (J(w + h * v) - 2 * J(w) + J(w - h * v)) / (h * h)
    ok_f &= abs(Q_pot - Q_fd) < 1e-5 * max(1, abs(Q_fd))
    log.append(f"(f) v={np.round(v,3)}: Q via potential derivative {Q_pot:.8f}, via 2nd FD of J {Q_fd:.8f}")
results["(f) 2nd directional derivative of J = <xi_v, D_alpha(f - p)[xi_v]>"] = bool(ok_f)

# (g) Hadamard derivative of potentials = implicit-function solve of the linearised Schrodinger system
#     (Goldfeld Lemma 6-8 / Hundrieser Lemma 5.7); A^X A^Y has row sums 1 - pi_{i,y1}/a_i < 1 (Hundrieser Remark 5.9(ii))
eps = 0.05; a = Phi(wa); b = nu
f, g, pi, _ = sinkhorn(a, b, C2, eps)
Kmat = np.exp((f[:, None] + g[None, :] - C2) / eps)      # e^{(f+g-C)/eps}
D_u = np.zeros((2 * m + 1, 2 * m)); D_ab = np.zeros((2 * m + 1, 2 * m))
for i in range(m):
    D_u[i, i] = np.sum(b * Kmat[i, :]) / eps; D_u[i, m:] = b * Kmat[i, :] / eps; D_ab[i, m:] = Kmat[i, :]
for j in range(m):
    D_u[m + j, :m] = a * Kmat[:, j] / eps; D_u[m + j, m + j] = np.sum(a * Kmat[:, j]) / eps; D_ab[m + j, :m] = Kmat[:, j]
D_u[2 * m, 0] = 1; D_u[2 * m, m] = -1                     # normalisation f_1 - g_1 = 0 (Goldfeld: phi1(x0) = phi2(x0))
xi = Z[:, 0] @ P; rhs = -D_ab @ np.r_[xi, np.zeros(m)]
du = np.linalg.lstsq(D_u, rhs, rcond=None)[0]
def norm_pot(a_):
    f_, g_, _, _ = sinkhorn(a_, b, C2, eps); s = 0.5 * (f_[0] - g_[0]); return np.r_[f_ - s, g_ + s]
h = 1e-5; du_fd = (norm_pot(a + h * xi) - norm_pot(a - h * xi)) / (2 * h)
err_g = np.max(np.abs(du - du_fd)); resid = np.max(np.abs(D_u @ du - rhs))
AX = (pi / a[:, None])[:, 1:]; AY = (pi / b[None, :])[:, 1:].T          # A^X: R^{m-1} -> R^m, A^Y: R^m -> R^{m-1}
rows = (AX @ AY).sum(axis=1); rows_pred = 1 - pi[:, 0] / a
ok_g = err_g < 1e-6 and resid < 1e-9 and np.all(rows < 1) and np.max(np.abs(rows - rows_pred)) < 1e-12
log.append(f"(g) |implicit-function derivative - FD| = {err_g:.2e} (residual {resid:.1e}); max row sum of A^X A^Y = {rows.max():.4f} = 1 - min_i pi_i1/a_i")
results["(g) potential derivative = -(D_u Psi)^{-1} D_ab Psi (linearised Schrodinger system); ||A^X A^Y||_inf < 1"] = bool(ok_g)
# (g2) contraction constant: 1 - ||A^X A^Y||_inf = min_x pi_{x y1}/a_x ; log of it vs 1/eps
cc = []
for e in (0.2, 0.1, 0.05, 0.025):
    _, _, pi_e, _ = sinkhorn(a, b, C2, e)
    cc.append((e, float((pi_e[:, 0] / a).min())))
slope_g = np.polyfit([1 / e for e, _ in cc], [np.log(v) for _, v in cc], 1)[0]
ok_g2 = all(cc[i + 1][1] < cc[i][1] for i in range(len(cc) - 1)) and slope_g < 0
log.append(f"(g2) 1 - ||A^X A^Y||_inf = min_x pi_x,y1/a_x for eps=0.2,0.1,0.05,0.025: {[(e, f'{v:.2e}') for e, v in cc]}; slope of log vs 1/eps = {slope_g:.4f} (= -(C_{{x,y1}}-f-g offsets) => e^{{-c/eps}} law)")
results["(g2) contraction constant 1-||A^X A^Y|| decays like exp(-c/eps)"] = bool(ok_g2)

# (h) eps -> inf: Hessian_T(eps) -> quadratic form ||xi_v||^2_{-C}; for C=(x-y)^2 in 1-D this is 2 (sum_k v_k m_k)^2, rank 1 for K=3
mk = P @ x                                                    # endmember means
Hlim = np.array([[2 * (Z[:, i] @ mk) * (Z[:, j] @ mk) for j in range(K - 1)] for i in range(K - 1)])
Hlim2 = np.array([[(Z[:, i] @ P) @ (-C2) @ (Z[:, j] @ P) for j in range(K - 1)] for i in range(K - 1)])
lam2 = []
for e in (1e1, 1e2, 1e3, 1e4):
    Jl = lambda w, e=e: S_eps(Phi(w), nu, C2, e)
    Hm = hess_T(wa, Jl, h=1e-2)[0]
    lam2.append((e, np.linalg.eigvalsh(Hm).min(), np.max(np.abs(Hm - Hlim))))
slope = np.polyfit(np.log([l[0] for l in lam2[1:]]), np.log([max(abs(l[1]), 1e-12) for l in lam2[1:]]), 1)[0]
ok_h = np.max(np.abs(Hlim - Hlim2)) < 1e-12 and lam2[-1][2] < 1e-3 and lam2[-1][2] < lam2[0][2] and abs(lam2[-1][1]) < lam2[0][1]
log.append(f"(h) ||xi||^2_(-C) form == 2(v.m)^2 form: {np.max(np.abs(Hlim - Hlim2)):.1e}; (eps, lambda_min(Hess_T), |Hess_T - limit|) = {[(l[0], round(l[1],7), round(l[2],6)) for l in lam2]}; log-log slope of lambda_min vs eps (eps>=1e2) = {slope:.3f}")
results["(h) Hessian_T(eps) -> MMD_{-C} Gram (rank <= dim X = 1 for K = 3): lambda_min -> 0"] = bool(ok_h)

# (i) W_2^2(Phi(w),Phi(w')) <= D^2 TV <= D^2 ||v||_1 / 2 (1-D discrete W_2 via quantiles)
def w2sq_1d(a, b, x):
    Fa, Fb = np.cumsum(a), np.cumsum(b); grid = np.unique(np.r_[0.0, Fa, Fb]); out = 0.0
    for u0, u1 in zip(grid[:-1], grid[1:]):
        um = 0.5 * (u0 + u1); qa = x[min(np.searchsorted(Fa, um), len(x) - 1)]; qb = x[min(np.searchsorted(Fb, um), len(x) - 1)]; out += (u1 - u0) * (qa - qb) ** 2
    return out
D = x.max() - x.min(); ok_i = True
for _ in range(50):
    w1, w2 = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K)); a, b = Phi(w1), Phi(w2)
    tv = 0.5 * np.sum(np.abs(a - b)); W2 = w2sq_1d(a, b, x)
    ok_i &= W2 <= D * D * tv + 1e-12 and tv <= 0.5 * np.sum(np.abs(w1 - w2)) + 1e-12
log.append(f"(i) W_2^2 <= D^2 TV and TV <= ||w-w'||_1/2 on 50 random pairs: {'hold' if ok_i else 'VIOLATION'}")
results["(i) W_2^2(Phi(w),Phi(w')) <= diam^2 TV <= diam^2 ||w-w'||_1 / 2"] = bool(ok_i)

print("=" * 60)
for k_, v_ in results.items(): print(("PASS " if v_ else "FAIL ") + k_)
print("=" * 60)
for line in log: print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
