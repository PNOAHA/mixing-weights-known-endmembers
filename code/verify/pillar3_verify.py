"""Machine verification for 学习/03_渐近与边界_vdVaart/定理重述.md §5 (Theorem P3.1), items (a)-(e)."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kstest, norm, chi2, poisson

rng = np.random.default_rng(20260917)
results, log = {}, []


def mle_theta(r):
    """argmax over [0,1] of sum log(1 + theta (r-1)); concave -> 1-D bounded optimisation."""
    if (r - 1).sum() <= 0:
        return 0.0
    f = lambda th: -np.log1p(th * (r - 1)).sum()
    res = minimize_scalar(f, bounds=(0.0, 1.0 - 1e-9), method="bounded", options={"xatol": 1e-10})
    return res.x


# ------------------------------------------------------------ (a),(b) regular case
m = 6
p1 = rng.dirichlet(np.ones(m)); p1 = 0.7 * p1 + 0.3 / m
p2 = rng.dirichlet(np.ones(m)); p2 = 0.7 * p2 + 0.3 / m           # mu_2 << mu_1 (full support), gamma finite
gamma = (p2 ** 2 / p1).sum(); I0 = gamma - 1
n, reps = 2000, 20000
X = rng.choice(m, size=(reps, n), p=p1)
rvals = p2 / p1
th = np.array([mle_theta(rvals[x]) for x in X])
p0 = (th == 0).mean()
pos = np.sqrt(n) * th[th > 0]
ks_half = kstest(pos, lambda z: 2 * norm.cdf(z * np.sqrt(I0)) - 1).statistic     # half-normal with variance 1/I0
ok_a = abs(p0 - 0.5) < 0.02 and ks_half < 0.02
log.append(f"(a) I0={I0:.4f}  P(theta_hat=0)={p0:.4f}  KS(sqrt(n)theta|>0, half-normal 1/I0)={ks_half:.4f}")
results["(a) regular case: P(theta_hat=0)->1/2 and half-normal limit"] = ok_a
LR = np.array([2 * np.log1p(t * (rvals[x] - 1)).sum() for t, x in zip(th, X)])
pLR0 = (LR <= 1e-9).mean(); ks_chi = kstest(LR[LR > 1e-9], chi2(df=1).cdf).statistic
ok_b = abs(pLR0 - 0.5) < 0.02 and ks_chi < 0.02
log.append(f"(b) P(LR=0)={pLR0:.4f}  KS(LR|>0, chi2_1)={ks_chi:.4f}")
results["(b) regular case: LR -> 1/2 chi2_0 + 1/2 chi2_1"] = ok_b

# ------------------------------------------------------------ (c) non-regular case, b > 0
p1n = np.r_[p1 * 0.9, 0.0, 0.0] ; p1n /= p1n.sum()
p2n = np.r_[p2 * 0.6, 0.25, 0.15]; p2n /= p2n.sum()                # mu_2 has mass b on {p1=0}
b = p2n[p1n == 0].sum()
Xn = rng.choice(m + 2, size=(reps // 4, n), p=p1n)
rn = np.where(p1n > 0, p2n / np.where(p1n > 0, p1n, 1), 0.0)
thn = np.array([mle_theta(rn[x]) for x in Xn])
p0n = (thn == 0).mean()
ts = np.array([0.5, 1.0, 2.0, 4.0])
llr = np.array([[np.log1p((t / n) * (rn[x] - 1)).sum() for t in ts] for x in Xn[:2000]])
mean_llr = llr.mean(axis=0)
ok_c = p0n > 0.99 and np.all(np.abs(mean_llr - (-b * ts)) < 0.03 * np.maximum(1, b * ts))
# N_n under theta = t/n vs Poisson(t b)
t = 3.0; theta_alt = t / n; palt = (1 - theta_alt) * p1n + theta_alt * p2n
Nn = rng.binomial(n, palt[p1n == 0].sum(), size=20000)
ks_pois = max(abs(np.mean(Nn <= k) - poisson.cdf(k, t * b)) for k in range(0, 12))
ok_c = ok_c and ks_pois < 0.01
log.append(f"(c) b={b:.4f}  P(theta_hat=0)={p0n:.4f}  E[l_n(t/n)-l_n(0)]={np.round(mean_llr,4)} vs -bt={np.round(-b*ts,4)}  sup|F_Nn - Pois(tb)|={ks_pois:.4f}")
results["(c) non-regular case: MLE=0 w.p.->1, linear local limit -bt, Poisson counts at scale 1/n"] = bool(ok_c)

# ------------------------------------------------------------ (d) I0 vs A1's tangent information at e_1
P = np.vstack([p1, p2]); w0 = np.array([1.0, 0.0])
pw = w0 @ P; G = (P / pw) @ P.T
u = np.array([-1, 1]) / np.sqrt(2)
results["(d) I0 = gamma-1 equals 2 u'G(e_1)u (theta-parametrisation)"] = abs(2 * u @ G @ u - I0) < 1e-12

# ------------------------------------------------------------ (e) Hellinger first/second order
def H2(p, q): return 0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()
ok_e = True
for theta in (1e-2, 1e-3, 1e-4):
    hR = H2((1 - theta) * p1 + theta * p2, p1)         # regular: ~ theta^2 I0 / 8
    hN = H2((1 - theta) * p1n + theta * p2n, p1n)      # non-regular: ~ theta b / 2
    if abs(hR / (theta ** 2 * I0 / 8) - 1) > 0.05 * (1 + 100 * theta): ok_e = False
    if abs(hN / (theta * b / 2) - 1) > 0.05 * (1 + 100 * theta): ok_e = False
    log.append(f"(e) theta={theta}: H2_reg/(theta^2 I0/8)={hR/(theta**2*I0/8):.4f}  H2_nonreg/(theta b/2)={hN/(theta*b/2):.4f}")
results["(e) Hellinger: second order (regular) vs first order (non-regular)"] = ok_e

print("=" * 60)
for k_, v_ in results.items(): print(("PASS " if v_ else "FAIL ") + k_)
print("=" * 60)
for line in log: print("  " + line)
print("ALL PASS" if all(results.values()) else "SOME FAIL")
