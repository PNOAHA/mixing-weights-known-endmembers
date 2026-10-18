# -*- coding: utf-8 -*-
"""S1 recheck: F1, K=4, theta=0.6, near-vertex lattice point (task 17, grid idx 164), n=6400.
Compares original fixed-iter MLE vs dual-gap/SLSQP-accepted MLE on the SAME multinomial draws.
Does not overwrite grid CSVs or figures; writes a JSON report only.
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
# Resolve paperA/code whether we sit in review folder or code folder
_cand = [
    os.path.join(HERE, 'code'),
    os.path.normpath(os.path.join(HERE, '..', '..', 'code')),
    os.path.normpath(os.path.join(HERE, '..')),
    HERE,
]
CODE = next(c for c in _cand if os.path.isfile(os.path.join(c, 'run_grid_d107.py')))
sys.path.insert(0, CODE)
from generator import tangent_basis, family_F1
import run_grid_d107 as rg

TID = 17
GRID_IDX = 164
N = 6400
REPS = 2000
THETA = 0.6
K = 4


def mle_original(P, Nmat, W0, iters_em=2000, iters_newton=20):
    """Exact pre-patch batch behaviour (LinAlgError aborts whole Newton loop)."""
    n = Nmat.sum(1, keepdims=True); W = W0.copy(); Z = tangent_basis(P.shape[0])
    for _ in range(iters_em):
        pw = W @ P
        W = W * ((np.where(Nmat > 0, Nmat / np.maximum(pw, 1e-300), 0.0)) @ P.T) / n
    def ll(W): return (Nmat * np.log(np.maximum(W @ P, 1e-300))).sum(1)
    for _ in range(iters_newton):
        pw = W @ P; R = P[None] / pw[:, None, :]
        grad = np.einsum('rm,rkm->rk', Nmat, R) @ Z
        H = np.einsum('rm,rkm,rlm->rkl', Nmat, R, R); H = np.einsum('ki,rkl,lj->rij', Z, H, Z)
        try: step = np.linalg.solve(H, grad[:, :, None])[:, :, 0] @ Z.T
        except np.linalg.LinAlgError: break
        alpha = np.ones((W.shape[0], 1)); ll0 = ll(W)
        for _bt in range(30):
            Wn = W + alpha * step; ok = (Wn.min(1) > 0) & (ll(Wn) >= ll0 - 1e-12); alpha[~ok] *= 0.5
            if ok.all(): break
        Wn = W + alpha * step; W = np.where((Wn.min(1) > 0)[:, None], Wn, W)
    return W


def main():
    t0 = time.time()
    rng = np.random.default_rng(20260919 + TID)
    Z = tangent_basis(K)
    P = family_F1(K, 2 * K, THETA, rng)
    Wlat = rg.simplex_lattice(K, 8)
    w = Wlat[GRID_IDX]
    assert np.allclose(w, (0.9625, 0.0125, 0.0125, 0.0125)), w

    # Reproduce RNG consumption of work_point up to this lattice index:
    # work_point creates dirs and moduli before MLE; we must burn the same RNG stream
    # as the original grid for this point only. Prefer matching review's reported CSV risk.
    # Original work_point seed = 1000*tid + i = 1000*17 + 164 = 17164
    seed = 1000 * TID + GRID_IDX
    rng_pt = np.random.default_rng(seed)
    # Burn same preliminary RNG uses as work_point before multinomial:
    _ = rg.dirs(K, Z, 40000)
    q = rg.point_q(P, w, Z)
    _ = rg.c_tv_and_delta0tv(P, w, Z, rg.dirs(K, Z, 40000))
    # moduli loop uses dirs U already built; modulus may call optimize (no RNG). deltas fixed.
    # So only multinomial consumes rng_pt after the preliminary deterministic stuff.
    # BUT c_tv and dirs don't use rng_pt. point_q doesn't. Only multinomial uses rng_pt in work_point.
    # work_point: rng = default_rng(seed); U=dirs(...); q=point_q; cTV=c_tv...(U); deltas; moduli; then multinomial.
    # None of those use rng except multinomial. Good.

    pw = w @ P
    Nmat = rng_pt.multinomial(N, pw, size=REPS)
    W0 = np.tile(np.ones(K) / K, (REPS, 1))

    W_old = mle_original(P, Nmat, W0.copy())
    gaps_old = rg.dual_gap(P, Nmat, W_old)
    risk_old = float(((W_old - w) ** 2).sum(1).mean())

    W_new, gaps_new, accepted, n_ref = rg.mle_vectorized(P, Nmat, W0.copy(), gap_tol=1e-8, refine=True)
    risk_new = float(((W_new - w) ** 2).sum(1).mean())
    x2_diff = ((W_new - w) ** 2).sum(1) - ((W_old - w) ** 2).sum(1)
    dist = np.linalg.norm(W_new - W_old, axis=1)

    out = {
        'task_id': TID,
        'grid_idx': GRID_IDX,
        'w': w.tolist(),
        'n': N,
        'reps': REPS,
        'theta': THETA,
        'K': K,
        'tau': float(q['tau']),
        'risk_old': risk_old,
        'risk_new': risk_new,
        'rel_change_pct': 100.0 * (risk_new - risk_old) / risk_old,
        'loss_diff_mean': float(x2_diff.mean()),
        'loss_diff_mean_se': float(x2_diff.std(ddof=1) / np.sqrt(REPS)),
        'dual_gap_old_max': float(gaps_old.max()),
        'dual_gap_old_med': float(np.median(gaps_old)),
        'dual_gap_new_max': float(gaps_new.max()),
        'dual_gap_new_med': float(np.median(gaps_new)),
        'accept_frac_new': float(accepted.mean()),
        'n_refined': int(n_ref),
        'max_euclid_weight_dist': float(dist.max()),
        'boundary_frac_old': float((W_old.min(1) < 1e-7).mean()),
        'boundary_frac_new': float((W_new.min(1) < 1e-7).mean()),
        'ratio_old': float(N * risk_old / q['tau']),
        'ratio_new': float(N * risk_new / q['tau']),
        'review_csv_risk': 0.0015318709511596168,
        'elapsed_sec': time.time() - t0,
    }
    out_path = os.path.join(HERE, 's1_near_vertex_recheck.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print('wrote', out_path)

if __name__ == '__main__':
    main()
