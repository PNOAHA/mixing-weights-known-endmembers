"""Small-simplex likelihood polishing with a strict, observed-cell KKT gate.

LN, 2026-10-14. This module does not sample or write grid results.
The gap bounds the error in the average concave log likelihood, not risk.
"""
import numpy as np
from scipy.optimize import minimize


def polish_mle(P, counts, initial, gap_tol=1e-8):
    """Return (weights, dual_gap, accepted, reason); permit exact faces.

    Newton steps use the current positive face. An excluded component is
    reintroduced when its score violates the global KKT inequality. Only
    observed cells enter the likelihood, so zero-count private cells cause
    neither 0/0 nor artificial positive-support requirements.
    """
    observed = counts > 0
    A = np.asarray(P[:, observed], float)
    f = np.asarray(counts[observed], float) / np.sum(counts)
    K = len(initial)
    w = np.maximum(np.asarray(initial, float), 0.0)
    w /= w.sum()

    def evaluate(x):
        p = x @ A
        if np.any(p <= 0) or not np.isfinite(p).all():
            return -np.inf, None, None
        ratio = A / p
        gradient = ratio @ f
        hessian = (ratio * f) @ ratio.T
        return float(f @ np.log(p)), gradient, hessian

    def newton(x):
        for iteration in range(120):
            value, g, H = evaluate(x)
            if g is None:
                return x, np.inf, False, 'invalid_observed_support'
            gap = float(g.max() - x @ g)
            feasible = (x.min() >= 0 and abs(x.sum() - 1) <= 1e-12)
            if feasible and np.isfinite(gap) and gap <= gap_tol:
                return x, gap, True, 'active_face_newton'
            support = np.flatnonzero(x > 0)
            local_residual = np.ptp(g[support]) if len(support) > 1 else 0.0
            if local_residual <= min(gap_tol / 10, gap / 10):
                j = int(np.argmax(g))
                if x[j] == 0:
                    x *= 1 - 1e-6
                    x[j] = 1e-6
                    continue
            if len(support) <= 1:
                return x, gap, False, 'face_stalled'
            # Columns e_i-e_last span the affine tangent of this face.
            B = np.zeros((K, len(support) - 1))
            B[support[:-1], np.arange(len(support) - 1)] = 1
            B[support[-1], :] = -1
            try:
                step = B @ np.linalg.solve(B.T @ H @ B, B.T @ g)
            except np.linalg.LinAlgError:
                return x, gap, False, 'singular_face'
            negative = step < 0
            alpha = min(1.0, float(np.min(-x[negative] / step[negative]))) if negative.any() else 1.0
            boundary = negative & (np.abs(x + alpha * step) <= 1e-14)
            slope = float(g @ step)
            if not np.isfinite(slope) or slope < -1e-13:
                return x, gap, False, 'nonascent_step'
            taken = False
            for backtrack in range(50):
                trial = x + alpha * step
                if backtrack == 0:
                    trial[boundary] = 0
                trial = np.maximum(trial, 0)
                trial /= trial.sum()
                next_value, _, _ = evaluate(trial)
                if next_value >= value + 1e-4 * alpha * max(slope, 0.0) - 2e-14:
                    x = trial
                    taken = True
                    break
                alpha /= 2
            if not taken:
                return x, gap, False, 'line_search_stalled'
        _, g, _ = evaluate(x)
        gap = float(g.max() - x @ g) if g is not None else np.inf
        return x, gap, False, 'iteration_limit'

    best = newton(w.copy())
    if best[2]:
        return best
    # Scale by n and use the analytic gradient; finite differences of the
    # unscaled likelihood were the source of the residual numerical floor.
    def objective(x):
        value, g, _ = evaluate(x)
        return (-value, -g) if g is not None else (1e100, np.zeros(K))
    constraint = {'type': 'eq', 'fun': lambda x: x.sum() - 1,
                  'jac': lambda x: np.ones(K)}
    for start in (best[0], np.full(K, 1 / K)):
        result = minimize(objective, start, jac=True, method='SLSQP',
                          bounds=[(0, 1)] * K, constraints=constraint,
                          options={'ftol': 1e-14, 'maxiter': 500})
        candidate = np.maximum(result.x, 0)
        candidate /= candidate.sum()
        polished = newton(candidate)
        if polished[1] < best[1]:
            best = polished
        if best[2]:
            return best
    return best
