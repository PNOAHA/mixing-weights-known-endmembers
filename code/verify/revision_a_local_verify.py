"""Small deterministic checks for revision A; no Monte Carlo or HPC work.

Run with the bundled Python. Existing local package directories are used only
when the bundled runtime lacks sympy/scipy; no package is installed or changed.
The output records the executable and actual library origins.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time


def load_existing_math_packages():
    dll_handles = []
    base = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "miniconda3"
    candidates = [base / "envs" / "geo-figs" / "Lib" / "site-packages",
                  base / "Lib" / "site-packages"]
    if any(importlib.util.find_spec(name) is None for name in ("sympy", "scipy")):
        for path in candidates:
            if path.is_dir() and str(path) not in sys.path:
                sys.path.append(str(path))
        dll_path = base / "envs" / "geo-figs" / "Library" / "bin"
        if os.name == "nt" and dll_path.is_dir():
            dll_handles.append(os.add_dll_directory(str(dll_path)))
    return dll_handles


DLL_HANDLES = load_existing_math_packages()
import numpy as np
import sympy as sp
from scipy.optimize import minimize
import scipy


def symbolic_checks():
    t = sp.symbols("t", real=True)
    s, n = sp.symbols("s n", positive=True)
    q = sp.cos(sp.pi * t / (2 * s)) ** 2 / s
    dq = sp.diff(q, t)
    fisher_integrand = sp.pi ** 2 / s ** 3 * sp.sin(sp.pi * t / (2 * s)) ** 2
    assert sp.simplify(dq ** 2 / q - fisher_integrand) == 0
    normalization = sp.simplify(sp.integrate(q, (t, -s, s)))
    mean_score = sp.simplify(sp.integrate(dq, (t, -s, s)))
    prior_fisher = sp.simplify(sp.integrate(fisher_integrand, (t, -s, s)))
    assert normalization == 1 and mean_score == 0
    assert prior_fisher == sp.pi ** 2 / s ** 2
    matrices = {str(d): str(prior_fisher * sp.eye(d)) for d in (1, 2, 3, 4)}

    a, b, c = sp.symbols("a b c", real=True)
    information = sp.Matrix([[a, b], [b, c]])
    C = information.inv()
    tau = sp.trace(C)
    assert sp.simplify(sp.trace(C * information * C.T) - tau) == 0
    exact_prior = prior_fisher * sp.trace(C * C.T)
    assert sp.simplify(exact_prior - prior_fisher * sp.trace(information ** -2)) == 0
    I3 = sp.Matrix([[5, 1, 1], [1, 4, 0], [1, 0, 3]])
    C3 = I3.inv()
    assert C3 * I3 * C3.T == C3
    assert sp.trace(C3 * C3.T) == sp.trace(I3 ** -2)

    rho, d, tau0, T2 = sp.symbols("rho d tau0 T2", positive=True)
    eta = rho * sp.sqrt(d) * n ** (-sp.Rational(1, 4))
    normalized_lower = 1 / (1 / (1 - eta) + sp.pi ** 2 * T2 / (sp.sqrt(n) * tau0))
    assert sp.limit(normalized_lower, n, sp.oo) == 1
    N, A, B = sp.symbols("N A B", positive=True)
    upper_expression = ((1 + eta) * (1 + A / sp.sqrt(n) + B / n)
                        + n * sp.exp(-n / N))
    assert sp.limit(upper_expression, n, sp.oo) == 1
    eigen, kappa = sp.symbols("eigen kappa", positive=True)
    N_formula = 2500 * d * kappa * (2 * kappa + 1) * rho ** 2 / eigen
    assert sp.diff(N_formula, kappa).is_positive
    assert sp.diff(N_formula, rho).is_positive
    assert sp.diff(N_formula, eigen).is_negative

    x, z = sp.symbols("x z", nonnegative=True)
    assert sp.expand((x + z) ** 2 - x ** 2 - z ** 2) == 2 * x * z
    delta = sp.symbols("delta", positive=True)
    h2 = 1 - sp.sqrt(1 - delta / sp.sqrt(2))
    asymptotic = sp.limit(sp.sqrt(h2) / sp.sqrt(delta), delta, 0, dir="+")
    assert sp.simplify(asymptotic - 2 ** (-sp.Rational(3, 4))) == 0
    return {
        "passed": True,
        "prior_normalization": str(normalization),
        "prior_mean_score": str(mean_score),
        "prior_fisher_one_coordinate": str(prior_fisher),
        "prior_information_matrices": matrices,
        "non_diagonal_matrix_trace_identity": True,
        "exact_prior_trace_term": str(sp.factor(exact_prior)),
        "fixed_parameter_lower_and_upper_limits": "1; 1",
        "vertex_asymptotic_constant": str(asymptotic),
    }


def tangent_basis(K):
    raw = np.eye(K)[:, :K - 1] - np.eye(K)[:, [-1]]
    return np.linalg.qr(raw)[0]


def information(P, w, Z):
    mixture = w @ P
    return Z.T @ ((P / mixture) @ P.T) @ Z


def parameters(P, w, Z):
    I = information(P, w, Z)
    eig = np.linalg.eigvalsh(I)
    return I, 1 / min(w), eig[0], eig[-1] / eig[0], np.trace(np.linalg.inv(I))


def risk_rhs(K, n, rho, lam, kappa, tau):
    d = K - 1
    theta = rho ** 2 / (n * lam)
    u = np.sqrt((2 * kappa + 1) * d * theta)
    N = 2500 * d * kappa * (2 * kappa + 1) * rho ** 2 / lam
    return (tau / n * (1 + 30 * np.sqrt(3 + theta) * u
                       + 34 * (3 + theta) * u ** 2)
            + 8 * 9 ** d * np.exp(-n / N))


def box_and_upper_checks():
    rng = np.random.default_rng(20261011)
    records = []
    for K in (2, 3, 5):
        d = K - 1
        Z = tangent_basis(K)
        P = rng.dirichlet(np.ones(2 * K + 2), size=K)
        w = np.arange(1, K + 1, dtype=float)
        w /= sum(w)
        I0, rho, lam, kappa, tau = parameters(P, w, Z)
        for eta in (0.15, 0.4, 0.5):
            s = eta / (rho * np.sqrt(d))
            worst_bound_ratio = 0.0
            worst_tau_ratio = 0.0
            min_psd_eigen = float("inf")
            for signs in itertools.product((-1, 1), repeat=d):
                y = s * np.asarray(signs)
                wy = w + Z @ y
                assert min(wy) > 0 and abs(sum(wy) - 1) < 1e-12
                density_ratio = (wy @ P) / (w @ P)
                assert min(density_ratio) >= 1 - eta - 1e-12
                assert max(density_ratio) <= 1 + eta + 1e-12
                Iy, rhoy, lamy, kappay, tauy = parameters(P, wy, Z)
                eig1 = np.linalg.eigvalsh(Iy - I0 / (1 + eta)).min()
                eig2 = np.linalg.eigvalsh(I0 / (1 - eta) - Iy).min()
                min_psd_eigen = min(min_psd_eigen, eig1, eig2)
                assert min(eig1, eig2) >= -1e-10
                assert rhoy <= 2 * rho * (1 + 1e-12)
                assert lamy >= 2 * lam / 3 * (1 - 1e-12)
                assert kappay <= 3 * kappa * (1 + 1e-12)
                assert tauy <= (1 + eta) * tau * (1 + 1e-12)
                worst_tau_ratio = max(worst_tau_ratio, tauy / ((1 + eta) * tau))
                for n in (1, 100, 10 ** 7):
                    individual = risk_rhs(K, n, rhoy, lamy, kappay, tauy)
                    uniform = risk_rhs(K, n, 2 * rho, 2 * lam / 3, 3 * kappa,
                                       (1 + eta) * tau)
                    assert individual <= uniform * (1 + 1e-12)
                    worst_bound_ratio = max(worst_bound_ratio, individual / uniform)
            records.append({"K": K, "eta": eta, "corners": 2 ** d,
                            "min_Loewner_difference_eigenvalue": float(min_psd_eigen),
                            "max_tau_to_uniform_cap": float(worst_tau_ratio),
                            "max_rhs_to_uniform_rhs": float(worst_bound_ratio)})
    return {"passed": True, "records": records,
            "scope": "finite discrete examples check algebra; no new arbitrary-family proof"}


def vertex_optimization_checks():
    rng = np.random.default_rng(107)
    records = []
    for delta in (0.001, 0.01, 0.05, 0.2, 0.5, 1.0):
        expected = 1 - np.sqrt(1 - delta / np.sqrt(2))
        def objective(v):
            t = float(sum(v))
            if t > 1:
                return 1 + 100 * (t - 1)
            return t / (1 + np.sqrt(max(0, 1 - t)))
        def distance_constraint(v):
            return sum(v) ** 2 + np.dot(v, v) - delta ** 2
        constraints = [{"type": "ineq", "fun": lambda v: 1 - sum(v)},
                       {"type": "ineq", "fun": distance_constraint}]
        starts = [np.array([0.45, 0.45]), np.array([0.7, 0.05]),
                  np.array([0.05, 0.7])]
        starts += [rng.dirichlet(np.ones(3))[1:] for _ in range(12)]
        valid = []
        for start in starts:
            result = minimize(objective, start, bounds=[(0, 1), (0, 1)],
                              constraints=constraints, method="SLSQP",
                              options={"ftol": 1e-14, "maxiter": 600})
            if (result.success and sum(result.x) <= 1 + 1e-10
                    and min(result.x) >= -1e-10
                    and distance_constraint(result.x) >= -1e-12):
                valid.append(result)
        assert valid, (delta, "no feasible successful optimization")
        best = min(valid, key=lambda result: result.fun)
        relative_error = abs(best.fun / expected - 1)
        assert relative_error < 2e-6, (delta, best.fun, expected, best.x)
        trial_w = np.array([1 - delta / np.sqrt(2), delta / np.sqrt(2), 0])
        direct_H2 = np.sum((np.sqrt(trial_w) - np.array([1, 0, 0])) ** 2) / 2
        assert abs(direct_H2 - expected) < 5e-15
        assert abs(np.linalg.norm(trial_w - np.array([1, 0, 0])) - delta) < 1e-14
        records.append({"delta": delta, "analytic_H2": float(expected),
                        "optimized_H2": float(best.fun),
                        "relative_error": float(relative_error),
                        "optimizer_w2_w3": [float(v) for v in best.x],
                        "successful_feasible_starts": len(valid)})
    return {"passed": True, "records": records,
            "method": "direct 2-variable SLSQP; simplex and Euclidean lower-distance constraints"}


def bernoulli_and_bridge_checks():
    a, b, t, y = sp.symbols("a b t y", real=True)
    n = sp.symbols("n", positive=True)
    q = b + (a - b) * (t + y / sp.sqrt(2))
    score_one = sp.diff(sp.log(q), y)
    score_zero = sp.diff(sp.log(1 - q), y)
    fisher = sp.simplify(q * score_one ** 2 + (1 - q) * score_zero ** 2)
    assert sp.simplify(fisher - (a - b) ** 2 / (2 * q * (1 - q))) == 0
    exact_unclipped_risk = 2 * q * (1 - q) / (n * (a - b) ** 2)
    assert sp.simplify(exact_unclipped_risk - 1 / (n * fisher)) == 0
    bridge_records = []
    families = [([1, 0, 0], [0, 1, 0]), ([0.7, 0.3], [0.2, 0.8]),
                ([0.5, 0.5, 0], [0, 0.5, 0.5]),
                ([0.3, 0.2, 0.5], [0.25, 0.35, 0.4])]
    for index, (left, right) in enumerate(families):
        p1, p2 = np.array(left), np.array(right)
        cTV = np.sum(abs(p1 - p2)) / np.sqrt(2)
        for t0, t1 in itertools.product((0.2, 0.5, 0.8), (0.03, 0.4, 0.97)):
            p = t0 * p1 + (1 - t0) * p2
            pp = t1 * p1 + (1 - t1) * p2
            mask = pp > 0
            kl_reverse = np.sum(pp[mask] * np.log(pp[mask] / p[mask]))
            mask_forward = p > 0
            kl_forward = np.sum(p[mask_forward] * np.log(p[mask_forward] / pp[mask_forward]))
            tv = np.sum(abs(pp - p)) / 2
            weight_loss = 2 * (t1 - t0) ** 2
            assert abs(tv - cTV * np.sqrt(weight_loss) / 2) < 1e-14
            assert tv ** 2 <= kl_reverse / 2 + 1e-14
            assert tv ** 2 <= kl_forward / 2 + 1e-14
            rhs_reverse = 2 * kl_reverse / cTV ** 2
            rhs_forward = 2 * kl_forward / cTV ** 2
            assert weight_loss <= rhs_reverse + 1e-13
            assert weight_loss <= rhs_forward + 1e-13
            bridge_records.append({"family": index, "cTV": float(cTV),
                                   "w1": t0, "w1_prime": t1,
                                   "weight_loss": float(weight_loss),
                                   "KL_wprime_over_w": float(kl_reverse),
                                   "KL_w_over_wprime": float(kl_forward),
                                   "two_KL_wprime_over_w_over_cTV_squared": float(rhs_reverse),
                                   "two_KL_w_over_wprime_over_cTV_squared": float(rhs_forward)})
    return {"passed": True, "symbolic_tangent_Fisher": str(fisher),
            "symbolic_unclipped_Euclidean_risk": str(exact_unclipped_risk),
            "symbolic_exact_risk_equals_trace_inverse_over_n": True,
            "bridge_records": bridge_records,
            "KL_directions": ["mixture at w relative to mixture at w_prime (manuscript Lemma 2.2)",
                              "mixture at w_prime relative to mixture at w"],
            "logarithm": "natural log"}


def point_mass_trace_checks():
    records = []
    for K in (2, 3, 4):
        d = K - 1
        Z = sp.zeros(K, d)
        for j in range(1, K):
            scale = sp.sqrt(j * (j + 1))
            for row in range(j):
                Z[row, j - 1] = 1 / scale
            Z[j, j - 1] = -j / scale
        assert sp.simplify(Z.T * Z - sp.eye(d)) == sp.zeros(d)
        coordinates = list(sp.symbols(f"w1:{K}", positive=True))
        w = sp.Matrix(coordinates + [1 - sum(coordinates)])
        information_matrix = Z.T * sp.diag(*[1 / value for value in w]) * Z
        covariance = sp.diag(*list(w)) - w * w.T
        candidate_inverse = Z.T * covariance * Z
        inverse_identity = (information_matrix * candidate_inverse - sp.eye(d)).applyfunc(sp.simplify)
        assert inverse_identity == sp.zeros(d)
        target = 1 - (w.T * w)[0]
        assert sp.simplify(sp.trace(candidate_inverse) - target) == 0
        numerical_w = np.arange(1, K + 1, dtype=float)
        numerical_w /= sum(numerical_w)
        numerical_Z = np.asarray(Z, dtype=float)
        numeric_trace = np.trace(np.linalg.inv(numerical_Z.T
                                @ np.diag(1 / numerical_w) @ numerical_Z))
        numeric_target = 1 - np.dot(numerical_w, numerical_w)
        assert abs(numeric_trace - numeric_target) < 1e-13
        records.append({"K": K,
                        "symbolic_I_times_candidate_inverse_equals_identity": True,
                        "symbolic_trace_equals_one_minus_sum_w_squared": True,
                        "candidate_inverse": "Z^T (diag(w)-w w^T) Z",
                        "numeric_trace": float(numeric_trace),
                        "numeric_one_minus_sum_w_squared": float(numeric_target)})
    return {"passed": True, "records": records,
            "condition": "interior simplex weights; fixed point-mass endmembers"}


def main():
    start = time.perf_counter()
    results = {
        "metadata": {"date": "2026-10-11", "author": "LN",
                     "python_executable": sys.executable,
                     "python_version": sys.version,
                     "libraries": {"numpy": [np.__version__, np.__file__],
                                   "sympy": [sp.__version__, sp.__file__],
                                   "scipy": [scipy.__version__, scipy.__file__]},
                     "scope": "deterministic symbolic/algebra/optimization checks; no Monte Carlo"},
        "symbolic_prior_and_matrices": symbolic_checks(),
        "box_and_uniform_MLE_parameters": box_and_upper_checks(),
        "vertex_Hellinger_optimization": vertex_optimization_checks(),
        "Bernoulli_example_and_KL_bridge": bernoulli_and_bridge_checks(),
        "point_mass_exact_trace": point_mass_trace_checks(),
    }
    results["all_passed"] = True
    results["elapsed_seconds"] = time.perf_counter() - start
    output = Path(__file__).with_name("revision_a_local_verify_out.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": True, "output": str(output),
                      "elapsed_seconds": results["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
