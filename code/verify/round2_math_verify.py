"""Round 2 independent small symbolic/enumeration checks; no Monte Carlo."""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

DLL_HANDLES = []
if importlib.util.find_spec("sympy") is None:
    task_conda = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "miniconda3"
    sys.path.append(str(task_conda / "Lib" / "site-packages"))
import sympy as sp
import numpy as np


def constants_and_elementary_algebra():
    R = sp.Rational
    assert 1 / (8 * sp.sqrt(3)) < R(1, 13)
    assert R(13, 12) * R(96, 83) ** 2 < R(145, 100)
    assert R(13, 14) / 32 > R(1, 35)
    assert 6 + 16 * R(145, 100) <= 30
    assert 16 * R(145, 100) ** 2 <= 34
    assert 105 * sp.sqrt(3) > 181
    assert R(2 * 35 ** 2 * 182, 181) < 2500
    assert 32 / sp.E < 12
    u = R(1, 120)
    theta = u ** 2 / 3
    bracket = 1 + 30 * sp.sqrt(3 + theta) * u + 34 * (3 + theta) * u ** 2
    assert bracket < R(14401, 10000) < R(3, 2)
    D, A = sp.symbols("D A", nonnegative=True)
    expression = 1 + 3 * D + 16 * A + 16 * A ** 2 - (1 + D) ** 2 * (1 + 2 * A) ** 2
    certificate = D * (1 - D) + 4 * (4 - (1 + D) ** 2) * (A + A ** 2)
    assert sp.expand(expression - certificate) == 0
    n, sigma2, B, t = sp.symbols("n sigma2 B t", positive=True)
    s = t / (n * sigma2 + B * t / 3)
    exponent = -s * t + n * sigma2 * s ** 2 / (2 * (1 - s * B / 3))
    assert sp.simplify(exponent + t ** 2 / (2 * (n * sigma2 + B * t / 3))) == 0
    for k in range(2, 51):
        assert math.factorial(k) >= 2 * 3 ** (k - 2)
    values = [-0.999, -0.8, -0.2, -1e-6, 0, 1e-6, 0.2, 1, 10, 1000]
    for y in values:
        assert np.log1p(y) <= y - y ** 2 / (2 * (1 + max(y, 0))) + 1e-13
        assert abs(1 / (1 + y) - (1 - y + y ** 2 / (1 + y))) < 1e-10
    threshold_records = []
    for K, kappa, rho in itertools.product((2, 3, 4, 10), (1, 2, 20, 1000), (10, 100, 1000)):
        if rho < K:
            continue
        d = K - 1
        M = 2500 * d * kappa * (2 * kappa + 1) * rho ** 2
        n0_times_lambda = 2 * M * math.log(12 * 9 ** d * M)
        assert n0_times_lambda > 29 * M
        assert n0_times_lambda >= 14400 * d * (2 * kappa + 1) * rho ** 2
        assert n0_times_lambda >= 32 * rho ** 2 / 9
        tail_exponent = 1 / (2 * 35 ** 2 * d * (2 * kappa + 1) * rho ** 2
                            * (kappa + 1 / (105 * np.sqrt((2 * kappa + 1) * d))))
        inverse_N = 1 / M
        assert tail_exponent >= inverse_N
        threshold_records.append([K, kappa, rho])
    minimum_M = 2500 * 1 * 1 * 3 * 2 ** 2
    assert 2 * np.log(12 * 9 * minimum_M) > 29
    return {"passed": True,
            "step2_nonnegative_polynomial_certificate": str(certificate),
            "corollary_bracket_at_u_1_over_120": float(bracket),
            "tail_constant_2_times_35_squared_times_182_over_181": float(R(2 * 35 ** 2 * 182, 181)),
            "threshold_parameter_combinations": len(threshold_records),
            "scope": "checks intermediate constants and exact algebra, not Monte Carlo risk"}


def zero_variance_counterexample():
    Z = sp.Matrix([1, -1]) / sp.sqrt(2)
    possible_r = [sp.Matrix([2, 0]), sp.Matrix([0, 2])]
    squares = [sp.simplify((Z.T * r)[0] ** 2) for r in possible_r]
    assert squares == [2, 2]
    centered = [v - 2 for v in squares]
    assert centered == [0, 0]
    covariance = sp.diag(sp.Rational(1, 2), sp.Rational(1, 2)) - sp.Matrix([sp.Rational(1, 2)] * 2) * sp.Matrix([[sp.Rational(1, 2)] * 2])
    assert covariance.det() == 0
    return {"verified": True,
            "model": "K=2; two disjoint point masses; w=(1/2,1/2)",
            "squared_tangent_score_values": [str(v) for v in squares],
            "centered_concentration_variable_values": [str(v) for v in centered],
            "impact": "matrix-net Bernstein application can have zero variance; tail event then has probability zero"}


def fourth_moment_exact_enumeration():
    vectors = [sp.Matrix([-1, -1]), sp.Matrix([2, -1]), sp.Matrix([-1, 2])]
    prob = sp.Rational(1, 3)
    assert sum(vectors, sp.zeros(2, 1)) == sp.zeros(2, 1)
    Sigma = sum((prob * v * v.T for v in vectors), sp.zeros(2))
    m2 = sp.trace(Sigma)
    m4 = sum(prob * (v.dot(v)) ** 2 for v in vectors)
    records = []
    for n in (1, 2, 3, 4):
        exact = 0
        for indices in itertools.product(range(3), repeat=n):
            summed = sum((vectors[index] for index in indices), sp.zeros(2, 1))
            exact += prob ** n * summed.dot(summed) ** 2
        formula = n * m4 + n * (n - 1) * (m2 ** 2 + 2 * sp.trace(Sigma ** 2))
        assert exact == formula
        assert exact <= n * m4 + 3 * n ** 2 * m2 ** 2
        records.append({"n": n, "enumerated_value": str(exact), "formula_value": str(formula)})
    return {"passed": True, "records": records}


def independent_tangent_basis(K):
    Z = np.zeros((K, K - 1))
    for j in range(1, K):
        Z[:j, j - 1] = 1 / np.sqrt(j * (j + 1))
        Z[j, j - 1] = -j / np.sqrt(j * (j + 1))
    return Z


def finite_pairs_and_common_background():
    records = []
    c = 8 / 9
    for K in (2, 3, 4):
        Z = independent_tangent_basis(K)
        for a in (1.0, 0.4, 0.03):
            P = np.concatenate([a * np.eye(K), np.full((K, 1), 1 - a)], axis=1)
            w = np.full(K, 1 / K)
            pw = w @ P
            support = pw > 0
            G = (P[:, support] / pw[support]) @ P[:, support].T
            I = Z.T @ G @ Z
            assert np.max(abs(I - a * K * np.eye(K - 1))) < 1e-12
            lam, Q = np.linalg.eigh(I)
            unit = Z @ Q[:, 0]
            def radius(direction):
                return min(w[j] / (-direction[j]) for j in range(K) if direction[j] < -1e-12)
            if radius(-unit) > radius(unit):
                unit *= -1
            r2 = 1 / (K * (K - 1))
            sufficient_n = int(np.ceil(8 * (K - 1) / (9 * a)))
            for n in (sufficient_n, sufficient_n + 10):
                delta = np.sqrt(c / (n * a * K))
                wp = w + delta * unit
                assert min(wp) >= -1e-12
                assert delta <= radius(unit) + 1e-12
                assert delta ** 2 <= r2 * (1 + 1e-12)
                pp = wp @ P
                chi2 = np.sum((pp[support] - pw[support]) ** 2 / pw[support])
                h2 = np.sum((np.sqrt(np.maximum(pp, 0)) - np.sqrt(pw)) ** 2) / 2
                mask = pp > 0
                kl = np.sum(pp[mask] * np.log(pp[mask] / pw[mask]))
                assert abs(n * chi2 - c) < 1e-11
                assert n * h2 <= c / 2 + 1e-12
                assert n * kl <= c + 1e-12
                assert abs(1 / (27 * n * a * K) - 1 / (27 * K * n * a)) < 1e-15
                records.append({"K": K, "a": a, "n": n,
                                "minimum_pair_weight": float(min(wp)),
                                "n_chi_squared": float(n * chi2),
                                "n_Hellinger_squared": float(n * h2),
                                "n_KL": float(n * kl)})
    x = sp.symbols("x", positive=True)
    first = x * sp.exp(-x) / 16
    second = x * (1 - sp.sqrt(x / 2)) / 8
    assert sp.simplify(sp.diff(first, x).subs(x, 1)) == 0
    assert sp.simplify(first.subs(x, 1)) == 1 / (16 * sp.E)
    assert 1 / (16 * sp.E) < sp.Rational(1, 27)
    assert sp.simplify(sp.diff(second, x).subs(x, sp.Rational(8, 9))) == 0
    assert sp.simplify(second.subs(x, sp.Rational(8, 9))) == sp.Rational(1, 27)
    return {"passed": True, "records": records,
            "two_point_constant_at_c_8_over_9": "1/27"}


def exact_selected_count_comparison():
    t, a, n = sp.symbols("t a n", positive=True)
    exact_count = (1 / a - (t ** 2 + (1 - t) ** 2)) / n
    exact_mle = 2 * t * (1 - t) / n
    proposed_ratio = 1 + (1 / a - 1) / (2 * t * (1 - t))
    assert sp.simplify(exact_count / exact_mle - proposed_ratio) == 0
    assert sp.simplify(proposed_ratio.subs(a, 1)) == 1
    assert sp.simplify(proposed_ratio.subs(t, sp.Rational(2, 5))).limit(a, 0, dir="+") == sp.oo
    records = []
    for t0 in (sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(4, 5)):
        for a0 in (sp.Rational(1), sp.Rational(1, 3), sp.Rational(1, 100)):
            probabilities = [t0 * a0, (1 - t0) * a0, t0 * (1 - a0), (1 - t0) * (1 - a0)]
            for n0 in (1, 2, 3):
                risk_count = sp.Rational(0)
                risk_mle = sp.Rational(0)
                for sample in itertools.product(range(4), repeat=n0):
                    probability = sp.prod(probabilities[index] for index in sample)
                    N1, N2 = sample.count(0), sample.count(1)
                    M1 = sample.count(0) + sample.count(2)
                    hat1, hat2 = N1 / (n0 * a0), N2 / (n0 * a0)
                    empirical = sp.Rational(M1, n0)
                    risk_count += probability * ((hat1 - t0) ** 2 + (hat2 - (1 - t0)) ** 2)
                    risk_mle += probability * 2 * (empirical - t0) ** 2
                count_expected = exact_count.subs({t: t0, a: a0, n: n0})
                mle_expected = exact_mle.subs({t: t0, n: n0})
                assert risk_count == count_expected
                assert risk_mle == mle_expected
                ratio = sp.simplify(risk_count / risk_mle)
                assert ratio >= 1
                if a0 == 1:
                    assert ratio == 1
                records.append({"t": str(t0), "a": str(a0), "n": n0,
                                "enumerated_count_risk": str(risk_count),
                                "enumerated_MLE_risk": str(risk_mle), "ratio": str(ratio)})
    return {"passed": True, "symbolic_ratio": str(proposed_ratio), "records": records,
            "realization": "disjoint uniform supports, each split into a selected region of mass a and its complement"}


def main():
    start = time.perf_counter()
    output = {
        "metadata": {"date": "2026-10-11", "author": "LN",
                     "python_executable": sys.executable,
                     "sympy_version": sp.__version__, "numpy_version": np.__version__,
                     "scope": "round 2 deterministic symbolic/enumeration checks; no Monte Carlo or HPC"},
        "constants_and_elementary_algebra": constants_and_elementary_algebra(),
        "zero_variance_matrix_net_counterexample": zero_variance_counterexample(),
        "fourth_moment_exact_enumeration": fourth_moment_exact_enumeration(),
        "finite_pairs_and_common_background": finite_pairs_and_common_background(),
        "exact_selected_count_vs_full_support_MLE": exact_selected_count_comparison(),
        "all_passed": True,
    }
    output["elapsed_seconds"] = time.perf_counter() - start
    path = Path(__file__).with_name("round2_math_verify_out.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": True, "elapsed_seconds": output["elapsed_seconds"],
                      "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
