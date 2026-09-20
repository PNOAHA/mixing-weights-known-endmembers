"""Cross-check of the v0.2 exact 1-D moduli (generator.modulus_k3) against the v0.1 multi-start SLSQP
(generator.modulus) on the same (theta, w, delta) grid as Figure 1; reports the maximal relative difference
over the points where SLSQP returned a finite value."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import tangent_basis, point_quantities, modulus, modulus_k3
rng = np.random.default_rng(20260917)
K, m = 3, 6; Z = tangent_basis(K)
base = rng.dirichlet(np.ones(m) * 0.7, size=K)
ws = [np.array([1, 1, 1]) / 3, np.array([0.6, 0.3, 0.1]), np.array([0.8, 0.15, 0.05])]
worst = {"TV": 0.0, "H": 0.0, "chi": 0.0}; npts = 0; slsqp_worse = {"TV": 0, "H": 0, "chi": 0}
for th in (0.0, 0.3, 0.6):
    P = (1 - th) * base + th / m
    for w in ws:
        q = point_quantities(P, w, Z, rng=rng)
        for d in np.linspace(0.02, q.delta_max * 0.98, 14):
            for kind in ("TV", "H", "chi"):
                a = modulus_k3(P, w, Z, d, kind); b = modulus(P, w, Z, d, kind, rng=rng)
                if np.isfinite(a) and np.isfinite(b):
                    worst[kind] = max(worst[kind], abs(a - b) / max(a, 1e-300))
                    if b > a * (1 + 1e-7): slsqp_worse[kind] += 1     # SLSQP stuck above the exact minimum
            npts += 1
print("points:", npts, "max relative difference exact-vs-SLSQP:", {k: f"{v:.2e}" for k, v in worst.items()},
      "| SLSQP strictly above exact (>1e-7):", slsqp_worse)
