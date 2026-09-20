# mixing-weights-known-endmembers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)]() [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)]()

Code, numerical results and figures of

> Pan, L. (2026). *Estimating the mixing weights of known endmembers: sharp maximum-likelihood risk, exact separation moduli, and the limits of anchor-based pairing.* Submitted to the *Electronic Journal of Statistics*.

Every number and figure of the paper's numerical sections (Section 6) and every constant of its proofs that was
checked by machine (Appendix C, Theorems 3.1, 3.2, 4.2, 5.1, 5.3, Propositions 3.5, A.1–A.3, Theorem B.1) is
produced by the scripts below. Paths in the paper are given relative to `code/`, exactly as in this repository.

## Layout

```
code/
├── generator.py              endmember families F1 (discrete, overlap θ), F2 (anchor, mass a), F3 (Gaussian, bandwidth h);
│                             exact one-dimensional moduli for K=3 (modulus_k3, c_tv_k3, delta0_tv_k3), Fisher/Gram quantities
├── run_local_k3_v02.py       Section 6.1: Figures 1–4 and results/*_v02.csv (K=3, laptop, ~90 s, seed 20260917)
├── run_grid_d107.py          Section 6.2: one Slurm array task = one family; writes results/d107/grid_*.csv
├── grid_d107.sbatch          the 26-task Slurm array used on the D107 cluster (seed 20260919 + task id)
├── make_figs_d107.py         Section 6.2: re-runs every check on the grid files, writes Figures 5–10 and
│                             results/d107/_summary_numbers.json (every number quoted in Section 6.2)
├── verify/                   machine checks of the proofs (each prints PASS/FAIL per item)
│   ├── a1_verify.py          Theorem 3.1 (χ-modulus, feasibility radius)
│   ├── a2_verify.py          Theorem 3.2 (TV and Hellinger moduli, boundary first-order behaviour)
│   ├── a3_verify.py          Theorems 5.1, 5.3, 5.5 and Proposition 5.4 (anchor family, two-point bound, pairing)
│   ├── a4_verify.py          Theorem 4.2 / Corollaries 4.3–4.4 (all numerical constants, pathwise identities, Monte Carlo);
│   │                         a4_verify_out.txt is the run quoted in Remark (2) of Section 4
│   ├── c2_verify.py          Proposition 3.5 (Gaussian endmembers, Λ_h) and the K=4 value 1.001 of Remark (2) after it
│   ├── pillar2_verify.py     Propositions A.1–A.3 (compressed kernel matrix, parity law for K ≤ 9)
│   ├── pillar3_verify.py     Theorem B.1 (K=2 boundary dichotomy)
│   └── modulus_k3_vs_slsqp.py  exact K=3 moduli against the multi-start SLSQP search (Section 6.1)
├── results/                  outputs of run_local_k3_v02.py (*_v02.csv) and of the cluster run (d107/, 46 grid files + summary)
└── figs/                     Figures 1–4 (fig_*.pdf/.png) and 5–10 (d107/figG*.pdf/.png)
```

## Reproducing

```bash
python -m pip install -r requirements.txt
cd code
python run_local_k3_v02.py            # Figures 1–4, results/*_v02.csv, prints all Section 6.1 checks
python make_figs_d107.py              # Figures 5–10 and results/d107/_summary_numbers.json from the shipped grid files
python verify/a4_verify.py            # etc.; a4 takes ~10 min (Monte Carlo up to n = 2·10^7)
```

Regenerating the grid files themselves needs a Slurm cluster: `sbatch grid_d107.sbatch` (26 array tasks; the F3 tasks
with the maximum likelihood estimator take about 4.5 h each on 4001 cells). Everything is seeded; the shipped CSV files
are the ones used in the paper.

Tested with Python 3.13.13, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9, sympy 1.14.0, mpmath 1.3.0 (Windows 11 and Rocky Linux 9).

## Citation

Please cite the paper (see `CITATION.cff`); the archived version of this repository has a Zenodo DOI (badge above once minted).

## License

MIT — see [LICENSE](LICENSE).

## Author

Lujia Pan, School of Mathematical Sciences, Hebei Normal University — peter205834@hebtu.edu.cn — ORCID [0009-0004-2103-0193](https://orcid.org/0009-0004-2103-0193)

Supported by the Doctoral Research Startup Fund of Hebei Normal University (L2024B05) and the Science Research Project of the Hebei Provincial Department of Education (QN2025106).
