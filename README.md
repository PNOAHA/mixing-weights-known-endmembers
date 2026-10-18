# mixing-weights-known-endmembers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22852168.svg)](https://doi.org/10.5281/zenodo.22852168) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)]()

Code, seeded numerical outputs and figures of

> Pan, L. (2026). *Finite-sample risk of maximum likelihood for mixing weights with known components.* Manuscript in preparation for submission to the *Journal of Multivariate Analysis*.

Release **v1.1.0** (2026-10-18) contains everything behind the numerical section (Section 5, Figures 1–3) of the
current manuscript. Release v1.0.0 (2026-09-20, https://doi.org/10.5281/zenodo.22852169) is the archive of an
earlier, longer manuscript (*Estimating the mixing weights of known endmembers: sharp maximum-likelihood risk, exact
separation moduli, and the limits of anchor-based pairing*); its files are kept unchanged below as "legacy".
Paths are given relative to `code/`.

## Current manuscript (v1.1.0)

| Manuscript | Files |
|---|---|
| Figure 1 (ratio n·E‖ŵ−w‖² / tr I(w)⁻¹, K=3, F1, θ=0.3) | `figs/d107/figG2_k3_mle_ratio.{pdf,png}` |
| Figure 2 (same ratio against θ_n = ρ_w²/(n λ_min(I(w))), F1, K=3,4, θ∈{0,0.3,0.6}) | `figs/d107/figG3_mle_ratio_vs_minw.{pdf,png}` |
| Figure 3 (anchor-count statistic vs MLE, F2, K=3, a∈{0.3,0.1,0.03}, n=1600) | `figs/d107/figG4_k3_anchor_vs_mle.{pdf,png}` |
| Every number quoted in Section 5 | `results/d107/_summary_fig123_dualgap.json` |
| Grid outputs read by Figures 1–3 | `results/d107/grid_{3_F1_0.0,3_F1_0.3,3_F1_0.6,3_F2_0.03,3_F2_0.1,3_F2_0.3,4_F1_0.0,4_F1_0.3,4_F1_0.6}{,_mle}.csv` |
| Component matrices of these nine grids | `results/d107/P_<K>_<family>_<parameter>.npy` |

```
code/
├── generator.py                 component families F1 (overlap θ), F2 (anchor mass a), Fisher/Gram quantities
├── run_grid_d107.py             one task = one (K, family, parameter); MLE = EM + tangent Newton, finished by
│                                mle_dualgap.polish_mle with a strict 1e-8 KKT dual-gap acceptance on the average
│                                log-likelihood (batches failing the gate are rejected); strict=False replays v1.0.0
├── mle_dualgap.py               likelihood polishing on the simplex (exact faces allowed) with the dual-gap gate
├── make_figs_d107.py            --fig123-only: Figures 1–3 and results/d107/_summary_fig123_dualgap.json
└── verify/
    ├── fig123_dualgap_acceptance.py  re-solves (same seeds, same preceding draws) every batch with an unaccepted
    │                                 replication and writes per-replication evidence to results/d107/strict_acceptance/
    ├── finalize_fig123_dualgap.py    validates the nine accepted CSVs against v1.0.0 (read from the git tag v1.0.0)
    ├── mle_dualgap_verify.py         independent checks: RNG alignment with the v1.0.0 archive, 720 polisher cases,
    │                                 gap snapshot of the shipped CSVs; output mle_dualgap_verify_out.json
    ├── recheck_s1_near_vertex.py     near-vertex point (K=4, θ=0.6, w=(0.9625,0.0125,0.0125,0.0125), n=6400):
    │                                 v1.0.0 solver vs dual-gap–accepted MLE on identical draws
    ├── revision_a_local_verify.py, round2_math_verify.py  symbolic / enumeration checks of constants and lemmas of
    │                                 the current manuscript (outputs *_out.json)
    └── pillar1_verify.py, pillar1_exam_verify.py, pillar4_verify.py  auxiliary checks from preparatory notes
```

Dual-gap acceptance: for every replication the reported estimate ŵ satisfies max_k s_k(ŵ) − ŵ·s(ŵ) ≤ 1e-8, where
s is the score of the average log-likelihood on the observed cells, i.e. the average log-likelihood is within 1e-8 of
its maximum over the simplex. All 4,164 rows (8,328,000 replications) of the nine `*_mle.csv` files are accepted
(`dual_gap_accept_frac = 1`, largest gap 9.99997e-9). Columns added in v1.1.0: `dual_gap_max`, `dual_gap_med`,
`dual_gap_accept_frac`, `n_refined`, `n_polished` and, where at least one batch had to be re-solved,
`feasibility_accept_frac` (filled for re-solved batches only; `grid_3_F2_0.3_mle.csv` needed no re-solve and so has
no such column).

The per-replication acceptance evidence (`results/d107/strict_acceptance/`: 9 JSON reports, 9 staged CSVs and
641 NPZ files, about 41 MB) is not included in the repository to keep it small; it is available from the author on
request and can be regenerated with `verify/fig123_dualgap_acceptance.py`.

### Reproducing (v1.1.0)

```bash
python -m pip install -r requirements.txt
cd code
python make_figs_d107.py --fig123-only      # Figures 1–3 + results/d107/_summary_fig123_dualgap.json (seconds)
python verify/mle_dualgap_verify.py          # independent checks (seconds; needs git for the v1.0.0 archive)
python run_grid_d107.py TASK_ID              # regenerate one grid; Figures 1–3 use TASK_ID 0 2 4 6 7 8 13 15 17
```

Everything is seeded (seed 20260919 + task id for the components, 1000·task id + lattice index for the samples).
Regenerated PNG files and `_summary_fig123_dualgap.json` are byte-identical to the shipped ones; PDF files differ only
in their `/CreationDate`. Tested with Python 3.13.13, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9 (Windows 11).

## Legacy files (v1.0.0, earlier manuscript)

The files below are unchanged from v1.0.0 (section, figure and theorem numbers refer to the earlier manuscript).
In v1.1.0 only the nine grids listed above, `run_grid_d107.py`, `make_figs_d107.py` and Figures G2–G4 were updated;
`results/d107/_summary_numbers.json`, Figures G1/G5/G6, the F3 grids and the `*_v02` outputs are as in v1.0.0.

### Layout (v1.0.0)

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

### Reproducing (v1.0.0)

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

Please cite the paper (see `CITATION.cff`). The archived repository is

> Pan, L. (2026). *mixing-weights-known-endmembers: code, numerical results and figures* (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.22852168 (concept DOI, all versions; v1.0.0: https://doi.org/10.5281/zenodo.22852169)

## License

MIT — see [LICENSE](LICENSE).

## Author

Lujia Pan, School of Mathematical Sciences, Hebei Normal University — peter205834@hebtu.edu.cn — ORCID [0009-0004-2103-0193](https://orcid.org/0009-0004-2103-0193)

Supported by the Doctoral Research Startup Fund of Hebei Normal University (L2024B05) and the Science Research Project of the Hebei Provincial Department of Education (QN2025106).
