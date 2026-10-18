"""LN independent dual-gap / RNG checks; never launches or overwrites a grid.

2026-10-14. Run from paperA or any cwd. Input CSVs are read-only snapshots.
Writes only the independent JSON report code/verify/mle_dualgap_verify_out.json.
This is a same-available-model independent check, not different-model review.
"""
import csv
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'code'
REVIEW = CODE / 'verify'  # recheck_s1_near_vertex.py lives here
LEGACY = CODE / 'results/d107/legacy_v1.0.0'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(REVIEW))
import run_grid_d107 as rg
from generator import family_F1, family_F2, tangent_basis
from mle_dualgap import polish_mle
from recheck_s1_near_vertex import mle_original


def legacy_csv_text(name):
    """v1.0.0 archive of results/d107/<name>.

    Read from code/results/d107/legacy_v1.0.0/<name> if present, otherwise from
    the git tag v1.0.0 of this repository (git show v1.0.0:code/results/d107/<name>).
    """
    path = LEGACY / name
    if path.exists():
        return path.read_text(encoding='utf-8')
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'v1.0.0:code/results/d107/{name}'],
                          capture_output=True, text=True, check=True).stdout


def check_rng_alignment():
    tid, K, index = 17, 4, 164
    P = family_F1(K, 2 * K, 0.6, np.random.default_rng(20260919 + tid))
    w = rg.simplex_lattice(K, 8)[index]
    pw = w @ P
    # v1.0.0 (pre-dual-gap) archive of this grid
    archived = list(csv.DictReader(io.StringIO(legacy_csv_text('grid_4_F1_0.6_mle.csv'))))[index * 4 + 3]
    assert json.loads(archived['w']) == [0.9625, 0.0125, 0.0125, 0.0125]
    assert int(archived['n']) == 6400
    result = {'archive_risk': float(archived['risk']), 'batches': {}}
    for burn in (False, True):
        rng = np.random.default_rng(1000 * tid + index)
        if burn:
            for n in rg.N_LIST[:-1]:
                rng.multinomial(n, pw, size=2000)
        counts = rng.multinomial(6400, pw, size=2000)
        old = mle_original(P, counts, np.full((2000, K), 1 / K))
        risk = float(np.mean(np.sum((old - w) ** 2, axis=1)))
        result['batches']['burn_previous_N_LIST' if burn else 'no_burn'] = {
            'risk': risk,
            'risk_minus_archive': risk - result['archive_risk'],
            'counts_sha256': hashlib.sha256(counts.tobytes()).hexdigest(),
        }
    assert abs(result['batches']['burn_previous_N_LIST']['risk_minus_archive']) < 1e-13
    assert abs(result['batches']['no_burn']['risk_minus_archive']) > 1e-5
    return result


def check_polisher():
    rng = np.random.default_rng(997)
    cases = []
    for K in (2, 3, 4):
        cases.append(('F1', family_F1(K, 2 * K, 0.0, rng)))
        cases.append(('F2', family_F2(K, 2 * K, 0.3, np.full(K, 0.1), rng)))
    cases.extend([
        ('identical', np.full((3, 4), 0.25)),
        ('identity', np.eye(4)),
        ('one-observed', np.array([[0.7, 0.3], [0.6, 0.4], [0.2, 0.8]])),
    ])
    accepted, worst_gap, issues = 0, 0.0, []
    for name, A in cases:
        K = A.shape[0]
        for n in (1, 3, 100, 6400):
            for rep in range(20):
                truth = rng.dirichlet(np.full(K, 0.3))
                counts = rng.multinomial(n, truth @ A)
                initial = np.eye(K)[rep % K] if rep % 2 else rng.dirichlet(np.ones(K))
                W, gap, ok, reason = polish_mle(A, counts, initial)
                observed = counts > 0
                prob = W @ A[:, observed]
                if np.any(prob <= 0):
                    independent_gap = np.inf
                else:
                    score = (A[:, observed] / prob) @ (counts[observed] / n)
                    independent_gap = float(score.max() - W @ score)
                feasible = (np.isfinite(W).all() and W.min() >= 0
                            and abs(W.sum() - 1) <= 1e-12)
                passed = (feasible and np.isfinite(independent_gap)
                          and independent_gap <= 1e-8)
                if ok != passed or not passed:
                    issues.append([name, n, rep, float(independent_gap), ok, reason])
                else:
                    accepted += 1
                    worst_gap = max(worst_gap, independent_gap)
    assert not issues, issues[:20]
    return {'cases': 720, 'accepted': accepted, 'worst_gap': worst_gap, 'issues': issues}


def snapshot_csvs():
    result = {}
    for tid in (0, 2, 4, 6, 7, 8, 13, 15, 17):
        K, family, par = rg.TASKS[tid]
        tag = f'{K}_{family}_{par}'
        path = CODE / 'results/d107' / f'grid_{tag}_mle.csv'
        rows = list(csv.DictReader(path.open()))
        result[tag] = {'rows': len(rows), 'has_gap_columns': 'dual_gap_max' in rows[0]}
        if result[tag]['has_gap_columns']:
            result[tag].update({
                'max_gap': max(float(r['dual_gap_max']) for r in rows),
                'min_accept_frac': min(float(r['dual_gap_accept_frac']) for r in rows),
                'failed_reps': round(sum((1 - float(r['dual_gap_accept_frac'])) * 2000
                                         for r in rows)),
            })
    return result


if __name__ == '__main__':
    output = {'generated_at': datetime.now().astimezone().isoformat(),
              'rng': check_rng_alignment(), 'polisher': check_polisher(),
              'csv_snapshot': snapshot_csvs()}
    print(json.dumps(output, indent=2))
    report_path = REVIEW / 'mle_dualgap_verify_out.json'
    report_path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
