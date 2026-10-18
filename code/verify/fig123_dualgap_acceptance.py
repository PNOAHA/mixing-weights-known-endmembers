"""Finish dual-gap acceptance of existing Fig1--3 grids; no modulus rerun.

Only batches containing unaccepted replications are reconstructed. The
original point seed and ALL preceding N_LIST draws are preserved. Results
are staged separately until all nine source tasks and strict gates pass.
Usage: python code/verify/fig123_dualgap_acceptance.py --tasks 0 2 ...
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
import run_grid_d107 as grid
from mle_dualgap import polish_mle

TASKS = [0, 2, 4, 6, 7, 8, 13, 15, 17]
# Repository-relative staging directory for per-replication acceptance evidence
# (JSON/CSV reports and NPZ files). Not shipped in git (~41 MB); available on request.
STAGE = ROOT / 'code/results/d107/strict_acceptance'
TOL = 1e-8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_point(args):
    tid, index, P, w, bad = args
    rng = np.random.default_rng(1000 * tid + index)
    K = len(w)
    results = []
    for n in grid.N_LIST:
        counts = rng.multinomial(n, w @ P, size=grid.REPS)
        if n not in bad:
            continue
        options = {'strict': False} if 'strict' in inspect.signature(grid.mle_vectorized).parameters else {}
        W, gaps, accepted, n_refined = grid.mle_vectorized(
            P, counts, np.full((grid.REPS, K), 1 / K), **options)
        x2_old = ((W - w) ** 2).sum(1)
        original = bad[n]
        if not np.isclose(x2_old.mean(), float(original['risk']), rtol=1e-10, atol=1e-13):
            raise RuntimeError(f'RNG/solver replay mismatch: task {tid}, point {index}, n {n}')
        before = gaps.copy()
        Wbefore = W.copy()
        reasons = np.full(grid.REPS, 'em_newton_slsqp_gap', dtype='<U40')
        repaired = 0
        for i in np.flatnonzero(~accepted | ~np.isfinite(gaps)):
            candidate, gap, ok, reason = polish_mle(P, counts[i], W[i], gap_tol=TOL)
            if not ok:
                raise RuntimeError(f'Strict gap gate failed: {tid}/{index}/{n}/{i}: {gap} ({reason})')
            W[i], gaps[i], reasons[i] = candidate, gap, reason
            repaired += 1
        feasible = np.isfinite(W).all(1) & (W.min(1) >= 0) & (np.abs(W.sum(1) - 1) <= 1e-12)
        accepted = feasible & np.isfinite(gaps) & (gaps <= TOL)
        if not accepted.all():
            raise RuntimeError(f'Feasibility/gap gate failed: {tid}/{index}/{n}')
        # Independently recompute the global dual gap, never trust a status flag.
        actual_gap = grid.dual_gap(P, counts, W)
        if not np.isfinite(actual_gap).all() or actual_gap.max() > TOL:
            raise RuntimeError(f'Independent dual-gap recomputation failed: {tid}/{index}/{n}')
        x2 = ((W - w) ** 2).sum(1)
        pw_old = np.maximum(Wbefore @ P, 1e-300)
        pw_new = np.maximum(W @ P, 1e-300)
        ll_change = (counts * (np.log(pw_new) - np.log(pw_old))).sum(1) / n
        if ll_change.min() < -1e-12:
            raise RuntimeError(f'Likelihood worsened: {tid}/{index}/{n}')
        dest = STAGE / f'reps_task{tid}_point{index}_n{n}.npz'
        np.savez_compressed(dest, initial_gap=before, final_gap=actual_gap,
                            weights=W, feasible=feasible, accepted=accepted,
                            reason=reasons, loss_change=x2-x2_old,
                            likelihood_change=ll_change)
        delta = x2 - x2_old
        results.append((n, dict(risk=float(x2.mean()), se=float(x2.std()/np.sqrt(grid.REPS)),
                      ratio=float(n*x2.mean()/float(original['tau'])),
                      boundary_frac=float((W.min(1)<1e-7).mean()),
                      dual_gap_max=float(actual_gap.max()), dual_gap_med=float(np.median(actual_gap)),
                      dual_gap_accept_frac=1.0, n_refined=n_refined,
                      n_polished=repaired, feasibility_accept_frac=1.0),
                      dict(point=index, n=n, n_polished=repaired,
                           paired_loss_change=float(delta.mean()),
                           paired_loss_change_se=float(delta.std(ddof=1)/np.sqrt(grid.REPS)),
                           min_average_ll_change=float(ll_change.min()),
                           max_weight_change=float(np.linalg.norm(W-Wbefore,axis=1).max()),
                           before_gap_max=float(before.max()), after_gap_max=float(actual_gap.max()),
                           reasons=dict(Counter(reasons)), evidence=str(dest.relative_to(ROOT)))))
    return index, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=int, nargs='+', default=TASKS)
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    for tid in args.tasks:
        K, family, parameter = grid.TASKS[tid]
        tag = f'{K}_{family}_{parameter}'
        source = CODE / 'results/d107' / f'grid_{tag}_mle.csv'
        sha = digest(source)
        report_path = STAGE / f'{tag}.json'
        if report_path.exists():
            existing=json.loads(report_path.read_text())
            staged_path=STAGE/source.name
            if sha in (existing['source_sha256'],existing['staged_sha256']) and staged_path.exists() and digest(staged_path)==existing['staged_sha256']:
                print(f'{tag}: already staged/promoted', flush=True)
                continue
        with source.open(newline='') as handle:
            reader = csv.DictReader(handle)
            rows, fields = list(reader), list(reader.fieldnames)
        if 'dual_gap_accept_frac' not in fields:
            raise RuntimeError(f'Task {tid} has not completed the dual-gap run')
        W = grid.simplex_lattice(K, 12 if K==3 else 8)
        if len(rows) != len(W)*len(grid.N_LIST):
            raise RuntimeError(f'Incomplete grid: {tag}')
        P = np.load(CODE / 'results/d107' / f'P_{tag}.npy')
        work = []
        for index, w in enumerate(W):
            point_rows = rows[index*4:(index+1)*4]
            if not all(np.allclose(json.loads(row['w']),w,atol=1e-6) for row in point_rows):
                raise RuntimeError('Unexpected lattice row ordering')
            bad = {int(row['n']):row for row in point_rows
                   if float(row['dual_gap_accept_frac']) < 1 or not np.isfinite(float(row['dual_gap_max']))
                   or float(row['dual_gap_max']) > TOL}
            if bad:
                work.append((tid,index,P,w,bad))
        print(f'{tag}: reconstructing {sum(len(x[-1]) for x in work)} failed batches / {len(rows)}', flush=True)
        with Pool(args.processes) as pool:
            updates = pool.map(repair_point, work)
        details = []
        for index, results in updates:
            for n, values, detail in results:
                row = rows[index*4+grid.N_LIST.index(n)]
                row.update(values)
                details.append(detail)
        for row in rows:
            row.setdefault('n_polished',0)
            # Unreconstructed batches retain the source solver's feasible simplex
            # iterates; their gate was already passed. Explicit sample feasibility
            # evidence is available for every reconstructed batch in its NPZ.
        fields.append('n_polished')
        if any('feasibility_accept_frac' in row for row in rows):
            fields.append('feasibility_accept_frac')
        if not all(float(row['dual_gap_accept_frac'])==1 and np.isfinite(float(row['dual_gap_max']))
                   and float(row['dual_gap_max'])<=TOL for row in rows):
            raise RuntimeError(f'Final aggregate gate failed: {tag}')
        destination = STAGE / source.name
        with destination.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        report=dict(task=tid,tag=tag,source_sha256=sha,staged_sha256=digest(destination),
                    rows=len(rows),replications=len(rows)*grid.REPS,
                    reconstructed_batches=len(details),n_polished=sum(x['n_polished'] for x in details),
                    gap_max=max(float(row['dual_gap_max']) for row in rows),accept_fraction=1.0,
                    replay_risk_rtol=1e-10, point_seed='1000*task_id+point_index',
                    preceding_draws=list(grid.N_LIST),details=details)
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'{tag}: PASS, {report["n_polished"]} polished, max gap={report["gap_max"]:.3g}',flush=True)


if __name__=='__main__':
    main()
