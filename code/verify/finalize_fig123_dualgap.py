"""Validate, compare, and optionally promote all nine accepted Fig1--3 CSVs.

No grid is run here. Original archives and first-pass dual-gap files remain
available. Run after fig123_dualgap_acceptance.py has staged all nine tasks.
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'code/results/d107'
# Repository-relative locations. STAGE holds the output of fig123_dualgap_acceptance.py
# (not shipped in git, ~41 MB; available on request). The v1.0.0 CSVs are read from
# LEGACY if present, otherwise from the git tag v1.0.0.
RUN=RESULTS/'strict_acceptance'
STAGE=RUN
LEGACY=RESULTS/'legacy_v1.0.0'
TAGS=['3_F1_0.0','3_F1_0.3','3_F1_0.6','3_F2_0.3','3_F2_0.1',
      '3_F2_0.03','4_F1_0.0','4_F1_0.3','4_F1_0.6']


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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    if isinstance(path,str):  # legacy v1.0.0 file name
        rows=list(csv.DictReader(io.StringIO(legacy_csv_text(path))))
        return {(r['w'],int(r['n'])):r for r in rows}
    with path.open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    return {(r['w'],int(r['n'])):r for r in rows}


def numbers(data):
    answer={'Figure1':{},'Figure2':{},'Figure3':{}}
    for n in (400,6400):
        rows=[r for (_,m),r in data['3_F1_0.3'].items() if m==n]
        inner=[float(r['ratio']) for r in rows if min(json.loads(r['w']))>=.1]
        answer['Figure1'][str(n)]={'inner_range':[min(inner),max(inner)],
              'inner_median':float(np.median(inner)),
              'all_min':min(float(r['ratio']) for r in rows),
              'face_fraction_max':max(float(r['boundary_frac']) for r in rows)}
    for K in (3,4):
        rows=[r for tag in (f'{K}_F1_{t}' for t in (0.0,.3,.6)) for r in data[tag].values()]
        for lo,hi in ((0,.3),(10,100)):
            values=[float(r['ratio']) for r in rows
                    if lo<=float(r['rho_w'])**2/(int(r['n'])*float(r['lam_min']))<hi]
            answer['Figure2'][f'K{K}_[{lo},{hi})']={'points':len(values),'range':[min(values),max(values)]}
    for a in (.3,.1,.03):
        values=[]
        for (_,n),r in data[f'3_F2_{a}'].items():
            if n==1600:
                w=np.array(json.loads(r['w']))
                values.append(float(np.sum(w*(1-w*a)/a)/n/float(r['risk'])))
        answer['Figure3'][str(a)]={'range':[min(values),max(values)],'median':float(np.median(values))}
    return answer


def printed_numbers(raw):
    """Rounding precision actually used in Section 5 (exclude unquoted bins)."""
    answer={'Figure1':{},'Figure2':{},'Figure3':{}}
    for n,values in raw['Figure1'].items():
        answer['Figure1'][n]={'inner_range':[f'{x:.2f}' for x in values['inner_range']],
          'inner_median':f'{values["inner_median"]:.3f}',
          'all_min':f'{values["all_min"]:.2f}',
          'face_percent':f'{100*values["face_fraction_max"]:.0f}'}
    for key,values in raw['Figure2'].items():
        if key=='K4_[10,100)':
            continue
        answer['Figure2'][key]={'points':values['points'],'range':[f'{x:.2f}' for x in values['range']]}
    for a,values in raw['Figure3'].items():
        answer['Figure3'][a]={'range':[f'{x:.1f}' for x in values['range']],
                            'median':f'{values["median"]:.1f}'}
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--promote',action='store_true')
    args=parser.parse_args()
    log=RUN/'run_master.log'  # optional in the repository copy
    if log.exists() and 'ALL DONE' not in log.read_text(encoding='utf-8'):
        raise RuntimeError('Existing full-grid runner has not completed')
    per_tag={}; before={}; after={}; reports=[]
    for tag in TAGS:
        name=f'grid_{tag}_mle.csv'
        old=before[tag]=load(name)
        new=after[tag]=load(STAGE/name)
        report=json.loads((STAGE/f'{tag}.json').read_text(encoding='utf-8'))
        current=sha(RESULTS/name)
        if current not in (report['source_sha256'],report['staged_sha256']):
            raise RuntimeError(f'Source changed outside this run: {tag}')
        if sha(STAGE/name)!=report['staged_sha256'] or old.keys()!=new.keys():
            raise RuntimeError(f'Incomplete/unmatched staged data: {tag}')
        expected=364 if tag.startswith('3_') else 660
        if len(new)!=expected:
            raise RuntimeError(f'Wrong grid size: {tag}')
        if not all(float(r['dual_gap_accept_frac'])==1 and np.isfinite(float(r['dual_gap_max']))
                   and float(r['dual_gap_max'])<=1e-8 for r in new.values()):
            raise RuntimeError(f'Acceptance failed: {tag}')
        changes=[]; standardized=[]; face_changes=[]; benchmark_change=0; event_changes=[]
        for key in old:
            o,r=old[key],new[key]
            changes.append(100*(float(r['risk'])/float(o['risk'])-1))
            standardized.append(abs(float(r['risk'])-float(o['risk']))/float(o['se']))
            face_changes.append(100*(float(r['boundary_frac'])-float(o['boundary_frac'])))
            event_changes.append(abs(float(r['P_E1'])-float(o['P_E1'])))
            for column in ('tau','lam_min','lam_max','rho_w'):
                benchmark_change=max(benchmark_change,abs(float(r[column])-float(o[column])))
                if not np.isclose(float(r[column]),float(o[column]),rtol=1e-10,atol=1e-12):
                    raise RuntimeError(f'Dictionary/Fisher/RNG benchmark changed: {tag}/{key}/{column}')
        per_tag[tag]={'rows':len(new),'gap_max':report['gap_max'],
             'n_polished':report['n_polished'],
             'reconstructed_batches':report['reconstructed_batches'],
             'risk_change_pct_range':[min(changes),max(changes)],
             'risk_abs_change_pct_max':max(map(abs,changes)),
             'rows_abs_risk_change_above_1pct':sum(abs(x)>1 for x in changes),
             'max_change_in_old_marginal_mc_se':max(standardized),
             'face_change_percentage_points_range':[min(face_changes),max(face_changes)],
             'P_E1_changed_rows':sum(x>0 for x in event_changes),
             'P_E1_max_absolute_change':max(event_changes),
             'max_absolute_benchmark_change':benchmark_change}
        reports.append(report)
    vertex_key=(json.dumps([.9625,.0125,.0125,.0125]),6400)
    o=before['4_F1_0.6'][vertex_key]; r=after['4_F1_0.6'][vertex_key]
    output={'tasks':9,'lattice_truths':1041,'mle_rows':4164,'replications':8328000,
            'accept_fraction':1.0,'gap_tolerance':1e-8,
            'gap_max':max(x['gap_max'] for x in reports),
            'n_polished':sum(x['n_polished'] for x in reports),
            'reconstructed_batches':sum(x['reconstructed_batches'] for x in reports),
            'scope':'All replication gaps checked in source/staged aggregate; per-rep feasibility replayed only for reconstructed batches.',
            'comparison':per_tag,'quoted_before':numbers(before),'quoted_after':numbers(after),
            'near_vertex':{'task':17,'point':164,'n':6400,'old_risk':float(o['risk']),
                           'new_risk':float(r['risk']),'risk_change_pct':100*(float(r['risk'])/float(o['risk'])-1)}}
    output['printed_before']=printed_numbers(output['quoted_before'])
    output['printed_after']=printed_numbers(output['quoted_after'])
    output['printed_numbers_changed']=output['printed_before']!=output['printed_after']
    output['F2_cross_archive_pairing_note']='F2 has seven P_E1 rows with probability-ULP-sensitive multinomial replay drift. Cross-archive risk differences include sampling drift; see P_E1_drift_audit_20261014.json. Within-first-pass strict-polish comparisons use identical counts.'
    if args.promote:
        backup=RUN/'backup_first_dualgap'
        backup.mkdir(exist_ok=True)
        for tag in TAGS:
            name=f'grid_{tag}_mle.csv'
            if not (backup/name).exists():
                shutil.copy2(RESULTS/name,backup/name)
            shutil.copy2(STAGE/name,RESULTS/name)
        output['promoted']=True
    (RUN/'FINAL_comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:output[k] for k in ('tasks','mle_rows','replications','gap_max','n_polished','near_vertex','quoted_after')},indent=2))


if __name__=='__main__':
    main()
