"""Recompute manuscript comparison values from the frozen raw records.

Run: python reproduce_tables.py
Requires Python 3 and NumPy. No GPU is needed to regenerate reported values.
"""
from pathlib import Path
from statistics import median
import csv
import hashlib
import itertools
import json
import numpy as np
from analytical_laminate import from_record

ROOT=Path(__file__).resolve().parent
E=ROOT/'manuscript_evidence'
OLD=E/'review_results/solver_study_20261022'
OUT=ROOT/'generated'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rel(a,b):return float(np.linalg.norm(np.array(a)-b)/np.linalg.norm(b))

def group(g,base):
    rows=[]
    for name in g['records']:
        p=(base/name).resolve();r=read(p);m=r['measurement']
        tele=read(p.parent.parent/'logs'/(p.stem+'_telemetry.json'))
        assert tele['returncode']==0 and tele['failure'] is None
        assert tele['measurement_sample_count']>0
        chfem='CH_project_convention' in m
        if chfem:
            assert m['passed'] and m['native_converged']
            t=c=m['core_seconds'];it=sum(m['iterations']);res=max(m['true_residuals'])
        else:
            assert r['complete']
            t=r['workflow_seconds'];c=m['seconds'];it=sum(v['iterations'] for v in m['calls']);res=r['max_true_residual']
        assert res<=1e-8
        rows.append(dict(time_s=t,compute_s=c,nvml_GiB=tele['peak_process_bytes_measured']/2**30,
            iterations=it,residual=res,CH=m.get('CH_project_convention',m.get('CH')),
            ku_ms=None if chfem else r['microbenchmark']['Ku']['median_ms']))
    assert len(rows)==3
    return dict(time_s=median(x['time_s'] for x in rows),compute_s=median(x['compute_s'] for x in rows),
        nvml_GiB=max(x['nvml_GiB'] for x in rows),iterations=sorted(set(x['iterations'] for x in rows)),
        residual=max(x['residual'] for x in rows),tensors=[x['CH'] for x in rows],
        ku_ms=None if rows[0]['ku_ms'] is None else median(x['ku_ms'] for x in rows),
        records=g['records'])

def csv_write(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    manifest=ROOT/'manifest.json'
    if manifest.exists():
        for item in read(manifest)['files']:
            assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    OUT.mkdir(exist_ok=True)
    cur=read(OLD/'summary_with_stencil27.json');old=read(OLD/'summary.json')
    groups={(g['case'],g['grid'],g['variant']):g for g in cur['groups']}
    hist={(g['case'],g['grid'],g['variant']):g for g in old['groups']}
    cache={}
    def get(case,n,v,historical=False):
        key=case,n,v,historical
        if key not in cache:cache[key]=group((hist if historical else groups)[case,n,v],OLD)
        return cache[key]
    comparison=[]
    for case in ['sphere','grosmont']:
        baseline=get(case,192,'chfem')
        for v in ['chfem','stencil27','paper','chain','latest']:
            g=get(case,192,v)
            discrepancy=0 if v=='chfem' else max(rel(a,b) for a,b in itertools.product(g['tensors'],baseline['tensors']))
            comparison.append(dict(case=case,variant=v,time_s=g['time_s'],speedup=baseline['time_s']/g['time_s'],
                nvml_GiB=g['nvml_GiB'],memory_ratio=g['nvml_GiB']/baseline['nvml_GiB'],iterations=g['iterations'][0],
                CH_relative=discrepancy))
    csv_write('table_solver_comparison.csv',comparison)
    ablation=[]
    specs=[('Modal action','paper','modal',False),('Hybrid storage','dense','hybrid',True),
           ('Background split','hybrid','latest',True),('Symmetric storage','stencil27','paper',False),
           ('Iteration data access','analytic','fusion',True)]
    for name,a,b,h in specs:
        x,y=get('woven',192,a,h),get('woven',192,b,h)
        assert x['iterations']==y['iterations']
        ablation.append(dict(module=name,before=a,after=b,compute_before=x['compute_s'],compute_after=y['compute_s'],
            nvml_before=x['nvml_GiB'],nvml_after=y['nvml_GiB'],iterations=x['iterations'][0],
            CH_relative=max(rel(p,q) for p,q in itertools.product(x['tensors'],y['tensors']))))
    csv_write('table_module_ablation.csv',ablation)
    activity=[]
    for case in ['fraction50','fraction90']:
        for v in ['hybrid','latest']:
            g=get(case,128,v);source=groups[case,128,v]
            analytical_error=max(rel(read(OLD/n)['measurement']['CH'],from_record(read(OLD/n))) for n in source['records'])
            activity.append(dict(active_fraction=source['active_fraction'],variant=v,compute_s=g['compute_s'],
                ku_ms=g['ku_ms'],nvml_GiB=g['nvml_GiB'],analytical_error=analytical_error))
    csv_write('table_active_fraction.csv',activity)
    precision=[];tensors={}
    base=E/'review_results/full_revision_20261024/fft_precision_192_20261024'
    for v in ['fp32','fp64']:
        names=[f'results/192_{v}_r{i}.json' for i in range(3)]
        g=group(dict(records=names),base);tensors[v]=g['tensors']
        restarts=[sum(c['restarts'] for c in read(base/n)['measurement']['calls']) for n in names]
        assert len(set(restarts))==1
        precision.append(dict(precision=v,time_s=g['time_s'],nvml_GiB=g['nvml_GiB'],
            iterations=g['iterations'][0],restarts=restarts[0],residual=g['residual']))
    csv_write('table_fft_precision.csv',precision)
    scaling=read(E/'review_results/grid_refinement_20261023/scaling_summary.json')
    curves=[];fine=None;sampled={}
    for g in scaling['groups']:
        result=group(g,E);sampled[g['grid'],g['variant']]=result
        curves.append(dict(grid=g['grid'],variant=g['variant'],time_s=result['time_s'],nvml_GiB=result['nvml_GiB']))
    csv_write('figure_grid_cost.csv',curves)
    fine=sampled[256,'latest']['tensors']
    csv_write('figure_grid_convergence.csv',[dict(grid=n,relative_error=max(rel(a,b) for a,b in itertools.product(sampled[n,'latest']['tensors'],fine))) for n in [64,96,128,160,192,224]])
    x,y=get('woven',192,'stencil27'),get('woven',192,'latest')
    headline=dict(woven_compute_reduction_percent=100*(1-y['compute_s']/x['compute_s']),
        woven_memory_reduction_percent=100*(1-y['nvml_GiB']/x['nvml_GiB']),
        FP32_complete_time_reduction_percent=100*(1-precision[0]['time_s']/precision[1]['time_s']),
        FP32_memory_reduction_percent=100*(1-precision[0]['nvml_GiB']/precision[1]['nvml_GiB']),
        FP32_FP64_CH_difference=max(rel(a,b) for a,b in itertools.product(tensors['fp32'],tensors['fp64'])))
    (OUT/'headline_values.json').write_text(json.dumps(headline,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(headline,indent=2))
    print('Regenerated four result tables and grid-figure data from raw measurements.')

if __name__=='__main__':main()
