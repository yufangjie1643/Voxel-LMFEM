"""Validate and aggregate the fresh-process woven-fraction records."""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import statistics
import numpy as np

VARIANTS=('elementwise','hybrid','background')


def analyze(root, allow_partial=False):
    raw=root/'raw'
    records=[]
    for p in sorted((raw/'results').glob('*.json')):
        r=json.loads(p.read_text()); r['_source']=str(p.relative_to(root))
        t=raw/'logs'/(p.stem+'_telemetry.json')
        if not t.exists():
            if allow_partial:continue
            raise FileNotFoundError(t)
        telemetry=json.loads(t.read_text())
        assert r['complete'] and telemetry['returncode']==0 and telemetry['failure'] is None
        assert telemetry['measurement_sample_count']>0
        assert [m['kind'] for m in telemetry['markers']]==['begin','end']
        assert r['max_true_residual']<=1e-8
        r['_peak_bytes']=telemetry['peak_process_bytes_measured']
        r['_iterations']=[c['iterations'] for c in r['measurement']['calls']]
        records.append(r)
    groups={}
    reference=records[0]['reference_design']
    for r in records:
        assert r['reference_design']==reference
        key=r['input']['exact_yarn_fraction']
        groups.setdefault(key,[]).append(r)
    rows=[]; checks=[]
    for f,rs in sorted(groups.items()):
        assert len({r['input_hash'] for r in rs})==1
        assert len({json.dumps(r['materials'],sort_keys=True) for r in rs})==1
        assert len({json.dumps(r['input'],sort_keys=True) for r in rs})==1
        errors=[]
        for a,b in itertools.product(rs,rs):
            ca,cb=np.array(a['measurement']['CH']),np.array(b['measurement']['CH'])
            errors.append(float(np.linalg.norm(ca-cb)/np.linalg.norm(ca)))
        check=dict(nominal_fraction=f,actual_fraction=rs[0]['input']['voxel_yarn_fraction'],
                   max_stiffness_difference=max(errors),
                   max_true_residual=max(r['max_true_residual'] for r in rs),
                   identical_iterations=len({tuple(r['_iterations']) for r in rs})==1,
                   iteration_sequences=sorted({tuple(r['_iterations']) for r in rs}),
                   input_hash=rs[0]['input_hash'])
        assert check['max_stiffness_difference']<1e-8
        checks.append(check)
        for v in VARIANTS:
            vv=[r for r in rs if r['variant']==v]
            if len(vv)!=3:
                if allow_partial:continue
                raise ValueError((f,v,len(vv)))
            assert {r['repeat'] for r in vv}=={0,1,2}
            times=[r['workflow_seconds'] for r in vv]
            memory=[r['_peak_bytes']/2**30 for r in vv]
            ss=vv[0]['storage']; material_bytes=ss['coefficient_bytes']+ss['index_bytes']
            if v!='elementwise':
                assert ss['coefficient_bytes']==(ss['n_local_voxels']+2)*168
                assert ss['index_bytes']==192**3*4
            row=dict(nominal_fraction=f,actual_fraction=check['actual_fraction'],variant=v,
                time_median=statistics.median(times),time_min=min(times),time_max=max(times),
                peak_gib=max(memory),memory_min_gib=min(memory),memory_max_gib=max(memory),
                time_samples=times,peak_gib_samples=memory,
                iterations=sorted({sum(r['_iterations']) for r in vv}),
                material_bytes=material_bytes,
                max_stiffness_difference=check['max_stiffness_difference'],
                max_true_residual=max(r['max_true_residual'] for r in vv),
                stage_medians={name:statistics.median(next(s['seconds'] for s in r['measurement']['stages'] if s['name']==name) for r in vv)
                    for name in [s['name'] for s in vv[0]['measurement']['stages']]},
                sources=[r['_source'] for r in vv])
            rows.append(row)
    complete=len(records)==189 and len(rows)==63 and len(checks)==21
    if not allow_partial:assert complete
    gains=[]
    for f in sorted(groups):
        rr={r['variant']:r for r in rows if r['nominal_fraction']==f}
        if len(rr)!=3:continue
        dense,hybrid,background=[rr[v] for v in VARIANTS]
        gains.append(dict(nominal_fraction=f,actual_fraction=dense['actual_fraction'],
            hybrid_memory_reduction=1-hybrid['peak_gib']/dense['peak_gib'],
            background_memory_reduction=1-background['peak_gib']/dense['peak_gib'],
            hybrid_time_reduction=1-hybrid['time_median']/dense['time_median'],
            background_time_reduction=1-background['time_median']/dense['time_median'],
            background_vs_hybrid_time_reduction=1-background['time_median']/hybrid['time_median'],
            material_reduction=1-hybrid['material_bytes']/dense['material_bytes']))
    summary=dict(complete=complete,record_count=len(records),rows=rows,checks=checks,gains=gains,
        max_stiffness_difference=max(c['max_stiffness_difference'] for c in checks),
        max_true_residual=max(c['max_true_residual'] for c in checks),
        identical_iterations=all(c['identical_iterations'] for c in checks),
        common_reference=reference,environment=records[0]['environment'],
        protocol=records[0]['protocol'],source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raw.rglob('*.json'))})
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['nominal_fraction','actual_fraction','variant','time_median','time_min','time_max',
            'peak_gib','memory_min_gib','memory_max_gib','iterations','material_bytes',
            'max_stiffness_difference','max_true_residual']
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in fields})
    print(json.dumps({k:summary[k] for k in ['complete','record_count','max_stiffness_difference','max_true_residual','identical_iterations']},indent=2))
    if gains:
        print(json.dumps({k:[min(g[k] for g in gains),max(g[k] for g in gains)] for k in gains[0] if 'reduction' in k},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args();analyze(a.root,a.allow_partial)
