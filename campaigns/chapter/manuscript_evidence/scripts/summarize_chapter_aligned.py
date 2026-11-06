"""Summarize measured chapter-aligned runs; no inferred or interpolated data."""
from pathlib import Path
import hashlib
import itertools
import json
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review_results/chapter_aligned_20261024'


def main():
    groups={}
    for folder in [OUT/'woven',OUT/'isotropic']:
        manifest=OUT/('woven_source_manifest.json' if folder.name=='woven' else 'source_manifest.json')
        hashes={x['path']:x['sha256'] for x in json.loads(manifest.read_text())}
        for path in sorted((folder/'results').glob('*.json')):
            d=json.loads(path.read_text())
            if not d.get('complete'):raise ValueError(path)
            for source,digest in d['sources'].items():
                name=source.split('/home/yufan/code/voxel-mechanics/',1)[-1]
                assert hashes.get(name)==digest,('Source snapshot mismatch',path.name,name)
            telemetry=folder/'logs'/f'{path.stem}_telemetry.json'
            t=json.loads(telemetry.read_text())
            assert t['returncode']==0 and t['measurement_sample_count']>0
            key=(d['case'],d['variant'])
            stages=d['measurement']['stages']
            prep=sum(s['seconds'] for s in stages if not s['name'].startswith(('rhs','solve','post')))
            solve=sum(s['seconds'] for s in stages if s['name'].startswith('solve'))
            recovery=sum(s['seconds'] for s in stages if s['name'].startswith(('rhs','post')))
            r=dict(repeat=d['repeat'],time=d['workflow_seconds'],compute=d['measurement']['seconds'],
                nvml=t['peak_process_bytes_measured']/2**30,
                iterations=sum(x['iterations'] for x in d['measurement']['calls']),
                restarts=sum(x['restarts'] for x in d['measurement']['calls']),
                residual=d['max_true_residual'],CH=d['measurement']['CH'],
                Ku=d['microbenchmark']['Ku']['median_ms'],preconditioner=d['microbenchmark']['preconditioner']['median_ms'],
                stage=[prep,solve,recovery,d['workflow_seconds']-prep-solve-recovery],
                reference=d.get('reference_design'),input_hash=d['input_hash'],
                input_metadata=d['input'].get('input_metadata'),
                source=str(path.relative_to(ROOT)),telemetry=str(telemetry.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),operator=d.get('operator'),storage=d.get('storage'))
            assert r['residual']<=1e-8 and min(r['stage'])>=0
            groups.setdefault(key,[]).append(r)
    required={(c,v) for c in ('sphere','grosmont') for v in ('csr_jacobi','stencil27_jacobi','stencil27_fft','execution_fp32')}
    required.update(('woven',v) for v in ('csr_jacobi','stencil27_jacobi','stencil27_fft','stencil14_fft',
        'direct_fft','modal_fft','hybrid_fft','background_fft','execution_fp64','execution_fp32'))
    assert required<=set(groups),('Missing measurements',sorted(required-set(groups)))
    normalized=json.loads((ROOT/'review_results/solver_study_20261022/chfem_normalized.json').read_text())
    for row in normalized['rows']:
        if row['variant']!='chfem':continue
        records=[]
        for entry in row['records']:
            p=ROOT/'review_results/solver_study_20261022'/entry.replace('\\','/')
            d=json.loads(p.read_text());records.append(d)
        groups[(row['case'],'chfem')]=[dict(repeat=i,time=d['measurement']['core_seconds'],nvml=row['nvml_GiB'],
            iterations=row['iterations'],CH=d['measurement']['CH_project_convention'],
            input_metadata=d['input_metadata'],
            source=str((ROOT/'review_results/solver_study_20261022'/row['records'][i].replace('\\','/')).relative_to(ROOT)))
            for i,d in enumerate(records)]
    result=[]
    for (case,variant),runs in groups.items():
        assert len(runs)==3 and len({x['repeat'] for x in runs})==3,(case,variant,len(runs))
        base=groups[(case,'chfem' if case!='woven' else 'stencil27_fft')]
        error=max(np.linalg.norm(np.array(a['CH'])-np.array(b['CH']))/np.linalg.norm(b['CH'])
                  for a,b in itertools.product(runs,base))
        row=dict(case=case,variant=variant,time=statistics.median(r['time'] for r in runs),
            time_min=min(r['time'] for r in runs),time_max=max(r['time'] for r in runs),
            memory=max(r['nvml'] for r in runs),iterations=sorted({r['iterations'] for r in runs}),
            error=float(error),runs=runs)
        if variant!='chfem':
            assert len({r['input_hash'] for r in runs})==1
            row.update(compute=statistics.median(r['compute'] for r in runs),
                Ku=statistics.median(r['Ku'] for r in runs),
                preconditioner=statistics.median(r['preconditioner'] for r in runs),
                residual=max(r['residual'] for r in runs),restarts=sorted({r['restarts'] for r in runs}))
        result.append(row)
    for case in ('woven','sphere','grosmont'):
        rows=[r for r in result if r['case']==case and r['variant']!='chfem']
        assert rows,case
        assert len({r['runs'][0]['input_hash'] for r in rows})==1
        if case!='woven':
            metadata=groups[(case,'chfem')][0]['input_metadata']
            for row in rows:
                for run in row['runs']:
                    for key in ('array_sha256','grid_shape_zyx','voxel_size_xyz','phase_materials'):
                        assert run['input_metadata'][key]==metadata[key],(case,key)
                    if row['variant']=='execution_fp32':
                        assert run['storage']['pool_rows']==2 and run['storage']['n_local_voxels']==0
        fft=[r for r in rows if 'jacobi' not in r['variant']]
        if fft:
            references=[np.array(run['reference']['reference']) for r in fft for run in r['runs']]
            assert max(np.max(np.abs(a-references[0])) for a in references)<1e-12
        for r in rows:
            assert r['error']<1e-6,(case,r['variant'],r['error'])
    legacy=ROOT/'review_results/solver_study_20261022'
    old_groups=json.loads((legacy/'summary_with_stencil27.json').read_text())['groups']
    activity=[]
    for case in ('fraction50','fraction90'):
        for variant in ('hybrid','latest'):
            group=next(g for g in old_groups if (g['case'],g['variant'])==(case,variant))
            records=[json.loads((legacy/name).read_text()) for name in group['records']]
            activity.append(dict(case=case,variant=variant,time=statistics.median(d['workflow_seconds'] for d in records),
                records=group['records'],timing='complete six-load call'))
    summary=dict(complete=True,groups=result,activity=activity,
        protocol='medians of three independent warmed process runs; maximum measured NVML peak; six load cases',
        legacy_chfem='same canonical 192^3 input and RTX5090; archived native-core runs retained')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for r in sorted(result,key=lambda x:(x['case'],x['variant'])):
        print(r['case'],r['variant'],f"{r['time']:.3f}s",f"{r['memory']:.3f}GiB",r['iterations'],f"error={r['error']:.3e}")


if __name__=='__main__':main()
