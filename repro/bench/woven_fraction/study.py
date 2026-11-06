"""Matched woven-fraction experiment, invoked through ./run woven-fraction."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

from project_paths import ROOT, source_path
from geometry import geometry, MATERIALS, FRACTIONS

VARIANTS = ('elementwise', 'hybrid', 'background')


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def configure(comparison, variant):
    import numpy as np
    import torch
    import operator_adapter as oa
    sys.path.insert(0,str(source_path('spectral')))
    import precision_adapter
    precision_adapter.install(comparison.cs,'fp32')
    b=comparison.cs.s.load('fi',configure=False)
    mf, hm=oa.prior.material_modules()

    def build(data, table, route, precision, stages):
        phase=data['phase_id']; orient=data['orientation']; grid=data['grid_shape']
        voxel=data['domain_size'][0]/np.array(grid[::-1])
        dense=variant=='elementwise'
        if dense:
            C=mf.build_material_field(phase,table,orientation=orient)
            index=torch.empty(0,device='cuda',dtype=torch.int32)
            b._storage=dict(mode='elementwise',n_elements=int(phase.size),
                coefficient_bytes=C.numel()*C.element_size(),index_bytes=0)
        else:
            C,index,b._storage=hm.build_hybrid_material(phase,table,orientation=orient)
            b._storage['mode']='hybrid'
        stages.end('material')
        op=oa.prior.HybridFIOperator(C,index,grid,voxel,dense=dense)
        stages.end('operator',op)
        # Fixed matrix-phase reference for every fraction and every variant.
        reference=mf.material_stiffness(table[0]); reference=reference/reference[3,3]
        op._compute_reference_medium=MethodType(lambda self,*a,**kw:reference[None].copy(),op)
        comparison.cs.s.f.formula.enable_fi_reference(op)
        b._reference_design=dict(reference=reference.tolist(),source='fixed normalized isotropic matrix phase')
        stages.end('reference',op)
        if variant=='background': oa.attach(op,data,table,'split_delta',64)
        b._operator_metadata=getattr(op,'_operator_metadata',{})
        stages.end('representation',op)
        return op

    b.build=build
    comparison.cs.a.configure(b,'fi','combined')
    b.TOL=1e-8
    return b


def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    sys.path.insert(0,str(ROOT/'repro/bench/solver_comparison'))
    comparison=load('woven_comparison',ROOT/'repro/bench/solver_comparison/study.py')
    b=configure(comparison,args.variant)
    data,ids,meta,digest=geometry(args.grid,args.fraction)
    del ids
    def task():
        start=time.perf_counter()
        result,_=b.workload(data,MATERIALS,'direct','mixed')
        return result,time.perf_counter()-start
    b.gpu_guard(); task(); b.gpu_guard()
    print('MEASUREMENT_BEGIN',flush=True)
    result,elapsed=task()
    print('MEASUREMENT_END',flush=True)
    residual=max(v for c in result['calls'] for v in c['true_residuals'])
    assert len(result['calls'])==6 and residual<=1e-8 and np.isfinite(result['CH']).all()
    sources=comparison.sources()
    for p in Path(__file__).parent.glob('*.py'):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(complete=True,case='woven_fraction',grid=args.grid,variant=args.variant,
        repeat=args.repeat,input_hash=digest,input=meta,materials=MATERIALS,
        measurement=result,workflow_seconds=elapsed,max_true_residual=residual,
        storage=b._storage,operator=b._operator_metadata,reference_design=b._reference_design,
        sources=sources,environment=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,python=sys.version,threads=torch.get_num_threads()),
        protocol='FP64 physical operator and PCG; common FP32 FFT and fused execution; fixed normalized matrix reference; '
        'six engineering unit strains; periodic zero-mean displacement; true FP64 residual <=1e-8; '
        'one complete same-grid warmup; fresh process; synchronized complete-call time includes material construction, '
        'operator and FFT setup, RHS, PCG, residual certification, stress and cleanup; '
        'geometry, JIT compilation, warmup and I/O excluded.')
    name=f'woven_f{round(args.fraction*10000):04d}_{args.grid}_{args.variant}_r{args.repeat}'
    save(args.out/'results'/f'{name}.json',report)
    print('PASS',name,elapsed,residual,flush=True)


def cpu_validate(args):
    import numpy as np
    checks=[]
    for f in FRACTIONS:
        data,ids,meta,digest=geometry(args.grid,f,full_orientation=False)
        assert min(meta['yarn_voxel_counts'])>0
        if args.grid==192: assert abs(meta['voxel_yarn_fraction']-f)<.002
        checks.append(dict(input_hash=digest,**meta))
    save(args.out/'geometry_validation.json',dict(complete=True,grid=args.grid,cases=checks))
    print('PASS geometry',len(checks),args.grid,flush=True)


def launch(args):
    runner=load('woven_nvml',ROOT/'repro/bench/solver_comparison/run_jobs.py')
    runner.RUN_DIR=args.out; nvml=runner.NVML()
    fractions=FRACTIONS if args.fractions=='all' else [float(x) for x in args.fractions.split(',')]
    jobs=[]
    for repeat in range(args.repeats):
        fs=fractions if repeat%2==0 else fractions[::-1]
        for i,f in enumerate(fs):
            order=list(VARIANTS); shift=(repeat+i)%3;order=order[shift:]+order[:shift]
            for variant in order:
                name=f'woven_f{round(f*10000):04d}_{args.grid}_{variant}_r{repeat}'
                jobs.append(dict(name=name,arguments=[str(Path(__file__))],command=[str(ROOT/'repro/run'),
                    'woven-fraction','--phase','worker','--grid',str(args.grid),'--fraction',str(f),
                    '--variant',variant,'--repeat',str(repeat),'--out',str(args.out)]))
    if (args.out/'plan.json').exists():
        assert json.loads((args.out/'plan.json').read_text())==jobs
    else: save(args.out/'plan.json',jobs)
    status=dict(complete=False,total=len(jobs),completed=[],failed=None)
    if (args.out/'status.json').exists():status=json.loads((args.out/'status.json').read_text())
    done={j['name'] for j in status['completed']}
    for job in jobs:
        if job['name'] in done:continue
        try:
            status['completed'].append(runner.execute(job,nvml,.02))
            save(args.out/'status.json',status)
            print('DONE',len(status['completed']),'/',len(jobs),job['name'],flush=True)
        except BaseException as error:
            status['failed']=dict(name=job['name'],error=repr(error));save(args.out/'status.json',status);raise
    status['complete']=True;save(args.out/'status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('geometry','worker','launch'),required=True)
    p.add_argument('--grid',type=int,default=192)
    p.add_argument('--fraction',type=float,default=.19)
    p.add_argument('--fractions',default='all')
    p.add_argument('--variant',choices=VARIANTS,default='background')
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve()
    if not a.out.is_relative_to(ROOT/'runs'):p.error('Output must be in runs/')
    a.out.mkdir(parents=True,exist_ok=True);os.environ['VOXEL_RUN_DIR']=str(a.out)
    if a.phase=='worker':worker(a)
    elif a.phase=='geometry':cpu_validate(a)
    else:launch(a)


if __name__=='__main__':main()
