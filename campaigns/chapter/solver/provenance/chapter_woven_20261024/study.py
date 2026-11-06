"""Chapter-aligned, controlled solver comparisons through the project ./run."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

from project_paths import ROOT,source_path

VARIANTS=('csr_jacobi','stencil27_jacobi','stencil27_fft','stencil14_fft',
          'direct_fft','modal_fft','hybrid_fft','background_fft','execution_fp64','execution_fp32')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def configure(study,variant):
    import numpy as np
    import torch
    from types import MethodType
    b=study.cs.s.load('fi',configure=False)
    sys.path.insert(0,str(source_path('hybrid_operator')))
    import operator_adapter as oa
    mf,hm=oa.prior.material_modules()
    from reference_closed_form import design_closed
    sys.path.insert(0,str(source_path('spectral')))
    import precision_adapter
    precision_adapter.install(study.cs,'fp32' if variant=='execution_fp32' else 'fp64')

    def build(data,material,route,precision,stages):
        grid=tuple(data['grid_shape']);voxel=data['domain_size'][0]/np.array(grid[::-1])
        phase,table,orientation=mf.from_legacy_yarn(data,material)
        modern=dict(grid_shape=grid,domain_size=data['domain_size'],phase_id=phase,orientation=orientation)
        hybrid=variant in ('hybrid_fft','background_fft','execution_fp64','execution_fp32')
        if hybrid:
            C,index,storage=hm.build_hybrid_material(phase,table,orientation=orientation)
        else:
            C=b.construct(data,0,1,material,True,True)
            index=torch.empty(0,device=C.device,dtype=torch.int32)
            storage=dict(mode='elementwise',coefficient_bytes=C.numel()*C.element_size())
        b._storage=storage;stages.end('material')
        if variant=='csr_jacobi':
            import cupy as cp
            from csr_baseline import AssembledCSR
            torch.cuda.empty_cache()
            op=AssembledCSR(cp.from_dlpack(C),grid,voxel)
            b._operator_metadata=op.metadata
            stages.end('assembly')
            return op
        if variant.startswith('stencil'):
            import spatial_adapter
            if variant.startswith('stencil27') and not getattr(spatial_adapter,'_chapter_full27',False):
                adapter=load('chapter_stencil27',ROOT/'experiments/stencil27_baseline/adapter.py')
                adapter.install();spatial_adapter._chapter_full27=True
            op=spatial_adapter.make_solver(C,grid,voxel,spectrum='real_half').op
            op._modal_kind='fi';op._modal_voxel=tuple(voxel)
            stages.end('assembly',op)
        elif variant=='direct_fft':
            import atomic_adapter
            atomic_adapter.enable_reference=lambda *a,**k: None
            op=atomic_adapter.make(C,grid,'atomic128',voxel).op
            op._modal_kind='fi';op._modal_voxel=tuple(voxel)
            stages.end('operator',op)
        else:
            op=oa.prior.HybridFIOperator(C,index,grid,voxel,dense=not hybrid)
            stages.end('operator',op)
        if variant=='stencil27_jacobi':
            diagonal=op._diag_cache
            if diagonal is None:raise RuntimeError('Missing assembled Jacobi diagonal')
            inverse=diagonal.reciprocal()
            op._fft_enabled=False
            op.apply_precond=MethodType(lambda self,r:self.project_zero_mean(r*inverse),op)
            b._reference_design=dict(preconditioner='scalar Jacobi with periodic zero-mean projection')
        else:
            # Same phase-based reference, normalized to a common shear modulus.
            reference=op._compute_reference_medium('zeman',verbose=False)[0]
            phases=[mf.material_stiffness(table[int(k)]) for k in np.unique(phase)]
            reference,metadata=design_closed(phases,reference,'fi')
            reference=reference/reference[3,3]
            op._compute_reference_medium=MethodType(lambda self,*a,**k: reference[None].copy(),op)
            study.cs.s.f.formula.enable_fi_reference(op)
            b._reference_design=dict(reference=reference.tolist(),design=metadata)
        stages.end('reference',op)
        if variant in ('background_fft','execution_fp64','execution_fp32'):
            oa.attach(op,modern,table,'split_delta',64)
        b._operator_metadata=dict(variant=variant,details=getattr(op,'_operator_metadata',None))
        stages.end('representation',op)
        return op

    b.build=build
    if variant in ('execution_fp64','execution_fp32'):
        study.cs.a.configure(b,'fi','combined')
    if variant=='csr_jacobi':
        import cupy as cp
        b.solve=lambda op,rhs,**kw:op.solve(rhs,1e-8)
        original_cleanup=b.cleanup
        def cleanup():
            original_cleanup();cp.cuda.Device().synchronize();cp.get_default_memory_pool().free_all_blocks()
        b.cleanup=cleanup
    b.TOL=1e-8
    return b


def micro(op,csr=False):
    if not csr:
        import torch
        torch.manual_seed(193)
        value=torch.randn((1,op.mesh.n_dofs),device='cuda',dtype=torch.float64)
        event=lambda:torch.cuda.Event(enable_timing=True)
        elapsed=lambda a,b:a.elapsed_time(b)
    else:
        import cupy as cp
        value=cp.random.RandomState(193).standard_normal(op.mesh.n_dofs,dtype=cp.float64)
        event=lambda:cp.cuda.Event()
        elapsed=lambda a,b:cp.cuda.get_elapsed_time(a,b)
    result={}
    for key,fn in [('Ku',op.apply),('preconditioner',op.apply_precond)]:
        for _ in range(5):fn(value)
        samples=[]
        for _ in range(5):
            a,z=event(),event();a.record()
            for _ in range(20):fn(value)
            z.record();z.synchronize();samples.append(elapsed(a,z)/20)
        result[key]=dict(samples_ms=samples,median_ms=statistics.median(samples))
    return result


def worker(args):
    import numpy as np
    import torch
    comparison=load('chapter_comparison',ROOT/'experiments/solver_comparison/study.py')
    b=configure(comparison,args.variant)
    adapt,digest,meta=comparison.inputs(b,None,args.case,args.grid)
    def task():
        start=time.perf_counter();data,material=adapt()
        result,_=b.workload(data,material,'direct','mixed')
        return result,time.perf_counter()-start
    b.gpu_guard();task();b.gpu_guard()
    print('MEASUREMENT_BEGIN',flush=True)
    result,elapsed=task()
    print('MEASUREMENT_END',flush=True)
    residual=max(v for row in result['calls'] for v in row['true_residuals'])
    assert len(result['calls'])==6 and residual<=1e-8 and np.isfinite(result['CH']).all()
    data,mat=adapt();b.cleanup();op=b.build(data,mat,'direct','mixed',b.Stages())
    bench=micro(op,args.variant=='csr_jacobi')
    sources=comparison.sources()
    for p in Path(__file__).parent.glob('*.py'):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(complete=True,variant=args.variant,case=args.case,grid=args.grid,repeat=args.repeat,
        input_hash=digest,input=meta,measurement=result,workflow_seconds=elapsed,max_true_residual=residual,
        microbenchmark=bench,reference_design=getattr(b,'_reference_design',None),storage=getattr(b,'_storage',None),
        operator=getattr(b,'_operator_metadata',None),sources=sources,
        environment=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        protocol='FP64 physical operator and PCG; scalar Jacobi or fixed normalized phase-based FFT; three independent processes; full warmup; six unit engineering strains; FP64 true residual <=1e-8')
    if args.variant=='csr_jacobi':
        import cupy as cp
        report['environment']['cupy']=cp.__version__
    name=f'{args.case}_{args.grid}_{args.variant}_r{args.repeat}'
    save(args.out/'results'/f'{name}.json',report)
    print('PASS',name,elapsed,residual,flush=True)


def launch(args):
    runner=load('chapter_nvml',ROOT/'experiments/solver_comparison/run_jobs.py')
    runner.RUN_DIR=args.out
    nvml=runner.NVML();jobs=[]
    variants=args.variants.split(',')
    for r in range(args.repeats):
        for case in args.cases.split(','):
            for variant in variants if r%2==0 else variants[::-1]:
                name=f'{case}_{args.grid}_{variant}_r{r}'
                cmd=[str(ROOT/'run'),'chapter-study','--phase','worker','--case',case,
                     '--grid',str(args.grid),'--variant',variant,'--repeat',str(r),'--out',str(args.out)]
                jobs.append(dict(name=name,arguments=[str(Path(__file__))],command=cmd))
    save(args.out/'plan.json',jobs)
    complete=[]
    for job in jobs:
        complete.append(runner.execute(job,nvml,.02))
        print('DONE',job['name'],flush=True)
        (args.out/'status.json').write_text(json.dumps(dict(completed=complete,total=len(jobs)),indent=2))
    save(args.out/'complete.json',dict(complete=True,jobs=complete))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('validate','worker','launch'),required=True)
    p.add_argument('--case',choices=('woven','sphere','grosmont'),default='woven')
    p.add_argument('--cases',default='woven')
    p.add_argument('--variant',choices=VARIANTS,default='csr_jacobi')
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--grid',type=int,default=192)
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve()
    if not a.out.is_relative_to(ROOT/'runs'):p.error('Outputs must stay in the project runs directory')
    a.out.mkdir(parents=True,exist_ok=True);os.environ['VOXEL_RUN_DIR']=str(a.out)
    if a.phase=='validate':
        from csr_baseline import validate
        result=validate();save(a.out/'validation.json',result);print(json.dumps(result))
    elif a.phase=='worker':worker(a)
    else:launch(a)


if __name__=='__main__':main()
