"""Matched 192³ six-load woven comparison: pure C21, matched dense, hybrid.

python compare.py --variant pure|dense|hybrid --repeat 0 [--mode full|micro]
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get("FEAD_CHAIN",str(__import__('project_paths').source_path('solver_chain'))))
os.environ["FEAD_TOL"]="1e-8"
arch=torch.cuda.get_device_capability();os.environ.setdefault("TORCH_CUDA_ARCH_LIST",f"{arch[0]}.{arch[1]}")
import chain_study as cs


def timing(fn,number=30):
    for _ in range(5):fn()
    torch.cuda.synchronize();rows=[]
    for _ in range(7):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for j in range(number):value=fn()
        end.record();end.synchronize();rows.append(start.elapsed_time(end)/number)
        del value
    return dict(milliseconds=rows,median_ms=statistics.median(rows),calls_per_repeat=number)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",choices=("pure","dense","hybrid"),required=True)
    p.add_argument("--repeat",type=int,default=0);p.add_argument("--grid",type=int,default=192)
    p.add_argument("--mode",choices=("full","micro"),default="full")
    args=p.parse_args();path=HERE/"results"/f"{args.mode}_{args.variant}_{args.grid}_r{args.repeat}.json"
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(exist_ok=True)
    b=cs.s.load("fi")
    if args.variant=="pure":
        from adapter import install
        mf=install(b,cs)
    else:
        from hybrid_adapter import install,extension
        extension();mf=install(b,cs,dense=args.variant=="dense")
    cs.a.configure(b,"fi","combined");cs.a.native.extension();b.TOL=1e-8
    data,geometry,material,input_hash=b.inputs(args.grid,"strong")

    def adapt():
        phase,table,orientation=mf.from_legacy_yarn(data,material)
        return dict(grid_shape=data["grid_shape"],domain_size=data["domain_size"],
                    phase_id=phase,orientation=orientation),table

    def task():
        t=time.perf_counter();adapted,table=adapt();adaptation=time.perf_counter()-t
        result,_=b.workload(adapted,table,"direct","mixed")
        return result,time.perf_counter()-t,adaptation

    b.gpu_guard()
    if args.mode=="full":
        task();b.gpu_guard();print("MEASUREMENT_BEGIN",flush=True)
        result,elapsed,adaptation=task();print("MEASUREMENT_END",flush=True)
        rr=max(v for call in result["calls"] for v in call["true_residuals"])
        assert len(result["calls"])==6 and np.isfinite(rr) and rr<=1e-8
        record=dict(measurement=result,workflow_with_adaptation_cleanup_seconds=elapsed,
                    adaptation_seconds=adaptation,max_true_residual=rr)
    else:
        adapted,table=adapt();b.cleanup();op=b.build(adapted,table,"direct","mixed",b.Stages())
        torch.manual_seed(197);u=torch.randn((1,op.mesh.n_dofs),device="cuda",dtype=torch.float64)
        macro=np.eye(6)[0]
        result=dict(apply=timing(lambda:op.apply(u)),rhs=timing(lambda:op.compute_rhs(macro)))
        # Recovery wrapper includes a CPU transfer; use wall time as well as CUDA events.
        t=time.perf_counter()
        for _ in range(20):op.compute_avg_stress(u,macro)
        result["stress_wrapper_ms"]=(time.perf_counter()-t)*1000/20
        record=dict(micro=result,reference_C0=op._formula_reference_C0)
        del op,u;gc.collect();torch.cuda.empty_cache()
    record.update(complete=True,variant=args.variant,grid=args.grid,repeat=args.repeat,mode=args.mode,
        case="strong_woven",input_hash=input_hash,geometry=geometry,material=material,
        yarn_fraction=float(np.mean(data["yarn_id"]>=0)),storage=getattr(b,"_storage",None),
        reference_design=getattr(b,"_reference_design",None),gpu=torch.cuda.get_device_name(),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,threads=torch.get_num_threads(),python=sys.version),
        provenance=cs.provenance(b),sources={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in HERE.iterdir() if q.suffix in (".py",".cu",".cuh")},
        protocol="Same 192³ STRONG woven inputs, FP64 physical fields, all nodes and full eight-point integration, frozen native FP32 FFT and combined PCG. Full same-grid six-load warmup; timed setup, six solves, true residual checks and average stress recovery. Full call additionally includes adaptation and cleanup. Three interleaved independent processes per variant. Dense/hybrid use identical arithmetic kernels with compile-time direct/indexed material access; pure is frozen generic C21 implementation.")
    b.gpu_guard()
    with path.open("x") as f:json.dump(record,f,indent=2,allow_nan=False)
    print("PASS",args.mode,args.variant,flush=True)


if __name__=="__main__":main()
