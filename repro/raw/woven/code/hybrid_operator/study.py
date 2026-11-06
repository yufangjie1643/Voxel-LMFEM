"""Operator tuning and full matched six-load FI comparisons.

python study.py --mode tune --grid 192
python study.py --mode full --variant split --block 128 --repeat 0
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get("FEAD_CHAIN",str(__import__('project_paths').source_path('solver_chain'))))
arch=torch.cuda.get_device_capability();os.environ.setdefault("TORCH_CUDA_ARCH_LIST",f"{arch[0]}.{arch[1]}")
os.environ["FEAD_TOL"]="1e-8"
import operator_adapter as a
import chain_study as cs


def timing(fn):
    for _ in range(8):fn()
    torch.cuda.synchronize();rows=[]
    for _ in range(5):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(20):out=fn()
        e.record();e.synchronize();rows.append(s.elapsed_time(e)/20)
        del out
    return rows


def sources():
    paths=[HERE/n for n in ("study.py","operator_adapter.py","background.py","operator_kernels.cu","fi_modal.cuh","fi_isotropic.cuh","generate_isotropic.py","generate_background.py","background_grouped.cuh")]
    paths += [p for p in a.HYBRID.iterdir() if p.suffix in (".py",".cu",".cuh")]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("tune","full"),required=True)
    p.add_argument("--variant",choices=("baseline","isotropic","split","split_delta","split_grouped"),default="baseline")
    p.add_argument("--block",type=int,default=128);p.add_argument("--grid",type=int,default=192)
    p.add_argument("--case",choices=("strong","near"),default="strong");p.add_argument("--repeat",type=int,default=0)
    p.add_argument("--tag",default="")
    args=p.parse_args();name=f"{args.mode}_{args.case}_{args.grid}" if args.mode=="tune" else f"full_{args.variant}_{args.block}_{args.case}_{args.grid}_r{args.repeat}"
    name+=args.tag
    path=HERE/"results"/(name+".json")
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(exist_ok=True)
    b=cs.s.load("fi");mf=a.prior.install(b,cs);cs.a.configure(b,"fi","combined")
    cs.a.native.extension();a.extension();b.TOL=1e-8;b.gpu_guard()
    if args.mode=="full":a.install(b,args.variant,args.block)
    data,geometry,material,input_hash=b.inputs(args.grid,args.case)
    def adapt():
        phase,table,orientation=mf.from_legacy_yarn(data,material)
        return dict(grid_shape=data["grid_shape"],domain_size=data["domain_size"],phase_id=phase,orientation=orientation),table
    if args.mode=="tune":
        adapted,table=adapt();b.cleanup();op=b.build(adapted,table,"direct","mixed",b.Stages())
        baseline=op.apply;original_stress=op.compute_avg_stress;original_reference=op._compute_reference_medium
        original_coeff=op._hybrid_coeff.clone()
        torch.manual_seed(193);u=torch.randn((1,op.mesh.n_dofs),device="cuda",dtype=torch.float64)
        expected=baseline(u);rows=[]
        options=[("baseline",128)]+[(v,k) for v in ("split_delta","split_grouped") for k in (64,128,256)]
        for repeat in range(3):
            order=options.copy();random.Random(107+repeat).shuffle(order)
            for variant,block in order:
                op._hybrid_coeff.copy_(original_coeff)
                op.apply=baseline;op.compute_avg_stress=original_stress;op._compute_reference_medium=original_reference
                a.attach(op,adapted,table,variant,block)
                actual=op.apply(u);err=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected));assert err<1e-13
                ms=timing(lambda:op.apply(u))
                row=dict(variant=variant,block=block,repeat=repeat,ms=ms,median_ms=statistics.median(ms),Ku_relative=err,metadata=op._operator_metadata)
                rows.append(row);print("TUNE",variant,block,repeat,row["median_ms"],err,flush=True)
        record=dict(rows=rows)
        del op,u,expected,actual,baseline,original_coeff,original_stress,original_reference;gc.collect();torch.cuda.empty_cache()
    else:
        def task():
            t=time.perf_counter();adapted,table=adapt();adaptation=time.perf_counter()-t
            result,_=b.workload(adapted,table,"direct","mixed")
            return result,time.perf_counter()-t,adaptation
        task();b.gpu_guard();print("MEASUREMENT_BEGIN",flush=True)
        result,total,adaptation=task();print("MEASUREMENT_END",flush=True)
        rr=max(v for call in result["calls"] for v in call["true_residuals"])
        assert len(result["calls"])==6 and np.isfinite(rr) and rr<=1e-8
        record=dict(measurement=result,workflow_with_adaptation_cleanup_seconds=total,adaptation_seconds=adaptation,
                    max_true_residual=rr,operator_metadata=b._operator_metadata,storage=b._storage,
                    reference_design=b._reference_design)
    record.update(complete=True,mode=args.mode,variant=args.variant,block=args.block,grid=args.grid,case=args.case,repeat=args.repeat,
        input_hash=input_hash,geometry=geometry,material=material,yarn_fraction=float(np.mean(data["yarn_id"]>=0)),
        gpu=torch.cuda.get_device_name(),environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,threads=torch.get_num_threads()),
        sources=sources(),provenance=cs.provenance(b),kernel_attributes=a.extension().attributes(),
        protocol="Exact operator/material representation optimization. Same physical K, RHS, recovery, reference, FFT and PCG; delta variants reuse the coefficient owner and apply exact periodic RHS/mean-stress identities. Full-grid same-size six-load warmup; all nodes and 8-point FI retained; FP64 physical fields, FP32 FFT, true residual <=1e-8. Full measured setup/solve/recovery plus outer adaptation/cleanup timer. Formal variants interleaved in fresh processes; NVML 20ms.")
    b.gpu_guard()
    with path.open("x") as f:json.dump(record,f,indent=2,allow_nan=False)
    print("PASS",name,flush=True)


if __name__=="__main__":main()
