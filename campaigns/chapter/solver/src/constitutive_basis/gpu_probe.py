"""Resident HG Ku/energy probe; not a full homogenization timing claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import random
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get("FEAD_FORMULA",str(__import__("project_paths").source_path('reference_medium'))))
import formula_study as old
import numpy as np


def main(grid):
    b=old.previous.load_runtime("hg");b.gpu_guard();torch=b.torch
    old.formula.configure(b,"hg","baseline")
    from torch.utils.cpp_extension import load
    ext=load(name="fead_structural_20261020_v2",sources=[str(HERE/"structural_kernels.cu")],
             extra_cuda_cflags=["-O3","--ptxas-options=-v"],extra_cflags=["-O3"],verbose=False)
    data,geometry,material,digest=b.inputs(grid,"strong")
    op=b.build(data,material,"direct","mixed",b.Stages())
    from femlib.materials import engineering_to_voigt,isotropic_to_voigt,build_rotation_matrices
    from femlib.c3d8r import hourglass_scale
    y=material["yarn"]
    cy=engineering_to_voigt(*(y[k] for k in ("E11","E22","E33","Nu12","Nu13","Nu23","G12","G13","G23")))
    cm=isotropic_to_voigt(material["matrix"]["E"],material["matrix"]["Nu"])
    lam=cy[1,2];mu=(cy[1,1]-cy[1,2])/2;alpha=cy[0,1]-lam;gamma=cy[5,5]-mu
    beta=cy[0,0]-cy[1,1]-2*alpha-4*gamma;delta=cy[3,3]-mu
    active=np.flatnonzero(np.asarray(data["yarn_id"]).reshape(-1)>=0)
    r=build_rotation_matrices(data["orient1"].reshape(-1,3)[active],data["orient2"].reshape(-1,3)[active])
    transverse=r[:,:,1,None]*r[:,None,:,2]+r[:,:,2,None]*r[:,None,:,1]
    frame=np.concatenate((r[:,:,0],np.stack([transverse[:,i,j] for i,j in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))],axis=-1)),axis=-1)
    idx=np.full(grid**3,-1,dtype=np.int32);idx[active]=np.arange(len(active),dtype=np.int32)
    params=[lam,mu,alpha,beta,gamma,delta,cm[0,1],cm[3,3],hourglass_scale(cm,op._modal_voxel,.1),hourglass_scale(cy,op._modal_voxel,.1)]
    frame=torch.as_tensor(np.ascontiguousarray(frame),device="cuda",dtype=torch.float64)
    idx=torch.as_tensor(idx,device="cuda");params=torch.as_tensor(params,device="cuda",dtype=torch.float64)
    torch.manual_seed(20261020);v=torch.randn((1,3*grid**3),device="cuda",dtype=torch.float64)
    relative=lambda a,c:float(torch.linalg.vector_norm(a-c)/torch.linalg.vector_norm(c))
    def candidate(mode,energy):
        value,partial=ext.apply(v,op.C,op.hg_scale,idx,frame,params,*old.formula.geometry(op),mode,energy)
        return (value,partial.sum().reshape(1)) if energy else value
    def timing(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize();samples=[]
        for _ in range(1):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):out=fn()
            end.record();end.synchronize();samples.append(begin.elapsed_time(end)/10)
        return dict(milliseconds=samples,median_ms=statistics.median(samples))
    rows=[]
    for energy in (False,True):
        baseline=op.apply_with_energy(v) if energy else op.apply(v)
        names={-1:"modal_baseline",0:"C21_projector",1:"structural_projector",2:"structural_modal"}
        funcs={-1:lambda:op.apply_with_energy(v) if energy else op.apply(v)}
        errors={-1:dict(Ku_relative=0.,energy_relative=0.)}
        for mode in (0,1,2):
            funcs[mode]=lambda mode=mode:candidate(mode,energy)
            actual=funcs[mode]()
            error=relative(actual[0],baseline[0]) if energy else relative(actual,baseline)
            errors[mode]=dict(Ku_relative=error)
            if energy:errors[mode]["energy_relative"]=relative(actual[1],baseline[1])
            if max(errors[mode].values())>=1e-12:raise RuntimeError("GPU algebra mismatch: "+str(errors[mode]))
        # Warm clocks/caches, then interleave all configurations to reduce order bias.
        until=time.perf_counter()+1
        while time.perf_counter()<until:
            for fn in funcs.values():fn()
        torch.cuda.synchronize();samples={mode:[] for mode in funcs}
        for repeat in range(7):
            modes=list(funcs);random.Random(20261020+repeat).shuffle(modes)
            for mode in modes:samples[mode].extend(timing(funcs[mode])["milliseconds"])
        for mode,values in samples.items():
            rows.append(dict(variant=names[mode],energy=energy,milliseconds=values,median_ms=statistics.median(values),**errors[mode]))
    result=dict(grid=grid,kind="hg",scope="resident physical Ku and fused pKu; event batches seven by ten; excludes all setup, FFT, PCG, residual verification, RHS and stress recovery",
        geometry=geometry,material=material,input_hash=digest,active_elements=len(active),rows=rows,
        representation_bytes=dict(old_C21=op.C.numel()*op.C.element_size(),old_HG_scale=op.hg_scale.numel()*op.hg_scale.element_size(),
            new_index=idx.numel()*idx.element_size(),new_frame=frame.numel()*frame.element_size(),new_parameters=params.numel()*params.element_size()),
        provenance=old.provenance(b),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in (".py",".cu",".cuh")})
    out=HERE/"results"/f"probe_hg_{grid}_interleaved.json"
    if out.exists():raise FileExistsError(out)
    b.gpu_guard();old.save(out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("provenance","source_sha256")},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--grid",type=int,required=True)
    main(p.parse_args().grid)
