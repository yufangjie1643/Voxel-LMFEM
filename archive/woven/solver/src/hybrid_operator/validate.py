"""Independent CPU/GPU checks for exact FI background split and phase specialization.

python validate.py --phase cpu|gpu --output results/validate_cpu.json
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get("FEAD_CHAIN",str(__import__('project_paths').source_path('solver_chain'))))
sys.path.insert(0,str(__import__('project_paths').source_path('runtime/repro/source')))
import operator_adapter as a
from background import uniform_stencil,apply_cpu


def independent():
    path=a.HYBRID/"independent_validation.py"
    if not path.exists():path=HERE.parent/"material_interface_20261021/validate.py"
    spec=importlib.util.spec_from_file_location("independent_fe",path)
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v


def cpu():
    v=independent();mf,_=a.prior.material_modules();rng=np.random.default_rng(27)
    rows=[]
    for shape in ((2,3,4),(5,4,3)):
        h=(.3,.5,.7);n=int(np.prod(shape));u=rng.normal(size=3*n)
        q=rng.normal(size=(6,6))
        for C in (mf.material_stiffness(dict(kind="isotropic",E=3500.,nu=.35)),q@q.T+2*np.eye(6)):
            K,*_=v.assemble(np.repeat(C[None],n,axis=0),shape,h)
            S=uniform_stencil(C,h);actual=apply_cpu(u,shape,S)
            error=v.relative(actual,K@u);assert error<1e-13
            assert np.max(np.abs(apply_cpu(np.tile([1.,2.,3.],n),shape,S)))==0
            rows.append(dict(shape=shape,relative=error))
    return dict(passed=True,device="cpu",rows=rows)


def fixture(v,kind):
    data,table,full,h=v.fixture("mixed" if kind in ("mixed","all_local") else "isotropic",grid=5)
    mf,_=a.prior.material_modules()
    if kind=="uniform":
        data["phase_id"][:]=10
    elif kind=="general_background":
        q=np.random.default_rng(13).normal(size=(6,6));C=1000*(q@q.T+np.eye(6))
        # Force an anisotropic shared background, while retaining another phase.
        data["phase_id"].reshape(-1)[:90]=10
        table[10]=dict(kind="anisotropic",frame="global",C=C)
    elif kind=="all_local":
        data["phase_id"][:]=70
    full=np.stack([mf.material_stiffness(table[int(k)]) for k in data["phase_id"].ravel()])
    if kind in ("mixed","all_local"):
        R=data["orientation"].reshape(-1,3,3)
        for e in np.flatnonzero(data["phase_id"].ravel()==70):full[e]=v.fourth_order_rotate(full[e],R[e])
    return data,table,full,h


def gpu():
    import chain_study as cs
    b=cs.s.load("fi",configure=False);b.TOL=1e-10
    a.extension();cs.a.native.extension();a.prior.extension()
    b.solve_pcg_batched=cs.a.make_solver("fi",True,True,b.solve_pcg_batched)
    v=independent();mf,hm=a.prior.material_modules();rows=[]
    for kind in ("isotropic","mixed","uniform","general_background","all_local"):
        data,table,full,h=fixture(v,kind);grid=data["grid_shape"];n=int(np.prod(grid))
        K,rhs_cpu,u_cpu,CH_cpu,recover=v.assemble(full,grid,h)
        pool,index,_=hm.build_hybrid_material(data["phase_id"],table,orientation=data.get("orientation"))
        torch.manual_seed(18);u=torch.randn((1,3*n),device="cuda",dtype=torch.float64)
        for variant in ("baseline","isotropic","split","split_delta","split_grouped"):
            op=a.prior.HybridFIOperator(pool.clone() if variant in ("split_delta","split_grouped") else pool,index,grid,h);a.attach(op,data,table,variant)
            error=v.relative(op.apply(u).cpu().numpy()[0],K@u.cpu().numpy()[0]);assert error<1e-13
            uniform=torch.tensor(np.tile([1.,2.,3.],n)[None],device="cuda",dtype=torch.float64)
            rigid=float(torch.max(torch.abs(op.apply(uniform))));assert rigid<1e-10
            # Random symmetry and positive-energy checks supplement direct K comparison.
            w=torch.randn_like(u);kw=op.apply(w);ku=op.apply(u)
            symmetry=float(torch.abs((u*kw).sum()-(w*ku).sum())/(torch.linalg.vector_norm(u)*torch.linalg.vector_norm(kw)))
            assert symmetry<1e-13 and float((u*ku).sum())>0
            cs.s.f.formula.enable_fi_reference(op);cs.a.native.attach(op,"fi","dense");cs.a.attach_raw(op)
            calls=[];columns=[]
            if kind!="uniform":
                for j,macro in enumerate(np.eye(6)):
                    rhs=op.project_zero_mean(op.compute_rhs(macro))
                    assert v.relative(rhs.cpu().numpy()[0],rhs_cpu[:,j])<1e-13
                    assert v.relative(op.compute_avg_stress(u,macro)[0],recover(u.cpu().numpy()[0],j))<1e-13
                    solution,info=b.solve(op,rhs)
                    value=solution.cpu().numpy()[0]
                    rr=float(np.linalg.norm(rhs_cpu[:,j]-K@value)/np.linalg.norm(rhs_cpu[:,j]));assert rr<=1.01e-10
                    columns.append(op.compute_avg_stress(solution,macro)[0])
                    calls.append(dict(independent_true_residual=rr,displacement_vs_cpu=v.relative(value.reshape(n,3),u_cpu[:,:,j]),report=info))
                ch_error=v.relative(np.column_stack(columns),CH_cpu);assert ch_error<1e-8
            else:ch_error=0.
            rows.append(dict(kind=kind,variant=variant,Ku_vs_cpu=error,rigid_max=rigid,symmetry=symmetry,CH_vs_cpu=ch_error,loads=calls,metadata=op._operator_metadata))
            print("PASS",kind,variant,error,ch_error,flush=True)
    return dict(passed=True,device=torch.cuda.get_device_name(),rows=rows,kernel_attributes=a.extension().attributes())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--phase",choices=("cpu","gpu"),required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();out=cpu() if args.phase=="cpu" else gpu()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(out,f,indent=2,allow_nan=False)
    print("PASS",args.phase,flush=True)
