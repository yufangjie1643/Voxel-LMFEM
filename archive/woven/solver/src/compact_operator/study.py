"""Seven-case GPU factorial/solver ablations with full six-load outputs."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import adapter as a
import numpy as np

HERE=Path(__file__).resolve().parent
CASES={"woven64":(64,"strong"),"woven128":(128,"strong"),"woven192":(192,"strong"),
       "near128":(128,"near"),"aligned128":(128,"aligned"),"contrast128":(128,"contrast0.1"),"layered128":(128,"layered")}
save=a.prior.save


def inputs(b,name,small=False):
    grid,case=CASES[name];grid=8 if small else grid
    data,geometry,material,digest=b.inputs(grid,"strong" if case=="layered" else case)
    analytical=None
    if case=="layered":
        # Periodic two-layer benchmark, 50/50 by volume, z-normal interfaces.
        # Oblique constant yarn orientation gives nonzero response for all six loads.
        data=copy.deepcopy(data);z=np.indices((grid,grid,grid))[0]
        mask=z<grid//2;data["yarn_id"]=np.where(mask,0,-1).reshape(np.shape(data["yarn_id"])).astype(data["yarn_id"].dtype)
        data["orient1"][...]=[1.,2.,3.];data["orient2"][...]=[0.,1.,0.]
        geometry=dict(type="two_equal_periodic_layers",normal="z",yarn_axis=[1.,2.,3.],yarn_fraction=.5)
        from femlib.materials import build_rotation_matrices,rotate_voigt_batch
        cm,cy=a.matrices(material);r=build_rotation_matrices(np.array([[1.,2.,3.]]),np.array([[0.,1.,0.]]))
        cy=rotate_voigt_batch(cy,r)[0];normal=[2,3,4];parallel=[0,1,5]
        inv=[];couplings=[];schur=[]
        for c in (cm,cy):
            aa=c[np.ix_(normal,normal)];ab=c[np.ix_(normal,parallel)]
            ia=np.linalg.inv(aa);inv.append(ia);couplings.append(ia@ab)
            schur.append(c[np.ix_(parallel,parallel)]-ab.T@ia@ab)
        h=np.linalg.inv(sum(inv)/2);l=sum(couplings)/2
        analytical=np.empty((6,6));analytical[np.ix_(normal,normal)]=h
        analytical[np.ix_(normal,parallel)]=h@l;analytical[np.ix_(parallel,normal)]=l.T@h
        analytical[np.ix_(parallel,parallel)]=sum(schur)/2+l.T@h@l
    digest=hashlib.sha256()
    for key in ("yarn_id","orient1","orient2"):digest.update(np.ascontiguousarray(data[key]).tobytes())
    digest.update(json.dumps(material,sort_keys=True).encode())
    return data,geometry,material,digest.hexdigest(),analytical


def provenance(b):
    p=a.prior.provenance(b)
    p["ablation_sources"]={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("adapter.py","study.py","ablation_kernels.cu")}
    structural=a.STRUCTURAL/"structural_modal.cuh"
    p["structural_header_sha256"]=hashlib.sha256(structural.read_bytes()).hexdigest()
    return p


def workload(b,parent,data,material,variant):
    torch=b.torch;b.cleanup();st=b.Stages();begin=time.perf_counter()
    op=parent(data,material,"direct","mixed",st)
    a.attach(op,data,material,variant);st.end("ablation_attachment",op)
    energy=variant.startswith("energy");custom=variant in ("energy_fixed","energy_certified")
    if custom:a.certificate_setup(op,material);st.end("certificate_setup",op)
    if energy:
        all_u=torch.empty((6,3*op.mesh.n_elem),device=op.device,dtype=torch.float64)
        all_r=torch.empty_like(all_u);st.end("energy_storage",op)
    ch=np.empty((6,6));calls=[];volume=op.mesh.n_elem*op.mesh.V_elem
    for j in range(6):
        macro=np.eye(6)[j];rhs=op.project_zero_mean(op.compute_rhs(macro));st.end(f"rhs{j}",op)
        if custom:u,residual,report=a.custom_solve(op,rhs,j,variant=="energy_certified")
        else:
            u,report=b.solve(op,rhs)
            if energy:residual=rhs-op.apply(u)
        st.end(f"solve{j}",op)
        ch[:,j]=op.compute_avg_stress(u,macro)[0];calls.append(report)
        if energy:all_u[j].copy_(u[0]);all_r[j].copy_(residual[0]);del residual
        st.end(f"post{j}",op);del rhs,u
    raw=ch.copy();bound=None
    if energy:
        correction=(all_u@all_r.T).cpu().numpy()/volume
        ch=ch-correction;del all_u,all_r
        if custom:bound=sum(r["bound"] for r in calls)
        st.end("cross_energy",op)
    b.synchronize();seconds=time.perf_counter()-begin
    result=dict(seconds=seconds,CH=ch.tolist(),stress_CH=raw.tolist(),calls=calls,stages=st.rows,
        peak_allocated_bytes=max(r["peak"] for r in st.rows),peak_reserved_bytes=max(r["reserved"] for r in st.rows),
        energy_error_absolute_bound_Mandel=bound)
    del op;b.cleanup();return result


def error(x,y,mandel=False):
    x,y=np.asarray(x),np.asarray(y)
    if mandel:
        w=np.array([1.,1.,1.,2**.5,2**.5,2**.5]);x=x*w[:,None]*w[None,:];y=y*w[:,None]*w[None,:]
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(x),np.linalg.norm(y),1e-300))


def group(case,repeat,small=False):
    b=a.prior.previous.load_runtime("hg");b.gpu_guard()
    a.prior.formula.configure(b,"hg","baseline");parent=b.build
    a.extension();a.prior.formula.extension();a.prior.formula.pipeline.extension();a.prior.prior.modal.extension()
    data,geometry,material,digest,analytical=inputs(b,case,small)
    output=HERE/"results"/f"{'smoke' if small else 'measure'}_{case}_{repeat}.json"
    if output.exists():raise FileExistsError(output)
    record=dict(status="running",case=case,grid=int(data["grid_shape"][0]),repeat=repeat,geometry=geometry,material=material,
        input_hash=digest,analytical_CH=analytical.tolist() if analytical is not None else None,provenance=provenance(b),runs=[],
        protocol="three independent process repetitions per case; configurations interleaved within process; each configuration one complete same-size six-load warmup then a complete timed six-load solve; CPU input geometry, compilation, warmup, disk excluded; GPU setup, attachment, certification, all solves, stress and cross energy included; cache cleared between configurations")
    order=list(a.VARIANTS);random.Random(20261021+repeat).shuffle(order)
    try:
        for variant in order:
            b.gpu_guard();workload(b,parent,data,material,variant)
            b.gpu_guard();measurement=workload(b,parent,data,material,variant);b.gpu_guard()
            row=dict(variant=variant,measurement=measurement)
            max_res=max(v for c in measurement["calls"] for v in c["true_residuals"])
            row["max_true_residual"]=max_res
            if variant!="energy_certified" and max_res>1e-8:raise RuntimeError("Residual protocol failed: "+variant)
            record["runs"].append(row);save(output,record)
            print("RUN",case,repeat,variant,measurement["seconds"],flush=True)
        reference=next(r["measurement"]["CH"] for r in record["runs"] if r["variant"]=="dense")
        for row in record["runs"]:
            m=row["measurement"];row["CH_relative_to_dense"]=error(m["CH"],reference)
            row["CH_Mandel_relative_to_dense"]=error(m["CH"],reference,True)
            constants=1/np.diag(np.linalg.inv(m["CH"]));ref_constants=1/np.diag(np.linalg.inv(reference))
            row["engineering_constants"]={name:float(v) for name,v in zip(("E11","E22","E33","G23","G13","G12"),constants)}
            row["engineering_constants_relative"]=error(constants,ref_constants)
            if m["energy_error_absolute_bound_Mandel"] is not None:
                w=np.array([1.,1.,1.,2**.5,2**.5,2**.5]);norm=np.linalg.norm(np.asarray(m["CH"])*w[:,None]*w[None,:])
                absolute=m["energy_error_absolute_bound_Mandel"]
                row["certified_relative_bound"]=absolute/max(norm-absolute,1e-300)
                if row["certified_relative_bound"]>1e-8:raise RuntimeError("Certificate exceeds stiffness target")
            if analytical is not None:
                row["CH_relative_to_analytical"]=error(m["CH"],analytical,True)
                if row["CH_relative_to_analytical"]>1e-8:raise RuntimeError("Layered analytical verification failed: "+str(row))
            if row["CH_Mandel_relative_to_dense"]>1e-8:raise RuntimeError("Stiffness mismatch: "+str(row))
        record["status"]="passed";save(output,record)
    except BaseException as exc:
        record.update(status="failed",error=repr(exc),traceback=traceback.format_exc());save(output,record);raise


def orchestrate():
    state_path=HERE/"status.json"
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),completed=[],total_groups=21)
    save(state_path,state)
    jobs=[(name,rep) for rep in range(3) for name in CASES]
    random.Random(20261021).shuffle(jobs)
    try:
        for name,rep in jobs:
            if a.prior.previous.foreign():raise RuntimeError("GPU occupied")
            label=f"{name}_{rep}";state["active"]=label;save(state_path,state)
            with (HERE/"logs"/(label+".log")).open("x") as log:
                child=subprocess.Popen([sys.executable,"-u",str(HERE/"study.py"),"--phase","group","--case",name,"--repeat",str(rep)],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                samples=[]
                while child.poll() is None:
                    time.sleep(2)
                    others=a.prior.previous.foreign(child.pid)
                    if others:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise RuntimeError("Stopped our job after foreign GPU activity: "+str(others))
                    raw=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
                    samples.append(dict(time=time.time(),gpu=raw))
            save(HERE/"logs"/(label+"_telemetry.json"),samples)
            if child.returncode:raise RuntimeError("Group failed: "+label)
            state["completed"].append(label);save(state_path,state)
            print("DONE",label,flush=True)
        state.update(status="complete",ended_utc=datetime.now(timezone.utc).isoformat());state.pop("active",None);save(state_path,state)
    except BaseException as exc:state.update(status="stopped",error=repr(exc));save(state_path,state);raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--phase",choices=("group","smoke","orchestrate"),required=True)
    p.add_argument("--case",choices=tuple(CASES),default="woven128");p.add_argument("--repeat",type=int,default=0)
    args=p.parse_args()
    for folder in ("results","logs"):(HERE/folder).mkdir(exist_ok=True)
    if args.phase=="orchestrate":orchestrate()
    else:group(args.case,args.repeat,args.phase=="smoke")
