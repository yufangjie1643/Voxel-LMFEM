"""FI controls for variational recovery and the stiffness-error stopping rule."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import adapter as a
import study as hg
import numpy as np
from scipy.linalg import eigh

HERE=Path(__file__).resolve().parent
CASES={"fi128":(128,"strong"),"fi192":(192,"strong"),"fi_near128":(128,"near")}
VARIANTS=("dense","energy_recovery","energy_fixed","energy_certified")


def certificate_setup(op,material):
    cm,cy=a.matrices(material);c0=np.asarray(op._formula_reference_C0)
    op._certificate_a=min(float(eigh(cm,c0,eigvals_only=True)[0]),float(eigh(cy,c0,eigvals_only=True)[0]))
    w=np.array([1.,1.,1.,2**.5,2**.5,2**.5])
    cmin=min(np.linalg.eigvalsh(cm*w[:,None]*w[None,:])[0],np.linalg.eigvalsh(cy*w[:,None]*w[None,:])[0])
    op._certificate_target=float(1e-8*np.sqrt(6)*cmin/6)
    # All attributes changed by enable_fi_reference belong to the shallow copy.
    # Shared material arrays are read only; the iteration preconditioner is unchanged.
    reference=copy.copy(op);a.prior.formula.enable_fi_reference(reference,"fp64")
    op.exact_precond=reference.apply_precond


def workload(b,parent,data,material,variant):
    torch=b.torch;b.cleanup();st=b.Stages();begin=time.perf_counter()
    op=parent(data,material,"direct","mixed",st)
    a.prior.formula.pipeline.attach_energy(op);st.end("fi_attachment",op)
    energy=variant!="dense";custom=variant in ("energy_fixed","energy_certified")
    if custom:certificate_setup(op,material);st.end("certificate_setup",op)
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
        ch=ch-(all_u@all_r.T).cpu().numpy()/volume;del all_u,all_r
        if custom:bound=sum(c["bound"] for c in calls)
        st.end("cross_energy",op)
    b.synchronize();seconds=time.perf_counter()-begin
    result=dict(seconds=seconds,CH=ch.tolist(),stress_CH=raw.tolist(),calls=calls,stages=st.rows,
        energy_error_absolute_bound_Mandel=bound,peak_allocated_bytes=max(s["peak"] for s in st.rows),
        peak_reserved_bytes=max(s["reserved"] for s in st.rows))
    del op;b.cleanup();return result


def group(case,repeat,small=False):
    b=a.prior.previous.load_runtime("fi");b.gpu_guard();a.prior.formula.configure(b,"fi","analytic");parent=b.build
    a.prior.formula.extension();a.prior.formula.pipeline.extension();a.prior.prior.modal.extension()
    grid,material_case=CASES[case];grid=8 if small else grid
    data,geom,material,digest=b.inputs(grid,material_case)
    output=HERE/"results"/f"{'fi_smoke' if small else 'fi_measure'}_{case}_{repeat}.json"
    if output.exists():raise FileExistsError(output)
    record=dict(status="running",element="FI",case=case,grid=grid,repeat=repeat,geometry=geom,material=material,
        input_hash=digest,provenance=hg.provenance(b),fi_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=[],
        protocol="FI current analytic FP32 reference baseline; three independent process repeats; interleaved variants; full same-size six-load warmup then full measurement; all certificate/storage/cross-energy costs included")
    order=list(VARIANTS);random.Random(20261021+repeat).shuffle(order)
    try:
        for variant in order:
            b.gpu_guard();workload(b,parent,data,material,variant)
            b.gpu_guard();m=workload(b,parent,data,material,variant);b.gpu_guard()
            maximum=max(v for c in m["calls"] for v in c["true_residuals"])
            if variant!="energy_certified" and maximum>1e-8:raise RuntimeError("FI true residual failed")
            record["runs"].append(dict(variant=variant,measurement=m,max_true_residual=maximum));hg.save(output,record)
            print("FI_RUN",case,repeat,variant,m["seconds"],flush=True)
        reference=next(r["measurement"]["CH"] for r in record["runs"] if r["variant"]=="dense")
        for row in record["runs"]:
            m=row["measurement"];row["CH_Mandel_relative_to_dense"]=hg.error(m["CH"],reference,True)
            if row["CH_Mandel_relative_to_dense"]>1e-8:raise RuntimeError("FI stiffness mismatch")
            constants=1/np.diag(np.linalg.inv(m["CH"]));base=1/np.diag(np.linalg.inv(reference))
            row["engineering_constants_relative"]=hg.error(constants,base)
            row["engineering_constants"]={name:float(v) for name,v in zip(("E11","E22","E33","G23","G13","G12"),constants)}
            if m["energy_error_absolute_bound_Mandel"] is not None:
                w=np.array([1.,1.,1.,2**.5,2**.5,2**.5]);norm=np.linalg.norm(np.asarray(m["CH"])*w[:,None]*w[None,:]);bound=m["energy_error_absolute_bound_Mandel"]
                row["certified_relative_bound"]=bound/max(norm-bound,1e-300)
                if row["certified_relative_bound"]>1e-8:raise RuntimeError("FI certificate exceeds target")
        record["status"]="passed";hg.save(output,record)
    except BaseException as exc:record.update(status="failed",error=repr(exc));hg.save(output,record);raise


def orchestrate():
    path=HERE/"fi_status.json"
    if path.exists():raise FileExistsError(path)
    state=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),completed=[],total_groups=9);hg.save(path,state)
    try:
        jobs=[(case,r) for r in range(3) for case in CASES];random.Random(20261021).shuffle(jobs)
        for case,repeat in jobs:
            if a.prior.previous.foreign():raise RuntimeError("GPU occupied")
            label=f"{case}_{repeat}";state["active"]=label;hg.save(path,state)
            with (HERE/"logs"/(label+".log")).open("x") as log:
                child=subprocess.Popen([sys.executable,"-u",str(Path(__file__)),"--phase","group","--case",case,"--repeat",str(repeat)],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                samples=[]
                while child.poll() is None:
                    time.sleep(2);others=a.prior.previous.foreign(child.pid)
                    if others:
                        import os,signal
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise RuntimeError("Stopped own FI job after foreign GPU activity")
                    samples.append(dict(time=time.time(),gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()))
            hg.save(HERE/"logs"/(label+"_telemetry.json"),samples)
            if child.returncode:raise RuntimeError("FI group failed: "+label)
            state["completed"].append(label);hg.save(path,state)
        state.update(status="complete",ended_utc=datetime.now(timezone.utc).isoformat());state.pop("active",None);hg.save(path,state)
    except BaseException as exc:state.update(status="stopped",error=repr(exc));hg.save(path,state);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("group","smoke","orchestrate"),required=True)
    p.add_argument("--case",choices=tuple(CASES),default="fi128");p.add_argument("--repeat",type=int,default=0);args=p.parse_args()
    if args.phase=="orchestrate":orchestrate()
    else:group(args.case,args.repeat,args.phase=="smoke")
