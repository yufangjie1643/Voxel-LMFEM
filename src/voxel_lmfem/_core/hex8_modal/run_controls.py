"""Secondary native-atomic controls to isolate arithmetic from assembly.

Run only after the main study releases the GPU. The original HG atomic variant
and FI eight-Gauss-point atomic route remain entirely unmodified.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import gpu_study as study


HERE=Path(__file__).resolve().parent
OUT=HERE/"native_controls"


def workload(kind,repeat,grid):
    b=study.load_runtime(kind)
    original=b.build
    def build(data,material,route,precision,stages):
        op=original(data,material,"direct",precision,stages)
        if kind=="hg":op.set_apply_backend("cuda",cuda_variant="atomic",cuda_block_size=128)
        return op
    b.build=build
    study.OUT=OUT
    OUT.mkdir(exist_ok=True)
    study.measure(kind,"baseline",repeat,grid)
    path=OUT/f"measure_{kind}_baseline_{grid}_{repeat}.json"
    record=json.loads(path.read_text())
    record["secondary_control"]="unmodified native HG atomic128" if kind=="hg" else "unmodified eight-Gauss-point FI atomic128"
    study.save(path,record)


def micro(kind,grid):
    b=study.load_runtime(kind)
    torch=b.torch
    b.gpu_guard()
    data,geom,material,digest=b.inputs(grid,"strong")
    op=b.build(data,material,"direct","mixed",b.Stages())
    if kind=="hg":op.set_apply_backend("cuda",cuda_variant="atomic",cuda_block_size=128)
    torch.manual_seed(91620)
    u=torch.randn((1,3*grid**3),device="cuda",dtype=torch.float64)
    for _ in range(3):value=op.apply(u)
    b.synchronize();samples=[]
    for _ in range(7):
        first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        first.record()
        for _ in range(10):value=op.apply(u)
        last.record();last.synchronize();samples.append(first.elapsed_time(last)/10)
    native_value=value
    import modal_adapter
    voxel=data['domain_size'][0]/b.np.array(data['grid_shape'][::-1])
    modal_adapter.attach(op,kind,voxel,variant="atomic",block=128)
    modal_value=op.apply(u)
    difference=float(torch.linalg.vector_norm(modal_value-native_value)/torch.linalg.vector_norm(native_value))
    if difference>=1e-12:raise RuntimeError("Native atomic versus modal mismatch")
    native_attributes=None
    if kind=="fi":
        import atomic_adapter
        native_attributes=atomic_adapter.extension().attributes()
    study.save(OUT/f"micro_{kind}_{grid}.json",dict(kind=kind,grid=grid,
        secondary_control="native HG atomic128" if kind=="hg" else "native FI eight-Gauss-point atomic128",
        milliseconds=samples,median_ms=statistics.median(samples),input_hash=digest,
        Ku_relative_to_modal=difference,native_attributes=native_attributes,provenance=study.provenance(b)))
    print("NATIVE MICRO",kind,statistics.median(samples),flush=True)


def orchestrate(grid):
    OUT.mkdir(exist_ok=True)
    status=OUT/"status.json"
    if status.exists():raise FileExistsError(status)
    state=dict(status="waiting_for_main_study",completed=[]);study.save(status,state)
    while True:
        main=json.loads((HERE/"status.json").read_text())
        if main["status"]=="complete":break
        if main["status"]!="running":raise RuntimeError("Main study did not finish: "+str(main))
        time.sleep(5)
    state["status"]="running";study.save(status,state)
    jobs=[("micro",kind,0) for kind in ("hg","fi")]
    jobs += [("measure",kind,repeat) for repeat in range(3) for kind in ("hg","fi")]
    try:
        for phase,kind,repeat in jobs:
            label=f"{phase}_{kind}_{repeat}"
            if study.foreign():raise RuntimeError("GPU occupied: "+str(study.foreign()))
            state["active"]=label;study.save(status,state)
            command=[sys.executable,"-u",str(Path(__file__).resolve()),"--phase",phase,"--kind",kind,
                     "--repeat",str(repeat),"--grid",str(grid)]
            samples=[]
            with (OUT/(label+".log")).open("w") as log:
                child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    time.sleep(1)
                    if study.foreign(child.pid):
                        child.terminate();child.wait(timeout=30)
                        raise RuntimeError("Stopped only our control because GPU became occupied")
                    telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                    samples.append(dict(time=time.time(),gpu=telemetry))
                code=child.returncode
            study.save(OUT/(label+"_telemetry.json"),samples)
            if code:raise RuntimeError(f"Control {label} failed")
            state["completed"].append(label);study.save(status,state);print("CONTROL DONE",label,flush=True)
        state["status"]="complete";state.pop("active",None);study.save(status,state)
    except BaseException as error:
        state.update(status="stopped",error=repr(error));study.save(status,state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["orchestrate","micro","measure"],required=True)
    p.add_argument("--kind",choices=["hg","fi"],default="hg")
    p.add_argument("--repeat",type=int,default=0)
    p.add_argument("--grid",type=int,default=192)
    a=p.parse_args()
    if a.phase=="orchestrate":orchestrate(a.grid)
    elif a.phase=="micro":micro(a.kind,a.grid)
    else:workload(a.kind,a.repeat,a.grid)


if __name__=="__main__":main()
