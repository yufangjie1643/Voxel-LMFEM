"""Follow-up: disposable cuFFT spectrum, reusable plans and shared workspace."""
from pathlib import Path
import argparse
import gc
import json
import random
import statistics
import subprocess
import sys
import time
import study as s
import native_adapter as native

OUT=s.OUT


def validate(kind):
    b=s.load(kind,configure=False);torch,np=b.torch,b.np;b.TOL=1e-10;native.extension()
    rows=[]
    for grid in [(5,6,7),(4,6,8)]:
        n=int(np.prod(grid));voxel=(.3,.5,.7);torch.manual_seed(993+n)
        a=torch.randn((1,n,6,6),device="cuda",dtype=torch.float64)
        full=a@a.transpose(-1,-2)+2*torch.eye(6,device="cuda",dtype=torch.float64)
        c=torch.stack([full[...,i,j] for i,j in b.PAIRS],dim=-1).contiguous()
        if kind=="hg":
            with s.f.formula.pipeline.lean_hg_initialization():op=b.make_solver(c,grid,voxel,fft=False).op
            s.f.prior.modal.attach(op,kind,voxel,variant="atomic",block=128)
            s.f.formula.pipeline.enable_analytic_reference(op)
        else:
            import atomic_adapter
            op=atomic_adapter.make(c,grid,"atomic128",voxel).op
            s.f.prior.modal.attach(op,kind,voxel,variant="atomic",block=128)
            s.f.formula.enable_fi_reference(op)
        op._validation_kind=kind
        matrix,rhs_cpu,u_cpu=s.f.previous.cpu_reference(b,op,full,voxel)
        r=torch.randn((1,3*n),device="cuda",dtype=torch.float64);r_before=r.clone();reference=op.apply_precond(r)
        for form in ("cached","dense"):
            native.attach(op,kind,form,release=False)
            plan=op._contest_plan;packed=s.c.extension().pack(r,*grid)
            frequency=plan.forward(packed)
            torch_frequency=torch.fft.rfftn(packed,dim=(-3,-2,-1)).contiguous()
            fft_error=s.relative(frequency,torch_frequency)
            inverse=plan.inverse_destructive(frequency)
            torch_inverse=torch.fft.irfftn(torch_frequency,s=grid,dim=(-3,-2,-1),norm="forward")
            ifft_error=s.relative(inverse,torch_inverse)
            assert fft_error<1e-6 and ifft_error<1e-6
            value=op.apply_precond(r);err=s.relative(value,reference);snapshot=value.clone();other=op.apply_precond(2*r)
            assert err<2e-6 and torch.equal(value,snapshot) and torch.equal(r,r_before)
            row=dict(kind=kind,grid=grid,form=form,preconditioner_relative=err,
                     native_fft_relative=fft_error,native_ifft_relative=ifft_error,workspace_bytes=plan.workspace_bytes)
            if grid==(5,6,7):
                calls=[];ch=[];trusted=[]
                for j,macro in enumerate(np.eye(6)):
                    rhs=op.project_zero_mean(op.compute_rhs(macro));u,report=b.solve(op,rhs)
                    v=u[0].cpu().numpy();true=float(np.linalg.norm(rhs_cpu[:,j]-matrix@v)/np.linalg.norm(rhs_cpu[:,j]))
                    assert true<=1.01e-10
                    ref_u=torch.as_tensor(u_cpu[:,:,j].ravel()[None,:].copy(),device="cuda")
                    ch.append(op.compute_avg_stress(u,macro)[0]);trusted.append(op.compute_avg_stress(ref_u,macro)[0])
                    calls.append(dict(report=report,independent_true_residual=true))
                row.update(CH=np.column_stack(ch).tolist(),CH_vs_direct=s.f.previous.relative(np.column_stack(ch),np.column_stack(trusted)),calls=calls)
                assert row["CH_vs_direct"]<1e-8
            rows.append(row);print("VALIDATE",kind,grid,form,err,fft_error,ifft_error,flush=True)
        del op,plan,a,full,c,r,r_before,reference,value,other,snapshot,packed,frequency,torch_frequency,inverse,torch_inverse
        gc.collect();torch.cuda.empty_cache()
    s.save(OUT/f"native_validate_{kind}.json",dict(complete=True,rows=rows,provenance=s.provenance(b)))


def probe(kind):
    b=s.load(kind);torch=b.torch;native.extension();n=192
    data,geometry,material,digest=b.inputs(n,"strong")
    op=b.build(data,material,"direct","mixed",b.Stages())
    torch.manual_seed(99);r=torch.randn((1,3*n**3),device="cuda",dtype=torch.float64)
    functions={"baseline":op.apply_precond};reference=op.apply_precond(r)
    selected=json.loads((OUT/f"selection_{kind}.json").read_text())["selected"]
    s.c.attach(op,kind,selected,release=False);functions[selected]=op.apply_precond
    for form in ("cached","dense"):
        native.attach(op,kind,form,release=False);functions["native_"+form]=op.apply_precond
    errors={};memory={};samples={name:[] for name in functions}
    for name,fn in functions.items():
        v=fn(r);errors[name]=s.relative(v,reference);assert errors[name]<2e-6
        for _ in range(3):v=fn(r)
        del v;b.synchronize()
        before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();v=fn(r);b.synchronize()
        memory[name]=torch.cuda.max_memory_allocated()-before;del v
    for repeat in range(9):
        order=list(functions);random.Random(881+repeat).shuffle(order)
        for name in order:
            b.gpu_guard();a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(10):v=functions[name](r)
            z.record();z.synchronize();samples[name].append(a.elapsed_time(z)/10);del v
    record=dict(kind=kind,grid=n,case="strong",complete=True,input_hash=digest,geometry=geometry,
                errors=errors,milliseconds=samples,median_ms={name:statistics.median(t) for name,t in samples.items()},
                incremental_peak_bytes=memory,scope="same-grid resident preconditioner; shared native workspace is persistent and excluded from incremental bytes; full workflow includes it",
                provenance=s.provenance(b))
    s.save(OUT/f"native_probe_{kind}.json",record)
    print(json.dumps({k:record[k] for k in ("kind","median_ms","incremental_peak_bytes")}),flush=True)


def measure(kind,variant,stage,repeat=0,grid=192,case="strong"):
    if not variant.startswith("native_"):
        return s.measure(kind,variant,grid,case,repeat,stage)
    b=s.load(kind);native.extension();old=b.build;form=variant.removeprefix("native_")
    def build(data,material,route,precision,stages):
        op=old(data,material,route,precision,stages);native.attach(op,kind,form);stages.end("contest_native_attachment",op);return op
    b.build=build;data,geometry,material,digest=b.inputs(grid,case)
    b.workload(data,material,"direct","mixed");b.gpu_guard()
    record,_=b.workload(data,material,"direct","mixed");b.gpu_guard()
    assert max(v for call in record["calls"] for v in call["true_residuals"])<=b.TOL
    path=OUT/f"{stage}_{kind}_{variant}_{case}_{grid}_{repeat}.json"
    if path.exists():raise FileExistsError(path)
    s.save(path,dict(kind=kind,variant=variant,grid=grid,case=case,repeat=repeat,stage=stage,tolerance=b.TOL,
                    input_hash=digest,geometry=geometry,material=material,measurement=record,provenance=s.provenance(b),
                    protocol="fresh process, full same-size warmup then measured six-load workflow; native plan construction and workspace included"))
    print(kind,variant,stage,record["seconds"],flush=True)


def orchestrate():
    assert (OUT/"complete.json").exists(),"Initial ablation must finish first"
    def run(*args):
        s.guard()
        with (OUT/("native_"+"_".join(map(str,args))+".log")).open("w") as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:time.sleep(2);s.guard((child.pid,))
            except Exception:
                child.terminate();child.wait();raise
            if child.returncode:raise RuntimeError(f"Native child failed {args}")
        print("DONE",*args,flush=True)
    for kind in ("hg","fi"):run("--phase","validate","--kind",kind)
    for kind in ("hg","fi"):run("--phase","probe","--kind",kind)
    selection={}
    for kind in ("hg","fi"):
        for variant in ("native_cached","native_dense"):
            run("--phase","measure","--kind",kind,"--variant",variant,"--stage","native_tune")
        times={v:json.loads((OUT/f"native_tune_{kind}_{v}_strong_192_0.json").read_text())["measurement"]["seconds"] for v in ("native_cached","native_dense")}
        selection[kind]=min(times,key=times.get)
        s.save(OUT/f"native_selection_{kind}.json",dict(times=times,selected=selection[kind]))
    for repeat in range(3):
        pairs=[]
        for kind in ("hg","fi"):
            old=json.loads((OUT/f"selection_{kind}.json").read_text())["selected"]
            pairs.extend((kind,v) for v in ("baseline",old,selection[kind]))
        random.Random(581+repeat).shuffle(pairs)
        for kind,variant in pairs:run("--phase","measure","--kind",kind,"--variant",variant,"--stage","native_formal","--repeat",repeat)
    for kind in ("hg","fi"):
        for variant in ("baseline",selection[kind]):
            run("--phase","measure","--kind",kind,"--variant",variant,"--stage","native_smoke","--grid",128,"--case","near")
    s.save(OUT/"native_complete.json",dict(complete=True,selection=selection))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("validate","probe","measure","orchestrate"),required=True)
    parser.add_argument("--kind",choices=("hg","fi"),default="hg")
    parser.add_argument("--variant",default="native_cached")
    parser.add_argument("--stage",default="native_tune")
    parser.add_argument("--repeat",type=int,default=0);parser.add_argument("--grid",type=int,default=192)
    parser.add_argument("--case",choices=("strong","near"),default="strong")
    args=parser.parse_args()
    if args.phase=="orchestrate":orchestrate()
    elif args.phase=="validate":validate(args.kind)
    elif args.phase=="probe":probe(args.kind)
    else:measure(args.kind,args.variant,args.stage,args.repeat,args.grid,args.case)


if __name__=="__main__":main()
