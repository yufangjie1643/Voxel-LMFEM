"""Fixed-grid exact-operator FFT ablation and complete six-load GPU study."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
sys.path.insert(0, os.environ.get("FEAD_FORMULA", str(__import__('project_paths').source_path('reference_medium'))))
import formula_study as f
import contest_adapter as c


def save(path, data):
    f.save(path, data)


def guard(allowed=()):
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    foreign = [int(s) for s in text.split() if s.isdigit() and int(s) not in {os.getpid(), *allowed}]
    if foreign:raise RuntimeError(f"Foreign GPU processes {foreign}; stop own experiment only")


def provenance(b):
    data = f.provenance(b)
    data["contest_source_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in HERE.iterdir() if p.suffix in (".py", ".cu")}
    data["unchanged"] = ["fine grid", "all frequencies", "physical K", "reference M", "FP64 physical fields", "FP32 FFT", "PCG recurrence", "FP64 true residual tolerance"]
    return data


def load(kind, configure=True):
    b = f.previous.load_runtime(kind)
    b.gpu_guard();c.extension();f.formula.extension();f.prior.modal.extension();f.formula.pipeline.extension()
    if configure:f.formula.configure(b, kind, "analytic" if kind == "fi" else "baseline")
    return b


def relative(a, b):
    import torch
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-300))


def validate(kind):
    b = load(kind, configure=False);torch, np = b.torch, b.np;b.TOL=1e-10
    rows = []
    for grid in [(5, 6, 7), (4, 6, 8)]:
        voxel = (.3, .5, .7);n = int(np.prod(grid));torch.manual_seed(993+n)
        a = torch.randn((1, n, 6, 6), device="cuda", dtype=torch.float64)
        full = a@a.transpose(-1, -2)+2*torch.eye(6, device="cuda", dtype=torch.float64)
        packed = torch.stack([full[..., i, j] for i, j in b.PAIRS], dim=-1).contiguous()
        if kind == "hg":
            with f.formula.pipeline.lean_hg_initialization():op=b.make_solver(packed, grid, voxel, fft=False).op
            f.prior.modal.attach(op, kind, voxel, variant="atomic", block=128)
            f.formula.pipeline.enable_analytic_reference(op)
        else:
            import atomic_adapter
            op=atomic_adapter.make(packed, grid, "atomic128", voxel).op
            f.prior.modal.attach(op, kind, voxel, variant="atomic", block=128)
            f.formula.enable_fi_reference(op)
        op._validation_kind=kind
        matrix, cpu_rhs, cpu_u = f.previous.cpu_reference(b, op, full, voxel)
        residual = torch.randn((1, 3*n), device="cuda", dtype=torch.float64)
        before = residual.clone()
        baseline = op.apply_precond
        reference = baseline(residual)
        expected_pack = residual.float().view(1, -1, 3).transpose(1, 2).contiguous().view(1, 3, *grid)
        assert torch.equal(expected_pack, c.extension().pack(residual, *grid))
        baseline_CH = None
        for variant in ("baseline", "fused", "inplace", "cached", "dense"):
            if variant != "baseline":c.attach(op, kind, variant, release=False)
            value = op.apply_precond(residual)
            e = relative(value, reference)
            assert e < 2e-6, (kind, variant, grid, e)
            snapshot = value.clone()
            second = op.apply_precond(2*residual)
            assert torch.equal(value, snapshot) and torch.equal(residual, before), "Output/residual alias corruption"
            row = dict(kind=kind, grid=grid, variant=variant, preconditioner_relative=e)
            if grid == (5, 6, 7):
                loads, columns = [], []
                for j, macro in enumerate(np.eye(6)):
                    rhs = op.project_zero_mean(op.compute_rhs(macro))
                    u, report = b.solve(op, rhs)
                    flat = u[0].cpu().numpy()
                    independent = float(np.linalg.norm(cpu_rhs[:, j]-matrix@flat)/np.linalg.norm(cpu_rhs[:, j]))
                    assert independent <= 1.01e-10
                    stress = op.compute_avg_stress(u, macro)[0]
                    columns.append(stress)
                    loads.append(dict(report=report, independent_true_residual=independent,
                                      displacement_vs_direct=f.previous.relative(flat.reshape(n, 3), cpu_u[:, :, j])))
                ch = np.column_stack(columns)
                if baseline_CH is None:baseline_CH = ch.copy()
                row.update(CH=ch.tolist(), CH_relative=f.previous.relative(ch, baseline_CH), loads=loads)
                assert row["CH_relative"] < 1e-8
            rows.append(row)
            print("VALIDATE", kind, grid, variant, e, row.get("CH_relative"), flush=True)
        del op, a, full, packed, residual, before, reference, value, second, baseline, snapshot
        gc.collect();torch.cuda.empty_cache()
    save(OUT/f"validate_{kind}.json", dict(complete=True, rows=rows, provenance=provenance(b)))


def probe(kind, grid):
    b = load(kind);torch = b.torch
    data, geometry, material, digest = b.inputs(grid, "strong")
    op = b.build(data, material, "direct", "mixed", b.Stages())
    baseline = op.apply_precond
    torch.manual_seed(99);r=torch.randn((1, 3*grid**3), device="cuda", dtype=torch.float64)
    reference=baseline(r)
    functions={"baseline":baseline};errors={"baseline":0.}
    coefficients={"baseline":op._fft_symbol_inv.numel()*4}
    for variant in ("fused", "inplace", "cached", "dense"):
        c.attach(op, kind, variant, release=False)
        functions[variant]=op.apply_precond
        coefficients[variant]=op._contest_metadata["coefficients_bytes"]
        out=functions[variant](r);errors[variant]=relative(out, reference)
        assert errors[variant] < 2e-6
    del out
    for fn in functions.values():
        for _ in range(3):out=fn(r)
    del out;b.synchronize()
    samples={v:[] for v in functions};memory={}
    for variant, fn in functions.items():
        b.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        out=fn(r);b.synchronize()
        memory[variant]=dict(incremental_peak_bytes=torch.cuda.max_memory_allocated()-before,
                             output_bytes=out.numel()*out.element_size())
        del out
    for repeat in range(9):
        order=list(functions);random.Random(901+repeat).shuffle(order)
        for variant in order:
            b.gpu_guard()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):out=functions[variant](r)
            end.record();end.synchronize();samples[variant].append(start.elapsed_time(end)/10)
            del out
    result=dict(kind=kind, grid=grid, complete=True, case="strong", geometry=geometry, input_hash=digest,
                errors=errors, milliseconds=samples, median_ms={v:statistics.median(t) for v,t in samples.items()},
                incremental_memory=memory, coefficient_bytes=coefficients,
                scope="resident same-grid same-reference preconditioner; 9 interleaved event batches x 10 calls; setup/JIT excluded; coefficient caches coexist for comparison",
                provenance=provenance(b))
    save(OUT/f"probe_{kind}_{grid}.json", result)
    print(json.dumps({k:result[k] for k in ("kind","grid","median_ms","incremental_memory")}), flush=True)


def measure(kind, variant, grid, case, repeat, stage):
    b=load(kind)
    if variant != "baseline":
        old_build=b.build
        def build(data, material, route, precision, stages):
            op=old_build(data, material, route, precision, stages)
            c.attach(op, kind, variant)
            stages.end("contest_attachment", op)
            return op
        b.build=build
    data, geometry, material, digest=b.inputs(grid, case)
    b.workload(data, material, "direct", "mixed")
    b.gpu_guard();result,_=b.workload(data, material, "direct", "mixed");b.gpu_guard()
    assert max(v for call in result["calls"] for v in call["true_residuals"]) <= b.TOL
    path=OUT/f"{stage}_{kind}_{variant}_{case}_{grid}_{repeat}.json"
    if path.exists():raise FileExistsError(path)
    record=dict(kind=kind,variant=variant,grid=grid,case=case,repeat=repeat,stage=stage,
                tolerance=b.TOL, input_hash=digest,geometry=geometry,material=material,
                measurement=result,provenance=provenance(b),
                protocol="fresh process, same-size full six-load warmup then complete six-load measured workflow; attachment/setup included")
    save(path,record)
    print(kind,variant,case,grid,repeat,result["seconds"],flush=True)


def orchestrate():
    OUT.mkdir(parents=True,exist_ok=True)
    def run(*args):
        guard();path=OUT/("_".join(map(str,args))+".log")
        with path.open("w") as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(2);guard((process.pid,))
            except Exception:
                process.terminate();process.wait();raise
            if process.returncode:raise RuntimeError(f"Child failed {args}: {path}")
        print("DONE",*args,flush=True)
    for kind in ("hg","fi"):run("--phase","validate","--kind",kind)
    for kind in ("hg","fi"):
        for grid in (64,128,192):run("--phase","probe","--kind",kind,"--grid",grid)
    selected={}
    variants=("baseline","fused","inplace","cached","dense")
    for kind in ("hg","fi"):
        for variant in variants:run("--phase","measure","--kind",kind,"--variant",variant,"--stage","tune")
        times={v:json.loads((OUT/f"tune_{kind}_{v}_strong_192_0.json").read_text())["measurement"]["seconds"] for v in variants}
        selected[kind]=min(variants[1:],key=lambda v:times[v])
        save(OUT/f"selection_{kind}.json",dict(times=times,selected=selected[kind],protocol="one complete-workflow tuning run per variant; selected fastest nonbaseline before formal repeats"))
    for repeat in range(3):
        pairs=[(kind,v) for kind in ("hg","fi") for v in ("baseline",selected[kind])]
        random.Random(391+repeat).shuffle(pairs)
        for kind,variant in pairs:run("--phase","measure","--kind",kind,"--variant",variant,"--stage","formal","--repeat",repeat)
    for kind in ("hg","fi"):
        for variant in ("baseline",selected[kind]):
            run("--phase","measure","--kind",kind,"--variant",variant,"--stage","smoke","--case","near","--grid",128)
    save(OUT/"complete.json",dict(complete=True,selected=selected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("validate","probe","measure","orchestrate"),required=True)
    parser.add_argument("--kind",choices=("hg","fi"),default="hg")
    parser.add_argument("--variant",choices=("baseline","fused","inplace","cached","dense"),default="baseline")
    parser.add_argument("--grid",type=int,default=192)
    parser.add_argument("--case",choices=("strong","near"),default="strong")
    parser.add_argument("--repeat",type=int,default=0)
    parser.add_argument("--stage",choices=("tune","formal","smoke"),default="tune")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=="orchestrate":orchestrate()
    elif args.phase=="validate":validate(args.kind)
    elif args.phase=="probe":probe(args.kind,args.grid)
    else:measure(args.kind,args.variant,args.grid,args.case,args.repeat,args.stage)


if __name__=="__main__":main()
