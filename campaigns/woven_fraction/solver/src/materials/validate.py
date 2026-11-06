"""CPU representation checks and independent eight-point FE verification.

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("FEAD_CHAIN", str(__import__('project_paths').source_path('solver_chain'))))
sys.path.insert(0, str(__import__('project_paths').source_path('runtime/repro/source')))
from hybrid_adapter import material_modules, HybridFIOperator, extension


def independent():
    path = HERE/"independent_validation.py"
    if not path.exists():
        path = HERE.parent/"material_interface_20261021/validate.py"
    spec = importlib.util.spec_from_file_location("independent_validation", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cpu():
    mf, hm = material_modules();v = independent();rows = []
    for kind in ("isotropic", "mixed"):
        data, table, full, h = v.fixture(kind, grid=5)
        phase = data["phase_id"]
        for chunk in (1, 37, 65536):
            pool, index, metadata = hm.build_hybrid_material(phase, table, orientation=data.get("orientation"), device="cpu", chunk_size=chunk)
            reconstructed = pool[index.long()].numpy()
            ref = mf.build_material_field(phase, table, orientation=data.get("orientation"), device="cpu", chunk_size=chunk).numpy()
            error = v.relative(reconstructed, ref)
            fourth = v.relative(reconstructed[0], np.stack([full[:, i, j] for i, j in mf.PAIRS], -1))
            assert error < 1e-14 and fourth < 1e-13
            rows.append(dict(kind=kind, chunk=chunk, dense_relative=error, independent_rotation_relative=fourth, storage=metadata))
        # Core API supports batches even though the GPU FI integration is B=1.
        batch = np.concatenate([phase, phase], axis=0)
        orient = None if kind == "isotropic" else np.concatenate([data["orientation"]]*2, axis=0)
        pool, index, meta = hm.build_hybrid_material(batch, table, orientation=orient, device="cpu", chunk_size=23)
        ref = mf.build_material_field(batch, table, orientation=orient, device="cpu", chunk_size=23)
        assert torch.allclose(pool[index.long()], ref, rtol=1e-14, atol=1e-12)
    # All-local case and global fully anisotropic phase, with arbitrary labels.
    rng = np.random.default_rng(71);A = rng.normal(size=(6, 6));C = A@A.T+np.eye(6)
    for local in (False, True):
        phase = np.full((3, 4, 5), -9, dtype=np.int64)
        table = {-9:dict(kind="anisotropic", frame="local" if local else "global", C=C)}
        kwargs = dict(orientation=np.eye(3) if local else None, device="cpu", chunk_size=13)
        pool, index, meta = hm.build_hybrid_material(phase, table, **kwargs)
        ref = mf.build_material_field(phase, table, **kwargs)
        assert torch.allclose(pool[index.long()], ref, rtol=1e-14, atol=1e-14)
        assert meta["n_local_voxels"] == (60 if local else 0)
        rows.append(dict(kind="all_local" if local else "global_anisotropic", storage=meta))
    for phase,table,kwargs in [
        (np.zeros((2,2,2),int), {1:dict(kind="isotropic",E=1,nu=.2)}, {}),
        (np.zeros((2,2,2),int), {0:dict(kind="anisotropic",frame="local",C=np.eye(6))}, {}),
        (np.zeros((2,2,2),int), {0:dict(kind="isotropic",E=1,nu=.2)}, {"chunk_size":0})]:
        try:hm.build_hybrid_material(phase,table,device="cpu",**kwargs)
        except ValueError:pass
        else:raise AssertionError("Invalid input accepted")
    return dict(passed=True, device="cpu", rows=rows)


def gpu():
    import chain_study as cs
    b = cs.s.load("fi", configure=False);b.TOL=1e-10
    cs.a.native.extension();extension()
    b.solve_pcg_batched = cs.a.make_solver("fi", True, True, b.solve_pcg_batched)
    v = independent();mf,hm = material_modules();rows=[]
    import atomic_adapter
    for kind in ("isotropic", "mixed"):
        data, table, full, h = v.fixture(kind, grid=5)
        grid = data["grid_shape"];n=int(np.prod(grid))
        K, rhs_cpu, u_cpu, CH_cpu, recover = v.assemble(full, grid, h)
        packed=mf.build_material_field(data["phase_id"],table,orientation=data.get("orientation"))
        pure = atomic_adapter.make(packed,grid,"atomic128",h).op
        cs.s.f.prior.modal.attach(pure,"fi",h,variant="atomic",block=128)
        cs.s.f.formula.enable_fi_reference(pure)
        reference=np.asarray(pure._formula_reference_C0)
        pool,index,meta=hm.build_hybrid_material(data["phase_id"],table,orientation=data.get("orientation"))
        hybrid=HybridFIOperator(pool,index,grid,h)
        dense=HybridFIOperator(packed,torch.empty(0,device="cuda",dtype=torch.int32),grid,h,dense=True)
        torch.manual_seed(72);u=torch.randn((1,3*n),device="cuda",dtype=torch.float64)
        expected=K@u.cpu().numpy()[0]
        for name,op in (("pure",pure),("dense",dense),("hybrid",hybrid)):
            computed=op._compute_reference_medium("zeman")[0]
            assert np.allclose(computed,reference,rtol=1e-14,atol=1e-12)
            if name!="pure":cs.s.f.formula.enable_fi_reference(op)
            cs.a.native.attach(op,"fi","dense");cs.a.attach_raw(op)
            Ku_error=v.relative(op.apply(u).cpu().numpy()[0],expected)
            assert Ku_error<1e-13
            calls=[];columns=[]
            for j in range(6):
                macro=np.eye(6)[j];rhs=op.project_zero_mean(op.compute_rhs(macro))
                rhs_error=v.relative(rhs.cpu().numpy()[0],rhs_cpu[:,j]);assert rhs_error<1e-13
                stress_error=v.relative(op.compute_avg_stress(u,macro)[0],recover(u.cpu().numpy()[0],j));assert stress_error<1e-13
                solution,info=b.solve(op,rhs)
                value=solution.cpu().numpy()[0]
                rr=float(np.linalg.norm(rhs_cpu[:,j]-K@value)/np.linalg.norm(rhs_cpu[:,j]))
                assert rr<=1.01e-10
                columns.append(op.compute_avg_stress(solution,macro)[0])
                calls.append(dict(independent_true_residual=rr,rhs_relative=rhs_error,
                                  random_stress_relative=stress_error,
                                  displacement_vs_cpu=v.relative(value.reshape(n,3),u_cpu[:,:,j]),report=info))
            error=v.relative(np.column_stack(columns),CH_cpu);assert error<1e-8
            rows.append(dict(kind=kind,variant=name,Ku_vs_cpu=Ku_error,CH_vs_cpu=error,loads=calls,storage=meta if name=="hybrid" else None))
            print("PASS",kind,name,Ku_error,error,flush=True)
        del pure,dense,hybrid,packed,pool,index,op,u,solution
    return dict(passed=True, device=torch.cuda.get_device_name(), rows=rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--phase",choices=("cpu","gpu"),required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=cpu() if a.phase=="cpu" else gpu()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print("PASS",a.phase,flush=True)
