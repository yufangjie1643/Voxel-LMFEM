"""192³ certification using the unchanged physical-C21 hybrid operator.

python certify.py --output results/certify_strong_192.json
This correctness run is separate from all reported performance measurements.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import study as s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    b=s.cs.s.load("fi");mf=s.a.prior.install(b,s.cs);s.cs.a.configure(b,"fi","combined")
    s.cs.a.native.extension();s.a.extension();s.a.install(b,"split_delta",64);b.TOL=1e-8
    b.gpu_guard();data,geometry,material,digest=b.inputs(192,"strong")
    phase,table,orient=mf.from_legacy_yarn(data,material)
    adapted=dict(grid_shape=data["grid_shape"],domain_size=data["domain_size"],phase_id=phase,orientation=orient)
    op=b.build(adapted,table,"direct","mixed",b.Stages())
    # Separate owner built from physical input, never derived from the delta pool.
    _,hm=s.a.prior.material_modules()
    pool,index,_=hm.build_hybrid_material(phase,table,orientation=orient)
    trusted=s.a.prior.HybridFIOperator(pool,index,data["grid_shape"],op._modal_voxel)
    rows=[];columns=[]
    for j,macro in enumerate(np.eye(6)):
        rhs=op.project_zero_mean(op.compute_rhs(macro));u,info=b.solve(op,rhs)
        original_rhs=trusted.project_zero_mean(trusted.compute_rhs(macro))
        norm=torch.linalg.vector_norm(original_rhs)
        physical_residual=float(torch.linalg.vector_norm(original_rhs-trusted.apply(u))/norm)
        rhs_error=float(torch.linalg.vector_norm(rhs-original_rhs)/norm)
        stress=op.compute_avg_stress(u,macro);reference_stress=trusted.compute_avg_stress(u,macro)
        stress_error=float(np.linalg.norm(stress-reference_stress)/np.linalg.norm(reference_stress))
        assert physical_residual<=1e-8 and rhs_error<1e-13 and stress_error<1e-13
        rows.append(dict(load=j,physical_operator_true_residual=physical_residual,rhs_relative=rhs_error,
                         stress_relative=stress_error,solve=info));columns.append(stress[0])
        print("CERTIFIED",j,physical_residual,stress_error,flush=True)
        del rhs,u,original_rhs
    b.gpu_guard();result=dict(passed=True,grid=192,case="strong",input_hash=digest,rows=rows,
        CH=np.column_stack(columns).tolist(),sources=s.sources(),
        scope="Correctness only: separately built physical-C21 pool and unchanged prior hybrid Ku/RHS/stress; not a timed memory/performance run.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=="__main__":main()
