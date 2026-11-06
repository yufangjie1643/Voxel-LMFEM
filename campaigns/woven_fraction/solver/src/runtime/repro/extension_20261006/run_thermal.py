"""Compact thermal-expansion application: three existing woven cells."""
import argparse
from pathlib import Path
import time
import numpy as np
from physics import *


def workload(data, B):
    results=[]
    torch.cuda.empty_cache();synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for lo in range(0,len(data['indices']),B):
        gpu_guard();hi=min(lo+B,len(data['indices']))
        solver=build_batch(data,lo,hi);op=solver.op
        synchronize();t=time.perf_counter()
        C,_,mechanical=elastic_basis(op)
        synchronize();six=time.perf_counter()-t
        t=time.perf_counter()
        R,mask=rotation_fields(data,lo,hi,op.device)
        alpha=expansion_field(R,mask)
        ah,u,average,thermal=thermal_response(op,C,alpha)
        synchronize();thermal_seconds=time.perf_counter()-t
        for i in range(hi-lo):
            results.append(dict(index=int(data['indices'][lo+i]),
                sample_id=str(data['sample_ids'][lo+i]),yarn_fraction=float(mask[i].double().mean()),
                alpha_H=ah[i].tolist(),C_H_raw=C[i].tolist(),thermal_stress=average[i].tolist(),
                thermal_true_residual=thermal['true_residuals'][i],
                mechanical_true_residuals=[m['true_residuals'][i] for m in mechanical],
                mechanical_batch_seconds=six,thermal_batch_seconds=thermal_seconds,
                mechanical_batch_iterations=[m['iterations'] for m in mechanical],
                thermal_batch_iterations=thermal['iterations']))
        del solver,op,R,mask,alpha,u;torch.cuda.empty_cache()
    synchronize()
    return dict(B=B,seconds=time.perf_counter()-start,results=results,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    gpu_guard();torch.set_num_threads(1)
    data=read_dataset(args.data,[0,42,63]);rows=[]
    # Identical full seven-load prewarm; results excluded from timing.
    for B in [1,3]:workload(data,B)
    for repeat,order in enumerate([[1,3],[3,1],[1,3]]):
        for B in order:
            r=workload(data,B);r['repeat']=repeat;rows.append(r)
            save_json(args.out,dict(status='running',runs=rows))
            print('THERMAL',B,repeat,r['seconds'],flush=True)
    reference=np.array([r['alpha_H'] for r in rows[0]['results']])
    error=max(float(np.linalg.norm(np.array([v['alpha_H'] for v in r['results']])-reference)/np.linalg.norm(reference)) for r in rows)
    assert error<1e-8
    # Independent combined free-expansion solve; not part of timing.
    free=[]
    for i in range(3):
        gpu_guard();solver=build_batch(data,i,i+1);op=solver.op
        C,_,_=elastic_basis(op);R,mask=rotation_fields(data,i,i+1,op.device)
        alpha=expansion_field(R,mask);ah,_,_,_=thermal_response(op,C,alpha)
        rhs=op.project_zero_mean(eigenstrain_rhs(op,alpha)+op.compute_rhs(ah[0]))
        u,report=solve_checked(op,rhs)
        _,stress=op.recover_center_fields(u,ah[0]);stress-=_packed_matvec(op.C,alpha)
        average=stress.mean(1)
        relative=float(torch.linalg.vector_norm(average)/torch.linalg.vector_norm(_packed_matvec(op.C,alpha).mean(1)))
        assert relative<1e-8
        free.append(dict(index=int(data['indices'][i]),relative_free_stress=relative,solve=report))
        del solver,op,R,mask,alpha,u,stress;torch.cuda.empty_cache()
    save_json(args.out,dict(status='passed',material=THERMAL,provenance=provenance(),
        dataset_sha256=data['sha256'],indices=data['indices'].tolist(),runs=rows,
        batch_alpha_relative_difference=error,free_expansion_checks=free,
        protocol='FP64; seven sequential loads; full setup/field recovery/cleanup; disk and JIT excluded; three interleaved repetitions'))


if __name__=='__main__':
    from femlib.batched.c3d8r_hourglass_operator import _packed_matvec
    main()
