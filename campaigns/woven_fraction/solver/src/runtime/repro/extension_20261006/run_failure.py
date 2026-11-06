"""End-to-end multiple-RVE first-failure envelopes with identical controls."""
import argparse,gc
from failure import *


def unique_dataset(data):
    seen={};keep=[];duplicates=[];hashes=[]
    for i in range(len(data['indices'])):
        mask=data['yarn_id'][i]>=0;h=hashlib.sha256()
        for x in [mask,data['orient1'][i][mask],data['orient2'][i][mask],data['domain_size'][i]]:
            h.update(np.ascontiguousarray(x).tobytes())
        digest=h.hexdigest();hashes.append(digest)
        if digest in seen:duplicates.append([i,seen[digest]])
        else:seen[digest]=i;keep.append(i)
    selected={k:(v[keep] if isinstance(v,np.ndarray) else v) for k,v in data.items()}
    return selected,dict(input_count=len(hashes),unique_count=len(keep),duplicates=duplicates,
                         physical_input_sha256=hashes,selected_indices=keep)


def workload(data,B,d):
    rows=[];stages=dict(build=0.,solve_and_localize=0.,envelope=0.,cleanup=0.)
    torch.cuda.empty_cache();synchronize();torch.cuda.reset_peak_memory_stats()
    total=0.
    for lo in range(0,len(data['indices']),B):
        gpu_guard() # Exclude scheduler/occupancy checks consistently for all B.
        hi=min(lo+B,len(data['indices']));start=time.perf_counter()
        solver=build_batch(data,lo,hi,FAILURE);op=solver.op
        R,mask=rotation_fields(data,lo,hi,op.device)
        synchronize();t=time.perf_counter();stages['build']+=t-start
        ch,S,reports=local_stress_basis(op,R)
        synchronize();t1=time.perf_counter();stages['solve_and_localize']+=t1-t
        onset,mode,inverse=envelope(S,mask,d)
        synchronize();t2=time.perf_counter();stages['envelope']+=t2-t1
        for i in range(hi-lo):
            rows.append(dict(index=int(data['indices'][lo+i]),sample_id=str(data['sample_ids'][lo+i]),
              C_H_raw=ch[i].tolist(),onset_MPa=onset[i].tolist(),controlling_mode=mode[i].tolist(),
              true_residuals=[r['true_residuals'][i] for r in reports],
              batch_iterations=[r['iterations'] for r in reports]))
        del op,solver,R,mask,S;torch.cuda.empty_cache();synchronize()
        end=time.perf_counter();stages['cleanup']+=end-t2;total+=end-start
    gpu_guard()
    return dict(B=B,seconds=total,stages=stages,results=rows,
       peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--angles',type=int,default=360)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    gpu_guard();torch.set_num_threads(1)
    data,audit=unique_dataset(read_dataset(args.data));d=directions(args.angles)
    print('DISTINCT INPUTS',audit['unique_count'],'OF',audit['input_count'],flush=True)
    runs=[]
    # Same complete-workload warmup for both controls, excluded from reporting.
    for B in [1,4]:workload(data,B,d)
    for repeat,order in enumerate([[1,4],[4,1],[1,4]]):
        for B in order:
            r=workload(data,B,d);r['repeat']=repeat;runs.append(r)
            save_json(args.out,dict(status='running',runs=runs,audit=audit))
            print('ENVELOPE',B,repeat,r['seconds'],r['stages'],flush=True)
    ref=np.array([r['onset_MPa'] for r in runs[0]['results']])
    error=max(float(np.max(np.abs(np.array([s['onset_MPa'] for s in r['results']])/ref-1))) for r in runs)
    assert error<1e-8
    save_json(args.out,dict(status='passed',material=FAILURE,provenance=provenance(),audit=audit,
       dataset_sha256=data['sha256'],directions=d.tolist(),runs=runs,batch_onset_relative_difference=error,
       protocol='FP64 true residual 1e-10; six basis loads for both controls; physical local-field recovery and all rays included; three interleaved repetitions; CPU input/disk/JIT and occupancy checks excluded; batch cleanup included',
       planes=['Sigma11-Sigma22','Sigma22-Sigma12'],angles_per_plane=args.angles))

if __name__=='__main__':main()
