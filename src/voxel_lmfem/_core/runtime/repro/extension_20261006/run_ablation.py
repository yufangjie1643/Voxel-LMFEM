"""One matched configuration per fresh process; full warmup and true residuals."""
import argparse,os,gc,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--repeat',type=int,default=0)
p.add_argument('--material',choices=['strong','near'],default='strong');p.add_argument('--validate-only',action='store_true');p.add_argument('--profile',action='store_true')
a=p.parse_args()
CONFIGS={'A00':(False,False,2,4,0),'A01':(False,True,2,4,0),'A10':(True,False,2,4,0),
 'A11':(True,True,2,4,0),'F0':(True,True,0,4,0),'F1':(True,True,1,4,0),
 'U':(False,False,0,4,0),'O27':(True,True,2,4,27),'O14':(True,True,2,4,14),
 'B1':(True,True,2,1,0),'B2':(True,True,2,2,0),'B8':(True,True,2,8,0),'B16':(True,True,2,16,0)}
selected,packed,fft,B,taps=CONFIGS[a.config]
if not packed:os.environ['FEAD_SOURCE']=str(Path(__file__).parent/'dense_control')
from material_control import *
from verify_thermal import rotate,assembled
from scipy.sparse.linalg import spsolve
from run_failure import unique_dataset
material=STRONG if a.material=='strong' else NEAR
gpu_guard();torch.set_num_threads(1)

if taps:
    # Reuse exactly the previously verified 27/14 HG assembly and apply code.
    source=ROOT/'evidence/hg_representation/benchmark.py';s=source.read_text()
    from torch.utils.cpp_extension import load
    from femlib.geometry import LOCAL_OFFSET
    from femlib.c3d8r import hourglass_basis
    from femlib.batched.c3d8r_hourglass_operator import BatchedC3D8RHourglassTwoStageOperator as HG
    OUT=source.parent
    exec(s[s.index('STENCIL=ASSEMBLY=None'):s.index('class Solver(')])
    extensions()

def build(data,lo,hi):
    solver=setup(data,lo,hi,material,selected,packed,fft)
    if taps:
        solver.op.S_hg=assemble(solver.op,taps);solver.op.__class__=HGStencil
    return solver

def validate():
    grid=(4,5,6);n=int(np.prod(grid));rng=np.random.default_rng(82173)
    Q=np.linalg.qr(rng.normal(size=(n,3,3)))[0];mask=rng.random(n)>.5
    data=dict(grid_shape=grid,domain_size=np.array([[.72,.85,.36]]),
       yarn_id=np.where(mask,0,-1).reshape(1,*grid),orient1=Q[:,:,0].reshape(1,*grid,3),orient2=Q[:,:,1].reshape(1,*grid,3))
    m,y=material['matrix'],material['yarn']
    cm=isotropic_to_voigt(m['E'],m['Nu']);cy=engineering_to_voigt(y['E11'],y['E22'],y['E33'],y['Nu12'],y['Nu13'],y['Nu23'],y['G12'],y['G13'],y['G23'])
    C=np.stack([rotate(cy,[0,0,0],Q[i])[0] if mask[i] else cm for i in range(n)])
    solver=build(data,0,1);op=solver.op
    value=op.C.cpu().numpy()[0];target=np.stack([C[:,i,j] for i,j in PAIRS],-1) if packed else C.reshape(n,36)
    errC=np.linalg.norm(value-target)/np.linalg.norm(target);assert errC<1e-12
    K=assembled(C,np.zeros((n,6)),grid,(.12,.17,.09),return_matrix=True)
    u=rng.normal(size=(3*n));action=op.apply(torch.tensor(u[None],device='cuda')).cpu().numpy()[0]
    errK=np.linalg.norm(action-K@u)/np.linalg.norm(K@u);assert errK<1e-12
    # Discrete reference inverse independently assembled in real space.
    c0=op._fft_args['C0_voigt'][0]
    K0=assembled(np.broadcast_to(c0,(n,6,6)),np.zeros((n,6)),grid,(.12,.17,.09),return_matrix=True)
    rhs=rng.normal(size=(n,3));rhs-=rhs.mean(0);rhs=rhs.ravel()
    x=np.zeros(3*n);x[3:]=spsolve(K0[3:,3:],rhs[3:]);x=(x.reshape(n,3)-x.reshape(n,3).mean(0)).ravel()
    inv=op.apply_precond(torch.tensor(rhs[None],device='cuda')).cpu().numpy()[0]
    errF=np.linalg.norm(inv-x)/np.linalg.norm(x);assert errF<1e-12
    ch,_,reports=elastic_basis(op);ref=assembled(C,np.zeros((n,6)),grid,(.12,.17,.09))[0]
    errH=np.linalg.norm(ch[0]-ref)/np.linalg.norm(ref);assert errH<1e-8
    return dict(material_relative_error=float(errC),action_relative_error=float(errK),
        inverse_relative_error=float(errF),CH_relative_error=float(errH),reports=reports)

def workload(data):
    rows=[];torch.cuda.empty_cache();synchronize();torch.cuda.reset_peak_memory_stats()
    total=0.;stage=dict(material=0.,setup=0.,solve=0.,cleanup=0.);peaks=[]
    for lo in range(0,len(data['indices']),B):
        gpu_guard();hi=min(lo+B,len(data['indices']));start=time.perf_counter()
        phase=[]
        if a.profile:synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        C=construct(data,lo,hi,material,selected,packed)
        if a.profile:
            synchronize();phase.append(dict(name='material',seconds=time.perf_counter()-start,peak=torch.cuda.max_memory_allocated(),live=torch.cuda.memory_allocated()));torch.cuda.reset_peak_memory_stats()
        tmat=time.perf_counter()
        nz,ny,nx=data['grid_shape'];voxel=data['domain_size'][lo]/np.array([nx,ny,nz])
        solver=make_solver(C,data['grid_shape'],voxel,fft=False);op=solver.op;del C
        if a.profile:
            synchronize();phase.append(dict(name='operator',seconds=time.perf_counter()-tmat,peak=torch.cuda.max_memory_allocated(),live=torch.cuda.memory_allocated()));torch.cuda.reset_peak_memory_stats()
        tfft=time.perf_counter()
        op._pack_real_half_fft_symbol=(fft==2)
        op.enable_fft_precond(ref_strategy='zeman',spectrum_mode='complex_full' if fft==0 else 'real_half',validation_mode='vectorized_hermitian',allow_symbol_sharing=False)
        if taps:op.S_hg=assemble(op,taps);op.__class__=HGStencil
        if a.profile:
            synchronize();phase.append(dict(name='fft',seconds=time.perf_counter()-tfft,peak=torch.cuda.max_memory_allocated(),live=torch.cuda.memory_allocated()));torch.cuda.reset_peak_memory_stats()
        tsetup=time.perf_counter();ch=np.empty((hi-lo,6,6));calls=[]
        for j in range(6):
            e=np.eye(6)[j];rhs=op.project_zero_mean(op.compute_rhs(e));u,r=solve_checked(op,rhs)
            # Same extra original-operator residual for all representations/configurations.
            from femlib.batched.c3d8r_hourglass_operator import BatchedC3D8RHourglassTwoStageOperator as HGBase
            rr=torch.linalg.vector_norm(rhs-HGBase.apply_cuda(op,u,block_size=128,variant='thread'),dim=1)/torch.linalg.vector_norm(rhs,dim=1).clamp_min(1e-300)
            if float(rr.max())>1e-10:raise RuntimeError('Original-operator residual outside tolerance')
            r['original_true_residuals']=rr.cpu().tolist();calls.append(r);ch[:,:,j]=op.compute_avg_stress(u,e)
        synchronize();tsolve=time.perf_counter()
        if a.profile:phase.append(dict(name='six_loads',seconds=tsolve-tsetup,peak=torch.cuda.max_memory_allocated(),live=torch.cuda.memory_allocated()))
        rows.append(dict(indices=data['indices'][lo:hi].tolist(),C_H_raw=ch.tolist(),calls=calls,
              phase_profile=phase,
              symbol_bytes=op._fft_symbol_inv.numel()*op._fft_symbol_inv.element_size(),material_bytes=op.C.numel()*op.C.element_size()))
        del solver,op,rhs,u,rr;torch.cuda.empty_cache();synchronize();end=time.perf_counter()
        total+=end-start
        for k,v in zip(stage,[tmat-start,tsetup-tmat,tsolve-tsetup,end-tsolve]):stage[k]+=v
    gpu_guard()
    return dict(seconds=total,stage_host_intervals_not_synchronized=stage,rows=rows,
       peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

if a.out.exists():raise FileExistsError(a.out)
validation=validate();print('VALIDATED',a.config,validation['material_relative_error'],validation['inverse_relative_error'],flush=True)
gc.collect();torch.cuda.empty_cache()
if a.validate_only:
    save_json(a.out,dict(status='passed',validation=validation));raise SystemExit(0)
data,input_audit=unique_dataset(read_dataset(a.data))
workload(data) # One complete unique-cell warmup, outside timed result.
gc.collect();torch.cuda.empty_cache();r=workload(data)
save_json(a.out,dict(status='passed',config=a.config,repeat=a.repeat,material=material,material_name=a.material,
   configuration=dict(selected=selected,packed=packed,fft=fft,B=B,taps=taps),validation=validation,
   measurement=r,provenance=provenance(),dataset_sha256=data['sha256'],input_audit=input_audit,diagnostic_profile=a.profile,
   protocol='fresh process per repetition; complete 64-unique-cell warmup; all six loads; raw unsymmetrized CH; same extra original-operator true residual; batch cleanup included; disk/JIT and GPU occupancy checks excluded; host stage intervals are not device stage timings'))
print('COMPLETE',a.config,a.material,a.repeat,r['seconds'],flush=True)
