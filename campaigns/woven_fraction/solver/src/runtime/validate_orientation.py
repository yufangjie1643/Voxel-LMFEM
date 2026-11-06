"""Independent assembled and original native HG validation of exact rotation."""
import benchmark as b
from benchmark import *
from verify_thermal import rotate,assembled
from orientation_adapter import attach,extension

def main():
    gpu_guard();torch.set_num_threads(1);extension();extensions()
    rng=np.random.default_rng(101616);grid=(4,5,6);n=int(np.prod(grid))
    Q=np.linalg.qr(rng.normal(size=(n,3,3)))[0];mask=rng.random(n)>.5
    data=dict(grid_shape=grid,domain_size=np.array([[.72,.85,.36]]),yarn_id=np.where(mask,0,-1).reshape(1,*grid),orient1=Q[:,:,0].reshape(1,*grid,3),orient2=Q[:,:,1].reshape(1,*grid,3))
    m,y=STRONG['matrix'],STRONG['yarn'];cm=isotropic_to_voigt(m['E'],m['Nu']);cy=engineering_to_voigt(y['E11'],y['E22'],y['E33'],y['Nu12'],y['Nu13'],y['Nu23'],y['G12'],y['G13'],y['G23'])
    C=np.stack([rotate(cy,[0,0,0],Q[i])[0] if mask[i] else cm for i in range(n)])
    cc,aa,uc,sc=assembled(C,np.zeros((n,6)),grid,(.12,.17,.09),return_fields=True)
    K=assembled(C,np.zeros((n,6)),grid,(.12,.17,.09),return_matrix=True)
    v=rng.normal(size=(1,3*n));u=torch.tensor(v,device='cuda');reference=K@v[0]
    records=[]
    for route in ['direct','tap14','matched21','orientation']:
      for precision in ['fp64','mixed']:
        op=build(data,STRONG,route,precision,Stages());action=op.apply(u).cpu().numpy()[0]
        err=np.linalg.norm(action-reference)/np.linalg.norm(reference);assert err<1e-12,(route,err)
        ch=[];fields=[];residuals=[]
        for j in range(6):
            e=np.eye(6)[j];rhs=op.project_zero_mean(op.compute_rhs(e));sol,r=solve(op,rhs)
            ch.append(op.compute_avg_stress(sol,e)[0]);fields.append(op.recover_center_fields(sol,e)[1].cpu().numpy()[0]);residuals+=r['true_residuals']
        ch=np.array(ch).T;fields=np.stack(fields,-1)
        herror=np.linalg.norm(ch-cc)/np.linalg.norm(cc);serror=np.linalg.norm(fields-sc[:,:,:6])/np.linalg.norm(sc[:,:,:6])
        assert herror<max(1e-9,20*TOL) and serror<max(1e-8,100*TOL) and max(residuals)<=TOL
        records.append(dict(route=route,precision=precision,action_error=err,CH_error=herror,stress_error=serror,true_residual_max=max(residuals)))
        del op,sol,rhs;cleanup()
    save_json(ROOT/'results/validation_orientation.json',dict(status='passed',tolerance=TOL,grid=grid,seed=101616,records=records))
    print(json.dumps(records,indent=2),flush=True)
if __name__=='__main__':main()
