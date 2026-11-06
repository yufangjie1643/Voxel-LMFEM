"""Independent eight-point CPU assembly for the new anisotropic FI paths."""
import fi_benchmark as f
from benchmark import *
from verify_thermal import rotate
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def main():
    gpu_guard();torch.set_num_threads(1)
    rng=np.random.default_rng(101616);grid=(4,5,6);nz,ny,nx=grid;n=int(np.prod(grid));voxel=np.array([.12,.17,.09])
    Q=np.linalg.qr(rng.normal(size=(n,3,3)))[0];mask=rng.random(n)>.5
    data=dict(grid_shape=grid,domain_size=np.array([[.72,.85,.36]]),yarn_id=np.where(mask,0,-1).reshape(1,*grid),orient1=Q[:,:,0].reshape(1,*grid,3),orient2=Q[:,:,1].reshape(1,*grid,3))
    m,y=STRONG['matrix'],STRONG['yarn'];cm=isotropic_to_voigt(m['E'],m['Nu']);cy=engineering_to_voigt(y['E11'],y['E22'],y['E33'],y['Nu12'],y['Nu13'],y['Nu23'],y['G12'],y['G13'],y['G23'])
    C=np.stack([rotate(cy,[0,0,0],Q[i])[0] if mask[i] else cm for i in range(n)])
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    signs=2*corners-1;Bs=[]
    for idx in np.ndindex(2,2,2):
        q=(2*np.array(idx)-1)/np.sqrt(3);B=np.zeros((6,24))
        for a,s in enumerate(signs):
            grad=np.array([s[k]*np.prod([1+s[l]*q[l] for l in range(3) if l!=k])/(4*voxel[k]) for k in range(3)])
            dx,dy,dz=grad;B[0,3*a]=dx;B[1,3*a+1]=dy;B[2,3*a+2]=dz
            B[3,3*a+1]=dz;B[3,3*a+2]=dy;B[4,3*a]=dz;B[4,3*a+2]=dx;B[5,3*a]=dy;B[5,3*a+1]=dx
        Bs.append(B)
    Bs=np.array(Bs);weight=np.prod(voxel)/8;rows=[];cols=[];values=[];connect=[];rhs=np.zeros((3*n,6))
    for z,y,x in np.ndindex(grid):
        e=(z*ny+y)*nx+x;ids=((z+corners[:,2])%nz*ny+(y+corners[:,1])%ny)*nx+(x+corners[:,0])%nx
        dofs=(3*ids[:,None]+np.arange(3)).ravel();connect.append(dofs)
        ke=sum(weight*B.T@C[e]@B for B in Bs)
        rows.extend(np.repeat(dofs,24));cols.extend(np.tile(dofs,24));values.extend(ke.ravel())
        np.add.at(rhs,dofs,-weight*Bs.sum(0).T@C[e])
    K=sp.coo_matrix((values,(rows,cols)),shape=(3*n,3*n)).tocsr();uc=np.zeros_like(rhs);uc[3:]=spsolve(K[3:,3:],rhs[3:]);uc=(uc.reshape(n,3,6)-uc.reshape(n,3,6).mean(0)).reshape(3*n,6)
    def stress(u):return np.array([np.array([C[e]@(B@u[d]+np.eye(6)) for B in Bs])for e,d in enumerate(connect)])
    sc=stress(uc);cc=sc.mean((0,1));v=rng.normal(size=(1,3*n));ref=K@v[0];records=[]
    for route in ['direct','tap14']:
      for precision in ['fp64','mixed']:
        op=f.build(data,STRONG,route,precision,Stages());action=op.apply(torch.tensor(v,device='cuda')).cpu().numpy()[0]
        ae=float(np.linalg.norm(action-ref)/np.linalg.norm(ref));ch=[];ug=[];residuals=[]
        for j in range(6):
            e=np.eye(6)[j];r=op.project_zero_mean(op.compute_rhs(e));sol,info=solve(op,r)
            ch.append(op.compute_avg_stress(sol,e)[0]);ug.append(sol.cpu().numpy()[0]);residuals+=info['true_residuals']
        ce=float(np.linalg.norm(np.array(ch).T-cc)/np.linalg.norm(cc));se=float(np.linalg.norm(stress(np.array(ug).T)-sc)/np.linalg.norm(sc))
        assert ae<1e-12 and ce<max(1e-9,20*TOL) and se<max(1e-8,100*TOL) and max(residuals)<=TOL,(route,precision,ae,ce,se)
        records.append(dict(route=route,precision=precision,action_error=ae,CH_error=ce,stress_error=se,true_residual_max=max(residuals)))
        del op,sol,r;cleanup()
    save_json(ROOT/'results/validation_fi.json',dict(status='passed',tolerance=TOL,grid=grid,voxel=voxel.tolist(),seed=101616,reference='Independent CPU eight-point assembly; local stress at all Gauss points reconstructed from GPU displacement',records=records))
    print(json.dumps(records,indent=2),flush=True)
if __name__=='__main__':main()
