"""Independent CPU criterion, local-field, superposition and onset checks."""
import argparse
from failure import *
from verify_thermal import assembled,rotate
from femlib.materials import isotropic_to_voigt,engineering_to_voigt


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    gpu_guard();torch.set_num_threads(1);rng=np.random.default_rng(61006)
    # Arbitrary full 6x6 localizations, non-round tile size, both constituent masks.
    s=rng.normal(size=(2,259,6,6))*20;mask=rng.random((2,259))>.5
    d=rng.normal(size=(33,6));d/=np.linalg.norm(d,axis=1)[:,None]
    lam,_,_=envelope(torch.tensor(s,device='cuda'),torch.tensor(mask,device='cuda'),d)
    ref=np.stack([1/numpy_inverse(np.einsum('nik,dk->dni',s[b],d),mask[b]).max(-1) for b in range(2)])
    kernel_error=float(np.max(np.abs(lam/ref-1)));assert kernel_error<1e-12
    # Uniaxial strengths and resin shear implied by the declared paraboloid.
    canonical=[]
    for is_yarn in [True,False]:
        for j,sign in [(0,1),(0,-1),(1,1),(1,-1),(3,1),(4,1),(5,1)]:
            state=np.zeros((1,6));state[0,j]=sign
            val=1/numpy_inverse(state,np.array([is_yarn]))[0]
            if is_yarn:expected=(FAILURE['yarn_tension'] if sign>0 else FAILURE['yarn_compression'])[j] if j<3 else FAILURE['yarn_shear'][j-3]
            else:expected=(FAILURE['matrix_tension'] if sign>0 else FAILURE['matrix_compression']) if j<3 else np.sqrt(FAILURE['matrix_tension']*FAILURE['matrix_compression']/3)
            assert abs(val/expected-1)<1e-12
            canonical.append(dict(yarn=is_yarn,component=j,sign=sign,strength=float(val)))
    # Independent small periodic assembly; elastic local fields, not only C_H.
    grid=(4,5,6);n=int(np.prod(grid));voxel=(.12,.17,.09)
    m=FAILURE['matrix'];y=FAILURE['yarn']
    cm=isotropic_to_voigt(m['E'],m['Nu'])
    cy=engineering_to_voigt(y['E11'],y['E22'],y['E33'],y['Nu12'],y['Nu13'],y['Nu23'],y['G12'],y['G13'],y['G23'])
    rotations=np.linalg.qr(rng.normal(size=(n,3,3)))[0]
    C=np.stack([rotate(cy,[0,0,0],rotations[i])[0] if i%3 else cm for i in range(n)])
    ch_cpu,ah_cpu,u_cpu,stress_cpu=assembled(C,np.zeros((n,6)),grid,voxel,return_fields=True)
    packed=torch.tensor(np.stack([C[:,i,j] for i,j in PAIRS],axis=-1)[None],device='cuda')
    solver=make_solver(packed,grid,voxel);op=solver.op
    ch_gpu,stress_gpu,reports=elastic_basis(op,retain='stress')
    field_error=float(np.linalg.norm(stress_gpu.cpu().numpy()[0]-stress_cpu[:,:,:6])/np.linalg.norm(stress_cpu[:,:,:6]));assert field_error<1e-8
    del solver,op,packed,stress_gpu;torch.cuda.empty_cache()
    # Actual woven cell: six-basis reconstruction vs directly solved mixed stress.
    data=read_dataset(a.data,[0]);solver=build_batch(data,0,1,FAILURE);op=solver.op
    R,mask=rotation_fields(data,0,1,op.device);ch,S,reports=local_stress_basis(op,R)
    checks=[]
    for direction in d[:6]:
        strain=np.linalg.solve(ch[0],direction)
        u,report=solve_checked(op,op.project_zero_mean(op.compute_rhs(strain)))
        direct=rotate_stress_local(op.recover_center_fields(u,strain)[1],R)
        recon=torch.einsum('bnij,j->bni',S,S.new_tensor(direction))
        error=float(torch.linalg.vector_norm(direct-recon)/torch.linalg.vector_norm(direct))
        ref_direct=1/numpy_inverse(direct.cpu().numpy()[0],mask.cpu().numpy()[0]).max()
        onset,mode,_=envelope(S,mask,direction[None])
        rel=abs(float(onset[0,0])/ref_direct-1)
        # Scale recovered stresses to the computed onset; criterion must equal 1.
        state=direct.cpu().numpy()[0]*onset[0,0];is_yarn=mask.cpu().numpy()[0]
        inv=numpy_inverse(state,is_yarn);closure=float(abs(inv.max()-1))
        assert error<1e-8 and rel<1e-8 and closure<1e-8
        checks.append(dict(direction=direction.tolist(),field_relative_error=error,
                          onset_relative_error=rel,onset_MPa=float(onset[0,0]),
                          criterion_closure_error=closure,solve=report))
    save_json(a.out,dict(status='passed',material=FAILURE,provenance=provenance(),
       independent_criterion_relative_error=kernel_error,canonical=canonical,
       independent_assembled_field_relative_error=field_error,combined_load_checks=checks,
       six_basis_solves=reports,dataset_sha256=data['sha256']))
    print('FAILURE VALIDATION PASSED',kernel_error,field_error,flush=True)

if __name__=='__main__':main()
