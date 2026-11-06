"""Independent assembled CPU and analytical checks of the new thermal load."""
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from physics import (torch, PAIRS, VOIGT, THERMAL, make_solver, elastic_basis,
                     thermal_response, gpu_guard, save_json, provenance)
from femlib.materials import isotropic_to_voigt, engineering_to_voigt


def analytical_layer(C, alpha, fractions):
    """Exact bonded laminate, layers normal to z, free overall expansion."""
    p, t = [0,1,5], [2,3,4]
    blocks, vectors, invs = [], [], []
    for c, a in zip(C, alpha):
        inv = np.linalg.inv(c[np.ix_(t,t)])
        invs.append(inv)
        ca = c@a
        blocks.append(c[np.ix_(p,p)]-c[np.ix_(p,t)]@inv@c[np.ix_(t,p)])
        vectors.append(ca[p]-c[np.ix_(p,t)]@inv@ca[t])
    ep = np.linalg.solve(np.einsum('s,sij->ij',fractions,blocks),
                         np.einsum('s,si->i',fractions,vectors))
    result = np.zeros(6)
    result[p] = ep
    result[t] = sum(w*inv@((c@a)[t]-c[np.ix_(t,p)]@ep)
                    for w,c,a,inv in zip(fractions,C,alpha,invs))
    return result


def assembled(C, alpha, grid, voxel, hg=.1, return_fields=False, return_matrix=False):
    """Explicit CPU assembly with independent shape-gradient/connectivity code."""
    nz,ny,nx = grid
    n = nx*ny*nz
    corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                        [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    signs = 2*corners-1
    b = np.zeros((6,24))
    for a, s in enumerate(signs):
        dx,dy,dz = s/(4*np.asarray(voxel))
        b[0,3*a]=dx; b[1,3*a+1]=dy; b[2,3*a+2]=dz
        b[3,3*a+1]=dz; b[3,3*a+2]=dy
        b[4,3*a]=dz; b[4,3*a+2]=dx
        b[5,3*a]=dy; b[5,3*a+1]=dx
    modes = np.stack([signs[:,0]*signs[:,1],signs[:,1]*signs[:,2],
                      signs[:,2]*signs[:,0],np.prod(signs,axis=1)],axis=1)/np.sqrt(8)
    h = np.zeros((24,12))
    for component in range(3): h[component::3,component*4:component*4+4] = modes
    volume = np.prod(voxel)
    rows,cols,vals = [],[],[]
    rhs = np.zeros((3*n,7))
    connectivity=[]
    for z,y,x in np.ndindex(grid):
        e=(z*ny+y)*nx+x
        ids=((z+corners[:,2])%nz*ny+(y+corners[:,1])%ny)*nx+(x+corners[:,0])%nx
        dofs=(ids[:,None]*3+np.arange(3)).ravel()
        connectivity.append(dofs)
        c=C[e]
        normal=np.trace(c[:3,:3])
        inv1=c[:3,:3].sum()
        inv2=normal+2*np.trace(c[3:,3:])
        scale=hg*(inv1+7*inv2)/60*volume/np.mean(voxel)**2
        k=volume*b.T@c@b+scale*h@h.T
        rows.extend(np.repeat(dofs,24)); cols.extend(np.tile(dofs,24)); vals.extend(k.ravel())
        np.add.at(rhs,dofs,np.column_stack([-volume*b.T@c,volume*b.T@c@alpha[e]]))
    K=sp.coo_matrix((vals,(rows,cols)),shape=(3*n,3*n)).tocsr()
    if return_matrix:return K
    u=np.zeros_like(rhs)
    u[3:]=spsolve(K[3:,3:],rhs[3:])
    u=(u.reshape(n,3,7)-u.reshape(n,3,7).mean(0)).reshape(3*n,7)
    stress=np.empty((n,6,7))
    for e,dofs in enumerate(connectivity):
        strain=b@u[dofs]
        strain[:,:6]+=np.eye(6)
        strain[:,6]-=alpha[e]
        stress[e]=C[e]@strain
    average=stress.mean(0)
    ch=average[:,:6]
    ah=np.linalg.solve(ch,-average[:,6])
    relative=np.linalg.norm(K@u-rhs,axis=0)/np.maximum(np.linalg.norm(rhs,axis=0),1e-30)
    return (ch,ah,u,stress) if return_fields else (ch,ah,u,relative)


def rotate(c, a, r):
    # Explicit fourth-order action by six unit global engineering strains.
    result=np.empty((6,6))
    for j in range(6):
        strain=np.zeros((3,3)); k,l=VOIGT[j]
        strain[k,l]=strain[l,k]=1 if k==l else .5
        el=r.T@strain@r
        ev=np.array([el[k,l]*(1 if k==l else 2) for k,l in VOIGT])
        sv=c@ev
        sl=np.zeros((3,3))
        for val,(k,l) in zip(sv,VOIGT):sl[k,l]=sl[l,k]=val
        sg=r@sl@r.T
        result[:,j]=[sg[k,l] for k,l in VOIGT]
    ag=r@np.diag(a)@r.T
    av=np.array([ag[k,l]*(1 if k==l else 2) for k,l in VOIGT])
    return result,av


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();gpu_guard();torch.set_num_threads(1)
    material=THERMAL; y=material['yarn']
    cm=isotropic_to_voigt(**{'E':material['matrix']['E'],'nu':material['matrix']['Nu']})
    cy=engineering_to_voigt(y['E11'],y['E22'],y['E33'],y['Nu12'],y['Nu13'],y['Nu23'],y['G12'],y['G13'],y['G23'])
    ay=np.array(material['alpha_yarn']);am=np.array([material['alpha_matrix']]*3+[0]*3)
    angle=.37
    r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    cr,ar=rotate(cy,ay,r)
    grid=(4,5,6);voxel=(.12,.17,.09);n=np.prod(grid)
    patterns=[]
    patterns.append(('homogeneous_rotated',np.repeat(cr[None],n,0),np.repeat(ar[None],n,0),ar))
    for fraction in [.25,.5,.75]:
        layers=np.indices(grid)[0].ravel()<round(grid[0]*fraction)
        cf=np.where(layers[:,None,None],cr,cm)
        af=np.where(layers[:,None],ar,am)
        expected=analytical_layer([cr,cm],[ar,am],[fraction,1-fraction])
        patterns.append((f'laminate_{fraction}',cf,af,expected))
    rng=np.random.default_rng(61006)
    cf=np.empty((n,6,6));af=np.empty((n,6))
    for i in range(n):
        if i%3==0:cf[i]=cm;af[i]=am
        else:
            q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q)
            cf[i],af[i]=rotate(cy,ay,q)
    patterns.append(('spatially_rotated',cf,af,None))
    results=[]
    for name,cf,af,expected in patterns:
        gpu_guard()
        packed=torch.tensor(np.stack([cf[:,i,j] for i,j in PAIRS],axis=-1)[None],device='cuda')
        solver=make_solver(packed,grid,voxel)
        ch,_,loads=elastic_basis(solver.op)
        ah,ut,avg,thermal=thermal_response(solver.op,ch,torch.tensor(af[None],device='cuda'))
        cc,aa,uu,cpu_res=assembled(cf,af,grid,voxel)
        row=dict(name=name,grid=list(grid),voxel=list(voxel),alpha_H=ah[0].tolist(),
                 alpha_H_assembled=aa.tolist(),elastic_loads=loads,thermal_load=thermal,
                 C_relative_error=float(np.linalg.norm(ch[0]-cc)/np.linalg.norm(cc)),
                 alpha_relative_error=float(np.linalg.norm(ah[0]-aa)/np.linalg.norm(aa)),
                 thermal_field_relative_error=float(np.linalg.norm(ut.cpu().numpy()[0]-uu[:,6])/max(np.linalg.norm(uu[:,6]),1e-30)))
        if expected is not None:
            row['alpha_analytical']=expected.tolist()
            row['analytical_relative_error']=float(np.linalg.norm(ah[0]-expected)/np.linalg.norm(expected))
            assert row['analytical_relative_error']<1e-8,row
        assert row['alpha_relative_error']<1e-8,row
        assert row['C_relative_error']<1e-8,row
        if name!='homogeneous_rotated':assert row['thermal_field_relative_error']<1e-8,row
        results.append(row);print(name,row['alpha_relative_error'],flush=True)
        del solver,packed;torch.cuda.empty_cache()
    save_json(args.out,dict(status='passed',material=material,results=results,provenance=provenance()))


if __name__=='__main__':main()
