"""Linear first-constituent-failure screening, not progressive damage.

Yarn-scale constants/strengths: Cao et al., Composite Structures 211 (2019)
175-186, Tables 4-5, carbon-yarn column (all three shear strengths explicit).
Resin constants/strengths: Zhou et al., Aerospace 9 (2022) 537, Tables 8,11.
This literature-defined numerical material combines two sources. It is not a
reconstruction of either paper's manufactured composite or damage model.
"""
from physics import *
from torch.utils.cpp_extension import load

FAILURE = {
    'yarn_source': 'https://doi.org/10.1016/j.compstruct.2018.12.024',
    'yarn_tables': [4,5],
    'matrix_source': 'https://doi.org/10.3390/aerospace9100537',
    'matrix_tables': [8,11],
    'matrix': dict(E=3350.,Nu=.35),
    'yarn': dict(E11=169490.,E22=10180.,E33=10180.,G12=5320.,G13=5320.,G23=3780.,
                 Nu12=.28,Nu13=.28,Nu23=.41),
    'yarn_tension': [2150.,62.3,62.3], 'yarn_compression': [2150.,199.8,199.8],
    'yarn_shear': [81.5,81.5,81.5], 'matrix_tension':126.6, 'matrix_compression':238.3,
    'yarn_criterion':'maximum stress in yarn axes, all six components',
    'matrix_criterion':'von-Mises-squared/(T*C)+(1/T-1/C)*trace(sigma)=1',
    'interpretation':'linear-elastic first-constituent-failure screening at fixed discretization',
}


def directions(count=360):
    theta=np.arange(count)*2*np.pi/count
    d=[]
    for p,q in [(0,1),(1,5)]:
        v=np.zeros((count,6));v[:,p]=np.cos(theta);v[:,q]=np.sin(theta);d.append(v)
    return np.concatenate(d)


def local_stress_basis(op,R):
    C=np.empty((op.B,6,6));saved=[];reports=[]
    for j in range(6):
        e=np.eye(6)[j]
        u,r=solve_checked(op,op.project_zero_mean(op.compute_rhs(e)))
        C[:,:,j]=op.compute_avg_stress(u,e)
        stress=op.recover_center_fields(u,e)[1]
        saved.append(rotate_stress_local(stress,R));reports.append(r)
    basis=torch.stack(saved,-1)
    inv=torch.as_tensor(np.linalg.inv(C),device=op.device,dtype=torch.float64)
    # Each cell uses its own complete, unsymmetrized stiffness matrix.
    return C,torch.einsum('bnij,bjk->bnik',basis,inv).contiguous(),reports


_extension=None
def extension():
    global _extension
    if _extension is None:
        p=Path(__file__).parent
        _extension=load(name='fead_initial_failure_fp64_v1',sources=[str(p/'failure.cpp'),str(p/'failure.cu')],
                        extra_cuda_cflags=['-O3'],verbose=False)
    return _extension


def envelope(basis,mask,d):
    props=basis.new_tensor(FAILURE['yarn_tension']+FAILURE['yarn_compression']+
                          FAILURE['yarn_shear']+[FAILURE['matrix_tension'],FAILURE['matrix_compression']])
    direction=torch.as_tensor(d,device=basis.device,dtype=torch.float64).contiguous()
    partial=extension().evaluate(basis,mask.contiguous(),direction,props)
    modes=partial.amax(-1) # B,D,2: yarn and matrix maximum inverse load factors.
    inv,mode=modes.max(-1)
    if not bool(torch.isfinite(inv).all()) or bool((inv<=0).any()):
        raise RuntimeError('Unbounded or invalid requested ray')
    return (1/inv).cpu().numpy(),mode.cpu().numpy(),modes.cpu().numpy()


def numpy_inverse(stress,mask):
    """Independent NumPy reference, stress (...,N,6); returns (...,N)."""
    s=np.asarray(stress);m=np.asarray(mask)
    normal=np.where(s[...,:3]>=0,s[...,:3]/FAILURE['yarn_tension'],
                    -s[...,:3]/FAILURE['yarn_compression'])
    shear=np.abs(s[...,3:])/FAILURE['yarn_shear']
    yarn=np.maximum(normal.max(-1),shear.max(-1))
    vm2=.5*((s[...,0]-s[...,1])**2+(s[...,1]-s[...,2])**2+(s[...,2]-s[...,0])**2)+3*(s[...,3:]**2).sum(-1)
    T,C=FAILURE['matrix_tension'],FAILURE['matrix_compression']
    a=vm2/(T*C);b=(1/T-1/C)*s[...,:3].sum(-1)
    # Rationalized expression avoids cancellation under hydrostatic compression.
    root=np.sqrt(b*b+4*a)
    resin=np.where(b>=0,.5*(root+b),2*a/np.maximum(root-b,1e-300))
    return np.where(m,yarn,resin)
