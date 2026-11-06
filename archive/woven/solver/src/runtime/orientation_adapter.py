"""Exact phase/frame constitutive action, matched node-gather C21 control.

The current setup uses the same temporary C21 constructor/reference builder,
then releases C21 for the orientation route. Report setup peak separately;
this is not a claim of a minimum-memory phase/orientation implementation.
"""
from pathlib import Path
from functools import lru_cache
from physics import *
from femlib.batched.c3d8r_hourglass_operator import BatchedC3D8RHourglassTwoStageOperator as HG
from femlib.materials import engineering_to_voigt,isotropic_to_voigt
from femlib.c3d8r import hourglass_basis

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='fead_orientation_20261016',sources=[str(Path(__file__).with_name('orientation_kernels.cu'))],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],verbose=False)

class OrientationHG(HG):
    def _run(self,u,E,mode):
        return extension().run(self.C,self._rotation,self._phase,self._local,self.B_center,self._hh,self.hg_scale,u,E,self.mesh.Nz,self.mesh.Ny,self.mesh.Nx,self.mesh.V_elem,self._oriented,mode)
    def apply(self,u):return self._run(u,self._zero,0)
    def compute_rhs(self,E):return self._run(self._zero,self._macro_strain_tensor(E).reshape(-1),1)
    def compute_avg_stress(self,u,E):return self._run(u,self._macro_strain_tensor(E).reshape(-1),2).mean(1).detach().cpu().numpy()
    def recover_center_fields(self,u,E):
        strain=torch.einsum('iap,beap->bei',self.B_center,self._gather_element_displacements(u))+self._macro_strain_tensor(E)
        return strain,self._run(u,self._macro_strain_tensor(E).reshape(-1),2)

def attach(op,data,material,route):
    if op.B!=1:raise ValueError('This controlled comparison uses B=1')
    op._oriented=route=='orientation'
    if op._oriented:
        R,mask=rotation_fields(data,0,1,op.device)
        op._rotation=R.contiguous();op._phase=mask.to(torch.int8).contiguous()
    else:
        op._rotation=torch.empty(0,device=op.device,dtype=op.dtype)
        op._phase=torch.empty(0,device=op.device,dtype=torch.int8)
    m,y=material['matrix'],material['yarn']
    cm=isotropic_to_voigt(m['E'],m['Nu'])
    cy=engineering_to_voigt(y['E11'],y['E22'],y['E33'],y['Nu12'],y['Nu13'],y['Nu23'],y['G12'],y['G13'],y['G23'])
    op._local=torch.tensor(np.stack([cm,cy]),device=op.device,dtype=op.dtype)
    H=hourglass_basis().reshape(8,3,12)[:,0,:]
    op._hh=torch.tensor(H@H.T,device=op.device,dtype=op.dtype)
    op._zero=torch.zeros(6,device=op.device,dtype=op.dtype)
    if op._oriented:
        op.C=torch.empty(0,device=op.device,dtype=op.dtype);op._C_voigt_gpu=op.C
    op.__class__=OrientationHG
