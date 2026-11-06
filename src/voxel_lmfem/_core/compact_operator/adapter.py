"""Isolated full-workflow ablations; prior experiment sources remain frozen."""
from functools import lru_cache
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch
from scipy.linalg import eigh

HERE=Path(__file__).resolve().parent
FORMULA=Path(os.environ.get("FEAD_FORMULA",str(__import__("project_paths").source_path('reference_medium'))))
STRUCTURAL=Path(os.environ.get("FEAD_STRUCTURAL",str(__import__("project_paths").source_path('constitutive_basis'))))
sys.path.insert(0,str(FORMULA))
import formula_study as prior

VARIANTS=("dense","constitutive","compact","projection","compact_projection","energy_recovery","energy_fixed","energy_certified")


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name="fead_structural_ablation_20261020_v1",sources=[str(HERE/"ablation_kernels.cu")],
        extra_include_paths=[str(STRUCTURAL)],extra_cuda_cflags=["-O3","--ptxas-options=-v"],extra_cflags=["-O3"],verbose=False)


def matrices(material):
    from femlib.materials import engineering_to_voigt,isotropic_to_voigt
    y=material["yarn"];m=material["matrix"]
    cy=engineering_to_voigt(*(y[k] for k in ("E11","E22","E33","Nu12","Nu13","Nu23","G12","G13","G23")))
    return isotropic_to_voigt(m["E"],m["Nu"]),cy


def attach(op,data,material,variant):
    from femlib.c3d8r import hourglass_scale
    cm,cy=matrices(material);h=op._modal_voxel
    op._ablation_variant=variant
    if variant=="dense":return op
    if variant.startswith("energy"):
        packed_mean=op.C[0].mean(dim=0).cpu().numpy();mean=np.zeros((6,6))
        for value,(i,j) in zip(packed_mean,[(i,j) for i in range(6) for j in range(i,6)]):mean[i,j]=mean[j,i]=value
        op._ablation_Cmean=mean
    if variant=="projection":
        ei=torch.empty(0,device=op.device,dtype=torch.int32);ed=torch.empty(0,device=op.device,dtype=torch.float64)
        def project(self,u,energy):
            value,partial=extension().apply(u.contiguous(),self.C,self.hg_scale,ei,ed,ed,*prior.formula.geometry(self),0,energy)
            return (value,partial.sum().reshape(1)) if energy else value
        op.apply=MethodType(lambda self,u:project(self,u,False),op)
        op.apply_with_energy=MethodType(lambda self,u:project(self,u,True),op)
        return op
    lam=cy[1,2];mu=(cy[1,1]-cy[1,2])/2;alpha=cy[0,1]-lam;gamma=cy[5,5]-mu
    beta=cy[0,0]-cy[1,1]-2*alpha-4*gamma;delta=cy[3,3]-mu
    active=np.flatnonzero(np.asarray(data["yarn_id"]).reshape(-1)>=0).astype(np.int32)
    a=torch.as_tensor(np.ascontiguousarray(data["orient1"].reshape(-1,3)[active]),device=op.device,dtype=torch.float64)
    b=torch.as_tensor(np.ascontiguousarray(data["orient2"].reshape(-1,3)[active]),device=op.device,dtype=torch.float64)
    a=a/torch.linalg.vector_norm(a,dim=-1,keepdim=True).clamp_min(1e-14)
    c=torch.linalg.cross(a,b);c=c/torch.linalg.vector_norm(c,dim=-1,keepdim=True).clamp_min(1e-14);b=torch.linalg.cross(c,a)
    transverse=torch.stack([b[:,i]*c[:,j]+c[:,i]*b[:,j] for i,j in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))],dim=-1)
    op._ablation_frame=torch.cat((a,transverse),dim=-1).contiguous()
    op._ablation_active=torch.as_tensor(active,device=op.device,dtype=torch.int32)
    idx=np.full(op.mesh.n_elem,-1,dtype=np.int32);idx[active]=np.arange(len(active),dtype=np.int32)
    op._ablation_index=torch.as_tensor(idx,device=op.device)
    op._ablation_params=torch.tensor([lam,mu,alpha,beta,gamma,delta,cm[0,1],cm[3,3],hourglass_scale(cm,h,.1),hourglass_scale(cy,h,.1)],device=op.device,dtype=torch.float64)
    op._ablation_background=cm
    mode=0 if variant=="projection" else 1 if variant=="compact_projection" else 2
    empty=torch.empty(0,device=op.device,dtype=torch.float64)
    def action(self,u,energy):
        values,partial=extension().apply(u.contiguous(),self.C if self.C is not None else empty,
            self.hg_scale if self.hg_scale is not None else empty,self._ablation_index,self._ablation_frame,self._ablation_params,
            *prior.formula.geometry(self),mode,energy)
        return (values,partial.sum().reshape(1)) if energy else values
    op.apply=MethodType(lambda self,u:action(self,u,False),op)
    op.apply_with_energy=MethodType(lambda self,u:action(self,u,True),op)
    if variant not in ("constitutive","projection"):
        def rhs(self,macro):
            e=torch.as_tensor(np.asarray(macro),device=self.device,dtype=torch.float64)
            return extension().rhs(self._ablation_active,self._ablation_frame,self._ablation_params,e,*prior.formula.geometry(self))
        def stress(self,u,macro):
            e=torch.as_tensor(np.asarray(macro),device=self.device,dtype=torch.float64)
            partial=extension().stress(self._ablation_active,self._ablation_frame,self._ablation_params,e,u.contiguous(),*prior.formula.geometry(self))
            return (partial.sum(dim=1).cpu().numpy()/self.mesh.n_elem+self._ablation_background@np.asarray(macro))[None,:]
        op.compute_rhs=MethodType(rhs,op);op.compute_avg_stress=MethodType(stress,op)
        op.C=None;op.hg_scale=None;op._C_voigt_gpu=None;op._modal_scales=None
    return op


def certificate_setup(op,material):
    from femlib.c3d8r import hourglass_scale
    cm,cy=matrices(material);c0=np.asarray(op._analytic_reference_C0);h=op._modal_voxel
    gamma0=hourglass_scale(c0,h,.1)
    a=min(float(eigh(cm,c0,eigvals_only=True)[0]),float(eigh(cy,c0,eigvals_only=True)[0]),
          hourglass_scale(cm,h,.1)/gamma0,hourglass_scale(cy,h,.1)/gamma0)
    weights=np.array([1.,1.,1.,np.sqrt(2.),np.sqrt(2.),np.sqrt(2.)])
    cmin=min(np.linalg.eigvalsh(cm*weights[:,None]*weights[None,:])[0],np.linalg.eigvalsh(cy*weights[:,None]*weights[None,:])[0])
    op._certificate_a=float(a);op._certificate_target=float(1e-8*np.sqrt(6)*cmin/6)
    op._certificate_tables=prior.formula.tables(op)
    op._certificate_coeff=prior.formula.extension().hg_coeff(*op._certificate_tables,*prior.formula.geometry(op),
        float(c0[0,1]),float(c0[3,3]),gamma0,float(c0[0,1]),float(c0[3,3]),gamma0)
    def exact_precond(self,r):
        shape=(self.mesh.Nz,self.mesh.Ny,self.mesh.Nx)
        field=r.view(1,-1,3).transpose(1,2).contiguous().view(1,3,*shape)
        spectrum=torch.fft.rfftn(field,dim=(-3,-2,-1)).contiguous()
        values=spectrum
        for _ in range(2):values=prior.formula.extension().project(self._certificate_coeff,*self._certificate_tables,values,*prior.formula.geometry(self),0)
        z=torch.fft.irfftn(values,s=shape,dim=(-3,-2,-1),norm="backward")
        return z.view(1,3,-1).transpose(1,2).contiguous().view_as(r)
    op.exact_precond=MethodType(exact_precond,op)


def _update(u,r,p,ap,rho,pap):
    alpha=torch.where(pap>0,rho/pap,torch.zeros_like(rho))[:,None]
    u=u+alpha*p;r=r-alpha*ap
    return u,r,(r*r).sum(dim=1)


def _direction(r,z,p,new,old):return z+torch.where(old>0,new/old,torch.zeros_like(new))[:,None]*p


@lru_cache(None)
def compiled_steps():
    opts={"triton.cudagraphs":False}
    return torch.compile(_update,fullgraph=True,options=opts),torch.compile(_direction,fullgraph=True,options=opts)


def custom_solve(op,rhs,j,certified):
    norm=torch.linalg.vector_norm(rhs)
    if float(norm)==0:return torch.zeros_like(rhs),torch.zeros_like(rhs),dict(iterations=0,restarts=0,true_residuals=[0.],bound=0.)
    update,direction=compiled_steps();u=torch.zeros_like(rhs);r=rhs.clone();z=op.apply_precond(r);p=z.clone();rho=(r*z).sum(dim=1)
    volume=op.mesh.n_elem*op.mesh.V_elem;w2=1. if j<3 else 2.;restarts=0
    for it in range(1,10001):
        ap,pap=op.apply_with_energy(p);u,r,rr=update(u,r,p,ap,rho,pap)
        z=op.apply_precond(r);new=(r*z).sum(dim=1)
        if it%4==0:
            candidate=(new*w2/(op._certificate_a*volume)<=.9*op._certificate_target) if certified else (rr<=1e-16*norm*norm)
            if bool(candidate):
                u=op.project_zero_mean(u);actual=rhs-op.apply(u)
                true=float(torch.linalg.vector_norm(actual)/norm)
                exact=op.exact_precond(actual);bound=float((actual*exact).sum())*w2/(op._certificate_a*volume)
                if not np.isfinite(true+bound) or bound<0:raise RuntimeError("Invalid certificate")
                if (bound<=op._certificate_target if certified else true<=1e-8):
                    return u,actual,dict(iterations=it,restarts=restarts,true_residuals=[true],bound=bound)
                r=actual;z=op.apply_precond(r);new=(r*z).sum(dim=1);p=z.clone();rho=new;restarts+=1;continue
        p=direction(r,z,p,new,rho);rho=new
    raise RuntimeError("Custom solve did not converge")
