"""Independent CPU checks of constitutive, HG-projector and energy formulas.

The explicit matrices below are reference checks on small periodic grids.
No production source, stopping tolerance or GPU default is modified.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repro/source/femlib"
OUT = ROOT / "review_results/structural_formula_20261020"


def references():
    package = types.ModuleType("_structural_reference")
    package.__path__ = [str(SOURCE)]
    sys.modules[package.__name__] = package
    result = {}
    for name in ("geometry", "c3d8r", "materials"):
        key = package.__name__ + "." + name
        spec = importlib.util.spec_from_file_location(key, SOURCE / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod; spec.loader.exec_module(mod)
        result[name] = mod
    return result["geometry"], result["c3d8r"], result["materials"]


def relative(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def tensor(e):
    out = np.zeros(np.shape(e)[:-1] + (3, 3))
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = e[..., 0], e[..., 1], e[..., 2]
    out[..., 1, 2] = out[..., 2, 1] = e[..., 3] / 2
    out[..., 0, 2] = out[..., 2, 0] = e[..., 4] / 2
    out[..., 0, 1] = out[..., 1, 0] = e[..., 5] / 2
    return out


def voigt_stress(s):
    return np.stack((s[..., 0, 0], s[..., 1, 1], s[..., 2, 2],
                     s[..., 1, 2], s[..., 0, 2], s[..., 0, 1]), axis=-1)


def factors(c):
    """Exact for the current six-parameter tetragonal local elastic law."""
    lam = c[1, 2]; mu = (c[1, 1]-c[1, 2])/2
    alpha = c[0, 1]-lam; gamma = c[5, 5]-mu
    beta = c[0, 0]-c[1, 1]-2*alpha-4*gamma
    delta = c[3, 3]-mu
    p = np.array([lam, mu, alpha, beta, gamma, delta])
    rebuilt = np.column_stack([structural_action(e, np.eye(3), p) for e in np.eye(6)])
    if relative(rebuilt, c) > 1e-13:
        raise ValueError("Material is outside this exact six-parameter symmetry class")
    return p


def structural_action(e, rotation, p, omit_delta=False):
    """Apply C to engineering strain through physical structural tensors."""
    lam, mu, alpha, beta, gamma, delta = p
    eps = tensor(np.asarray(e)); n = rotation[..., :, 0]
    b, c = rotation[..., :, 1], rotation[..., :, 2]
    r = np.einsum("...ij,...j->...i", eps, n)
    q = np.sum(n*r, axis=-1); t = np.trace(eps, axis1=-2, axis2=-1)
    nn = n[..., :, None]*n[..., None, :]
    shear = b[..., :, None]*c[..., None, :] + c[..., :, None]*b[..., None, :]
    sigma = ((lam*t+alpha*q)[..., None, None]*np.eye(3) + 2*mu*eps
             + (alpha*t+beta*q)[..., None, None]*nn
             + 2*gamma*(n[..., :, None]*r[..., None, :] + r[..., :, None]*n[..., None, :]))
    if not omit_delta:
        v = np.sum(eps*shear, axis=(-2, -1))
        sigma += (delta*v)[..., None, None]*shear
    return voigt_stress(sigma)


def isotropic_action(e, c):
    out = 2*c[3, 3]*tensor(e)
    out += (c[0, 1]*np.sum(e[..., :3], axis=-1))[..., None, None]*np.eye(3)
    return voigt_stress(out)


def projector_action(u, c, h, scale, geometry):
    """HG action using I minus the constant/linear nodal projector."""
    b, weight = geometry.make_B_matrices(*h, element_type="C3D8R")
    b = b[0]
    affine = np.column_stack((np.ones(8), 2*geometry.LOCAL_OFFSET-1))
    field = np.asarray(u).reshape(-1, 8, 3)
    q = np.einsum("ai,nad->nid", affine, field)
    complement = field-np.einsum("ai,nid->nad", affine, q)/8
    strain = np.asarray(u).reshape(-1, 24) @ b.T
    return (weight*(strain @ c.T) @ b + scale*complement.reshape(-1, 24)).reshape(np.shape(u))


def material_checks(materials, material):
    rng = np.random.default_rng(20261021)
    r = materials.build_rotation_matrices(rng.normal(size=(2048, 3)), rng.normal(size=(2048, 3)))
    e = rng.normal(size=(2048, 6)); rows = []
    for name, values in (("strong", material["yarn"]), ("near", dict(E11=74000., E22=74000., E33=74000.,
                          Nu12=.2, Nu13=.2, Nu23=.2, G12=30800., G13=30800., G23=30800.))):
        c = materials.engineering_to_voigt(*(values[k] for k in
             ("E11", "E22", "E33", "Nu12", "Nu13", "Nu23", "G12", "G13", "G23")))
        p = factors(c); rotated = materials.rotate_voigt_batch(c, r)
        expected = np.einsum("nij,nj->ni", rotated, e)
        actual = structural_action(e, r, p)
        # A transverse pure shear is a deliberate rejection test for dropping delta.
        shear = np.array([0., 0., 0., 1., 0., 0.])
        rows.append(dict(case=name, rotations=len(r), coefficients=p.tolist(),
            constitutive_relative=relative(expected, actual),
            max_rotation_relative=max(relative(a,b) for a,b in zip(expected, actual)),
            false_TI_pure_transverse_shear_relative=relative(c@shear, structural_action(shear, np.eye(3), p, True))))
    return rows


def projector_checks(geometry, hg):
    rng = np.random.default_rng(1620170504); h_basis = hg.hourglass_basis()
    affine = np.column_stack((np.ones(8), 2*geometry.LOCAL_OFFSET-1))
    complement = np.kron(np.eye(8)-affine@affine.T/8, np.eye(3))
    result = dict(projector_relative=relative(h_basis@h_basis.T, complement), rows=[])
    for h in ((.3,.5,.7), (1/192,1/192,.4/192)):
        a = rng.normal(size=(6,6)); c = a.T@a+np.eye(6)
        u = rng.normal(size=(64,24))
        for coefficient in (0., .1, 1.):
            k = hg.make_c3d8r_element_stiffness(c,h,hourglass_coefficient=coefficient)
            scale = hg.hourglass_scale(c,h,coefficient)
            actual = projector_action(u,c,h,scale,geometry); expected = u@k.T
            energy = np.sum(u*actual,axis=1)
            result["rows"].append(dict(voxel=h,coefficient=coefficient,
                force_relative=relative(expected,actual),
                energy_relative=relative(energy,np.einsum("ni,ij,nj->n",u,k,u))))
    return result


def periodic_checks(geometry,hg,materials,material,kind):
    rng=np.random.default_rng(20261022)
    shape=(4,5,6); nz,ny,nx=shape; n=int(np.prod(shape)); h=(.3,.5,.7); volume=n*np.prod(h)
    cm=materials.isotropic_to_voigt(material["matrix"]["E"],material["matrix"]["Nu"])
    y=material["yarn"]
    cy=materials.engineering_to_voigt(*(y[k] for k in ("E11","E22","E33","Nu12","Nu13","Nu23","G12","G13","G23")))
    p=factors(cy)
    rotation=materials.build_rotation_matrices(rng.normal(size=(n,3)),rng.normal(size=(n,3)))
    active=np.zeros(n,dtype=bool); active[rng.choice(n,n//5,replace=False)]=True
    c=materials.rotate_voigt_batch(cy,rotation);c[~active]=cm
    b,weight=geometry.make_B_matrices(*h,element_type="C3D8R" if kind=="hg" else "C3D8")
    K=np.zeros((3*n,3*n)); alt=np.zeros_like(K); F=np.zeros((3*n,6)); f_alt=np.zeros_like(F)
    Cavg=np.mean(c,axis=0); nodes=[]
    affine=np.column_stack((np.ones(8),2*geometry.LOCAL_OFFSET-1))
    complement=np.kron(np.eye(8)-affine@affine.T/8,np.eye(3))
    # Isotropic reference may differ from the physical matrix; the bound holds for any SPD C0.
    c0=materials.isotropic_to_voigt(15000.,.28); M=np.zeros_like(K)
    a_bound=min(float(eigh(cm,c0,eigvals_only=True)[0]),float(eigh(cy,c0,eigvals_only=True)[0]))
    if kind=="hg":
        gamma0=hg.hourglass_scale(c0,h,.1)
        a_bound=min(a_bound,hg.hourglass_scale(cm,h,.1)/gamma0,hg.hourglass_scale(cy,h,.1)/gamma0)
    k0=weight*np.einsum("gip,ij,gjq->pq",b,c0,b)
    if kind=="hg":k0+=gamma0*(hg.hourglass_basis()@hg.hourglass_basis().T)
    for e in range(n):
        z,rem=divmod(e,ny*nx); yy,x=divmod(rem,nx)
        ids=np.array([(((z+oz)%nz*ny+(yy+oy)%ny)*nx+(x+ox)%nx) for ox,oy,oz in geometry.LOCAL_OFFSET])
        nodes.append(ids); dofs=(3*ids[:,None]+np.arange(3)).ravel()
        apply=lambda strain:structural_action(strain,rotation[e],p) if active[e] else isotropic_action(strain,cm)
        ke=weight*np.einsum("gip,ij,gjq->pq",b,c[e],b)
        other=sum(weight*point.T@apply(point.T).T for point in b)
        if kind=="hg":
            scale=hg.hourglass_scale(c[e],h,.1)
            ke+=scale*(hg.hourglass_basis()@hg.hourglass_basis().T)
            other+=scale*complement
        K[np.ix_(dofs,dofs)]+=ke;alt[np.ix_(dofs,dofs)]+=other;M[np.ix_(dofs,dofs)]+=k0
        F[dofs]-=weight*np.sum(b.transpose(0,2,1)@c[e],axis=0)
        f_alt[dofs]-=sum(weight*point.T@apply(np.eye(6)).T for point in b)
    # Pin only translation for the independent solve, then report zero-mean fields.
    U=np.zeros_like(F);U[3:]=np.linalg.solve(K[3:,3:],F[3:]); U=U.reshape(n,3,6);U-=U.mean(axis=0,keepdims=True);U=U.reshape(3*n,6)
    Ualt=np.zeros_like(F);Ualt[3:]=np.linalg.solve(alt[3:,3:],f_alt[3:]);Ualt=Ualt.reshape(n,3,6);Ualt-=Ualt.mean(axis=0,keepdims=True);Ualt=Ualt.reshape(3*n,6)
    CH=Cavg-F.T@U/volume
    stress_columns=[]
    for j in range(6):
        average=np.zeros(6)
        for e,ids in enumerate(nodes):
            strain=b@Ualt.reshape(n,3,6)[ids,:,j].reshape(24)+np.eye(6)[j]
            values=structural_action(strain,rotation[e],p) if active[e] else isotropic_action(strain,cm)
            average+=values.mean(axis=0)/n
        stress_columns.append(average)
    # Exact SPD preconditioner for a mathematically certified CPU experiment.
    ev,Q=np.linalg.eigh(M);keep=ev>max(ev)*1e-12
    assert int(keep.sum())==3*n-3,"Only the three periodic translation modes may be excluded"
    inv=(Q[:,keep]/ev[keep])@Q[:,keep].T
    evk=np.linalg.eigvalsh((Q[:,keep].T@(K-a_bound*M)@Q[:,keep]))
    cg=[]; certified=[]
    # Lower bound on the Mandel effective tensor gives a strict relative-Frobenius target.
    w=materials.MANDEL_W
    cmin=min(np.linalg.eigvalsh(materials.voigt_to_mandel(cm))[0],np.linalg.eigvalsh(materials.voigt_to_mandel(cy))[0])
    target=1e-8*np.sqrt(6)*cmin/6
    for j in range(6):
        f=F[:,j]; u=np.zeros_like(f);r=f.copy();z=inv@r;direction=z.copy();rho=float(r@z)
        normf=np.linalg.norm(f); first=None
        for it in range(1,501):
            ap=K@direction; alpha=rho/float(direction@ap); u+=alpha*direction; r-=alpha*ap
            actual=f-K@u; bound=float(actual@inv@actual)/(a_bound*volume)*w[j]**2
            true=float(np.linalg.norm(actual)/normf)
            if first is None and bound<=target:
                first=dict(iterations=it,true_residual=true,diagonal_Mandel_energy_error_bound=bound,
                           u=u.copy())
            if true<=1e-8:
                cg.append(dict(iterations=it,true_residual=true));break
            z=inv@r;new=float(r@z);direction=z+(new/rho)*direction;rho=new
        else:raise RuntimeError("CG failed")
        if first is None:raise RuntimeError("Energy certificate not reached")
        certified.append(first)
    approx=np.column_stack([row.pop("u") for row in certified])
    energy=Cavg+(approx.T@K@approx-approx.T@F-F.T@approx)/volume
    error=approx-U;identity=error.T@K@error/volume
    residual=F-K@approx;upper=residual.T@inv@residual/(a_bound*volume)
    cm_error=(energy-CH)*w[:,None]*w[None,:]
    diag_bound=sum(row["diagonal_Mandel_energy_error_bound"] for row in certified)
    trace_bound=diag_bound/np.linalg.norm(materials.voigt_to_mandel(CH))
    arbitrary=rng.normal(size=U.shape);arbitrary*=np.linalg.norm(U)*1e-3/np.linalg.norm(arbitrary)
    errors=[]
    for scale in (1.,.1,.01):
        v=U+scale*arbitrary
        stress=Cavg-F.T@v/volume
        variational=Cavg+(v.T@K@v-v.T@F-F.T@v)/volume
        errors.append(dict(perturbation_scale=scale,stress_formula_error=relative(stress,CH),
                           variational_formula_error=relative(variational,CH)))
    return dict(kind=kind,grid=shape,active_elements=int(active.sum()),
        K_relative=relative(K,alt),RHS_relative=relative(F,f_alt),
        displacement_relative=relative(U,Ualt),CH_relative=relative(CH,np.column_stack(stress_columns)),
        reference_lower_bound_a=a_bound,minimum_eigenvalue_K_minus_aM=float(evk.min()),
        energy_error_identity_relative=relative(energy-CH,identity),
        energy_error_identity_absolute_scaled=float(np.linalg.norm(energy-CH-identity)/np.linalg.norm(CH)),
        certificate_minimum_eigenvalue=float(np.linalg.eigvalsh((upper-identity+upper.T-identity.T)/2).min()),
        certified_CH_Mandel_relative_error=float(np.linalg.norm(cm_error)/np.linalg.norm(materials.voigt_to_mandel(CH))),
        certified_CH_relative_error_upper_bound=trace_bound,
        conventional_true_residual_solves=cg,energy_certified_solves=certified,
        perturbation_errors=errors)


def main():
    geometry,hg,materials=references()
    source=ROOT/"review_results/formula_gpu_20261020/remote/results/measure_hg_baseline_192_0.json"
    record=json.loads(source.read_text());material=record["material"]
    result=dict(protocol="CPU FP64 independent frozen material rotation, conventional B integration and pinned direct solves; no GPU timing",
        material_source=str(source.relative_to(ROOT)),material=material,
        constitutive=material_checks(materials,material),hourglass_projector=projector_checks(geometry,hg),
        periodic=[periodic_checks(geometry,hg,materials,material,k) for k in ("hg","fi")],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [SOURCE/(name+".py") for name in ("geometry","c3d8r","materials")]+[source,Path(__file__)]})
    # The energy identity compares small differences; use an absolute, scaled check there.
    result["passed"]=(all(r["max_rotation_relative"]<1e-12 for r in result["constitutive"])
        and result["hourglass_projector"]["projector_relative"]<1e-14
        and all(r["force_relative"]<1e-12 and r["energy_relative"]<1e-12 for r in result["hourglass_projector"]["rows"])
        and all(r["K_relative"]<1e-12 and r["RHS_relative"]<1e-12 and r["displacement_relative"]<1e-10
                and r["CH_relative"]<1e-10 and r["certified_CH_Mandel_relative_error"]<=r["certified_CH_relative_error_upper_bound"]*1.001
                and r["energy_error_identity_absolute_scaled"]<1e-12 and r["minimum_eigenvalue_K_minus_aM"]>=-1e-10
                and r["certificate_minimum_eigenvalue"]>=-1e-10
                and r["certified_CH_relative_error_upper_bound"]<=1e-8 for r in result["periodic"]))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
