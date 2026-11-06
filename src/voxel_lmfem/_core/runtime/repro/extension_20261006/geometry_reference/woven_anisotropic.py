# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9"]
# ///
"""Continuous sinusoidal plain weave + tangent-following anisotropic elasticity.

Fourier geometry is analytic in both cross-section (Zernike/Bessel) and
axial direction (Jacobi-Anger expansion). Only the variable constitutive mass
matrix is quadrature-integrated. Independent 3D fully integrated Q1 FE reference.
This idealized weave uses vertical elliptical sections, NOT normal-plane sweeps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import jv

from run_experiment import basis_values, metrics, modes, stiffness


ROOT = Path(__file__).parent
SQRT2 = np.sqrt(2.0)
MANDEL = np.zeros((6, 3, 3))
for _i in range(3):
    MANDEL[_i, _i, _i] = 1
for _i, (_j, _k) in enumerate([(1, 2), (0, 2), (0, 1)], 3):
    MANDEL[_i, _j, _k] = MANDEL[_i, _k, _j] = 1 / SQRT2


@dataclass(frozen=True)
class Weave:
    width: float = 0.18   # transverse ellipse semiaxis
    thickness: float = 0.035  # vertical ellipse semiaxis
    amplitude: float = 0.09
    height: float = 0.4

    @property
    def fiber_fraction(self):
        return np.pi * self.width * self.thickness / self.height

    @property
    def fraction(self):
        return 4 * self.fiber_fraction

    def validate(self):
        # Sufficient global nonintersection bound in every warp/weft overlap patch.
        cross_gap = 2 * self.amplitude * np.cos(2 * np.pi * self.width) - 2 * self.thickness
        layer_gap = self.height - 2 * (self.amplitude + self.thickness)
        if not (0 < self.width < 0.25 and self.thickness > 0 and cross_gap > 0 and layer_gap > 0):
            raise ValueError("Geometry must have strictly separated periodic yarn volumes")
        return {"conservative_warp_weft_vertical_gap": float(cross_gap),
                "periodic_layer_gap": float(layer_gap), "exact_fiber_fraction": self.fraction,
                "max_tangent_angle_degrees": float(np.rad2deg(np.arctan(2 * np.pi * self.amplitude)))}


@dataclass(frozen=True)
class Yarn:
    axis: int
    center: float
    sign: int


YARNS = [Yarn(0, 0.0, 1), Yarn(0, 0.5, -1), Yarn(1, 0.0, -1), Yarn(1, 0.5, 1)]


def local_ti(EL=40.0, ET=5.0, GLT=3.0, nuLT=0.25, nuTT=0.3):
    """Transversely isotropic Mandel stiffness, local fiber direction = axis 1.

    nuLT = -epsilon_T/epsilon_L under longitudinal uniaxial stress.
    Reciprocal Poisson ratio is nuTL=nuLT*ET/EL.
    """
    s = np.zeros((6, 6))
    s[0, 0], s[1, 1], s[2, 2] = 1 / EL, 1 / ET, 1 / ET
    s[0, 1] = s[1, 0] = s[0, 2] = s[2, 0] = -nuLT / EL
    s[1, 2] = s[2, 1] = -nuTT / ET
    s[3, 3] = (1 + nuTT) / ET
    s[4, 4] = s[5, 5] = 1 / (2 * GLT)
    if np.linalg.eigvalsh(s).min() <= 0:
        raise ValueError("Material compliance is not positive definite")
    return np.linalg.inv(s)


MATERIALS = {
    "ti40": {"EL": 40.0, "ET": 5.0, "GLT": 3.0, "nuLT": 0.25, "nuTT": 0.3},
    "ti100": {"EL": 100.0, "ET": 5.0, "GLT": 3.0, "nuLT": 0.25, "nuTT": 0.3},
    "iso5": {"EL": 5.0, "ET": 5.0, "GLT": 5 / 2.6, "nuLT": 0.3, "nuTT": 0.3},
}


def rotation(s, yarn, geom, follow_tangent=True):
    slope = -yarn.sign * geom.amplitude * 2 * np.pi * np.sin(2 * np.pi * np.asarray(s))
    if not follow_tangent:
        slope = np.zeros_like(slope)
    tangent = np.zeros((len(slope), 3))
    tangent[:, yarn.axis] = 1
    tangent[:, 2] = slope
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    transverse = np.zeros_like(tangent)
    transverse[:, 1 - yarn.axis] = 1 if yarn.axis == 0 else -1
    normal = np.cross(tangent, transverse)
    return np.stack([tangent, transverse, normal], axis=-1)


def mandel_rotation(q):
    """Orthogonal rotation on symmetric tensors: T_ab = M_a : Q M_b Q^T."""
    return np.einsum("aij,...ik,bkl,...jl->...ab", MANDEL, q, MANDEL, q, optimize=True)


def yarn_stiffness(s, yarn, geom, material, follow_tangent=True):
    t = mandel_rotation(rotation(s, yarn, geom, follow_tangent))
    return t @ local_ti(**MATERIALS[material]) @ t.swapaxes(-1, -2)


def axial_modes(order):
    return [(0, "constant")] + [(l, kind) for l in range(1, order + 1) for kind in ["cos", "sin"]]


def axial_values(s, order):
    return np.stack([np.ones_like(s) if l == 0 else
                     SQRT2 * (np.cos(2 * np.pi * l * s) if kind == "cos" else np.sin(2 * np.pi * l * s))
                     for l, kind in axial_modes(order)], axis=-1)


def scalar_basis(degree, order):
    return [(cross, axial) for cross in modes(degree) for axial in axial_modes(order)]


def geometry_fourier(k, geom, degree, order):
    """Exact normalized transforms of supported real cross/axial product modes.

    k: integer triplets. Physical k_z is 2*pi*k_z/height.
    Fiber mapping x=(s,yc+a*u,sign*A*cos(2*pi*s)+b*v) for warp;
    swap x,y for weft. Volume Jacobian a*b is exactly constant.
    """
    cross = modes(degree)
    ks = np.asarray(k, dtype=int)
    results = []
    for yarn in YARNS:
        u = 2 * np.pi * geom.width * ks[:, 1 - yarn.axis]
        v = 2 * np.pi * geom.thickness * ks[:, 2] / geom.height
        rho, theta = np.hypot(u, v), np.arctan2(v, u)
        radials = {}
        for n in {c[0] for c in cross}:
            radials[n] = np.divide(jv(n + 1, rho), rho, out=np.zeros_like(rho), where=rho != 0)
            radials[n][rho == 0] = 0.5 if n == 0 else 0.0
        f_cross = []
        for n, m, kind in cross:
            norm = np.sqrt((n + 1) * (1 if m == 0 else 2))
            angular = np.cos(m * theta) if kind == "cos" else np.sin(m * theta)
            f_cross.append(2 * geom.fiber_fraction * (-1j)**n * norm * radials[n] * angular)
        f_cross = np.stack(f_cross, axis=-1)
        beta = 2 * np.pi * yarn.sign * geom.amplitude * ks[:, 2] / geom.height

        def t(n):
            return (-1j)**(n % 4) * jv(n, beta)

        f_axial = []
        for l, kind in axial_modes(order):
            n = ks[:, yarn.axis]
            if l == 0:
                f_axial.append(t(n))
            elif kind == "cos":
                f_axial.append((t(n - l) + t(n + l)) / SQRT2)
            else:
                f_axial.append((t(n - l) - t(n + l)) / (1j * SQRT2))
        f_axial = np.stack(f_axial, axis=-1)
        phase = np.exp(-2j * np.pi * ks[:, 1 - yarn.axis] * yarn.center)
        results.append((f_cross[:, :, None] * f_axial[:, None, :] * phase[:, None, None]).reshape(len(k), -1))
    return np.concatenate(results, axis=1)


def strain_b(k):
    b = np.zeros((len(k), 6, 3))
    for d in range(3):
        b[:, d, d] = k[:, d]
    for row, (i, j) in enumerate([(1, 2), (0, 2), (0, 1)], 3):
        b[:, row, i] = k[:, j] / SQRT2
        b[:, row, j] = k[:, i] / SQRT2
    return b


def green3d(k, geom, c0):
    q = np.asarray(k, dtype=float) / np.array([1.0, 1.0, geom.height])
    b = strain_b(q)
    acoustic = b.swapaxes(1, 2) @ c0 @ b
    acoustic[np.all(q == 0, axis=1)] = np.eye(3)
    return b @ np.linalg.inv(acoustic) @ b.swapaxes(1, 2)


def kernel_scan(geom, degree, order, cutoffs, c0, cache_dir):
    """Real-polarization Galerkin matrix from all ±k; incremental cubic shells."""
    p = 4 * len(scalar_basis(degree, order))
    h = np.zeros((p, 6, p, 6))
    previous, cumulative = 0, 0.0
    signature = json.dumps(asdict(geom), sort_keys=True)
    for cutoff in sorted(cutoffs):
        cache = cache_dir / f"kernel_p{degree}_l{order}_k{cutoff}.npz"
        if cache.exists():
            with np.load(cache) as saved:
                if str(saved["geometry"]) != signature or not np.array_equal(saved["c0"], c0):
                    raise ValueError("Kernel cache metadata mismatch")
                h = saved["h"].reshape(p, 6, p, 6)
                cumulative = float(saved["build_seconds"])
        else:
            started = time.perf_counter()
            transverse = np.arange(-cutoff, cutoff + 1)
            iy, iz = np.meshgrid(transverse, transverse, indexing="ij")
            yz = np.column_stack([iy.ravel(), iz.ravel()])
            for ix in range(-cutoff, cutoff + 1):
                k = np.column_stack([np.full(len(yz), ix), yz])
                k = k[np.max(np.abs(k), axis=1) > previous]
                if not len(k):
                    continue
                f = geometry_fourier(k, geom, degree, order)
                fr, fi = np.ascontiguousarray(f.real), np.ascontiguousarray(f.imag)
                g = green3d(k, geom, c0)
                for a in range(6):
                    for b in range(a, 6):
                        weights = g[:, a, b, None]
                        block = fr.T @ (weights * fr) + fi.T @ (weights * fi)
                        h[:, a, :, b] += block
                        if a != b:
                            h[:, b, :, a] += block
            cumulative += time.perf_counter() - started
            np.savez_compressed(cache, h=h.reshape(p * 6, p * 6), geometry=signature, c0=c0,
                                build_seconds=cumulative)
        yield cutoff, h.reshape(p * 6, p * 6), cumulative
        previous = cutoff


def material_blocks(geom, degree, order, material, quadrature=256, follow_tangent=True):
    """Exact cross-section mass, periodic 1D quadrature for variable anisotropy."""
    s = np.arange(quadrature) / quadrature
    axial = axial_values(s, order)
    nc, na = len(modes(degree)), len(axial_modes(order))
    p = nc * na
    mass = np.zeros((4 * p * 6, 4 * p * 6))
    rhs = np.zeros((4 * p * 6, 6))
    c0 = stiffness()
    avg_c = (1 - geom.fraction) * c0
    avg_s = (1 - geom.fraction) * np.linalg.inv(c0)
    min_delta = np.inf
    for iy, yarn in enumerate(YARNS):
        c = yarn_stiffness(s, yarn, geom, material, follow_tangent)
        delta = c - c0
        min_delta = min(min_delta, np.linalg.eigvalsh(delta).min())
        if min_delta <= 0:
            raise ValueError("Polarization reference requires strictly positive C_yarn-C_matrix")
        inv = np.linalg.inv(delta)
        block = geom.fiber_fraction * np.einsum("sl,sab,sm->lamb", axial, inv, axial, optimize=True) / quadrature
        block = block.reshape(na * 6, na * 6)
        for cross in range(nc):
            idx = (iy * p + cross * na) * 6
            mass[idx:idx + na*6, idx:idx + na*6] = block
        rhs[iy*p*6:iy*p*6+6] = geom.fiber_fraction * np.eye(6)
        avg_c += geom.fiber_fraction * c.mean(axis=0)
        avg_s += geom.fiber_fraction * np.linalg.inv(c).mean(axis=0)
    return mass, rhs, {"voigt": avg_c, "reuss": np.linalg.inv(avg_s), "min_delta_eigenvalue": float(min_delta)}


def solve_modal(h, geom, max_degree, max_order, degree, order, material, follow_tangent=True, quadrature=256):
    started = time.perf_counter()
    large_basis = scalar_basis(max_degree, max_order)
    small_basis = scalar_basis(degree, order)
    scalar = [y * len(large_basis) + large_basis.index(m) for y in range(4) for m in small_basis]
    idx = (np.array(scalar)[:, None] * 6 + np.arange(6)).ravel()
    mass, rhs, bounds = material_blocks(geom, degree, order, material, quadrature, follow_tangent)
    a = h[np.ix_(idx, idx)] + mass
    t = cho_solve(cho_factor(a), rhs)
    c = stiffness() + rhs.T @ t
    residual = np.linalg.norm(a @ t - rhs) / np.linalg.norm(rhs)
    return c, {"unknowns": len(idx), "solve_seconds": time.perf_counter() - started,
               "relative_linear_residual": float(residual),
               "min_delta_eigenvalue": bounds["min_delta_eigenvalue"],
               "voigt": bounds["voigt"].tolist(), "reuss": bounds["reuss"].tolist()}


def material_grid(shape, geom, material, follow_tangent=True):
    """Voxel oracle only: direct membership in the same continuous geometry."""
    nx, ny, nz = shape
    xyz = [(np.arange(n) + 0.5) / n for n in shape]
    xyz[2] = (xyz[2] - 0.5) * geom.height
    points = np.meshgrid(*xyz, indexing="ij", sparse=True)
    ids = np.zeros(shape, dtype=np.int32)
    ctable = [stiffness()[None]]
    offset = 1
    overlap = 0
    for yarn in YARNS:
        s = points[yarn.axis]
        d = (points[1 - yarn.axis] - yarn.center + 0.5) % 1 - 0.5
        zc = yarn.sign * geom.amplitude * np.cos(2 * np.pi * s)
        dz = (points[2] - zc + geom.height / 2) % geom.height - geom.height / 2
        inside = (d / geom.width)**2 + (dz / geom.thickness)**2 < 1
        overlap += int(np.count_nonzero(inside & (ids != 0)))
        n = shape[yarn.axis]
        axshape = [1, 1, 1]
        axshape[yarn.axis] = n
        which = np.broadcast_to(np.arange(n).reshape(axshape) + offset, shape)
        ids[inside] = which[inside]
        ctable.append(yarn_stiffness(xyz[yarn.axis], yarn, geom, material, follow_tangent))
        offset += n
    if overlap:
        raise ValueError(f"Overlapping yarn sample points: {overlap}")
    return ids, np.concatenate(ctable), float((ids != 0).mean())


def q1_tables(shape, geom, ctable):
    corners = np.array(list(product([0, 1], repeat=3)), dtype=int)
    sizes = np.array([1, 1, geom.height]) / shape
    signs = 2 * corners - 1
    ke = np.zeros((len(ctable), 24, 24))
    le = np.zeros((len(ctable), 24, 6))
    weight = np.prod(sizes) / 8
    for gauss in product([-1 / np.sqrt(3), 1 / np.sqrt(3)], repeat=3):
        grads = np.zeros((8, 3))
        for d in range(3):
            other = [q for q in range(3) if q != d]
            grads[:, d] = signs[:, d] * np.prod(1 + signs[:, other] * np.array(gauss)[other], axis=1) / (4 * sizes[d])
        b = strain_b(grads).transpose(1, 0, 2).reshape(6, 24)
        ke += (b.T @ ctable @ b) * weight
        le += (b.T @ ctable) * weight
    return corners, ke, le


def voxel_reference(shape, geom, material, follow_tangent=True, rtol=1e-8, uniform=None, laminate=False):
    """CPU 3D periodic C3D8, 8-point full integration, block-sparse assembly."""
    started = time.perf_counter()
    shape = tuple(shape)
    nnode = int(np.prod(shape))
    if uniform is not None:
        ids, ctable, vf = np.zeros(shape, np.int32), np.asarray(uniform)[None], 0.0
    elif laminate:
        ids = np.broadcast_to((np.arange(shape[0]) >= shape[0] // 2)[:, None, None], shape).astype(np.int32)
        ctable, vf = np.stack([stiffness(), stiffness(10)]), 0.5
    else:
        ids, ctable, vf = material_grid(shape, geom, material, follow_tangent)
    corners, ke, le = q1_tables(shape, geom, ctable)
    offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    offsets_lookup = {tuple(d): i for i, d in enumerate(offsets)}
    data = np.zeros((nnode, 27, 3, 3))
    rhs = np.zeros(shape + (3, 6))
    for ia, ca in enumerate(corners):
        shifted_ids = np.roll(ids, tuple(ca), axis=(0, 1, 2)).ravel()
        rhs += le[shifted_ids, ia*3:ia*3+3].reshape(shape + (3, 6))
        for ib, cb in enumerate(corners):
            tap = offsets_lookup[tuple(cb - ca)]
            data[:, tap] += ke[shifted_ids, ia*3:ia*3+3, ib*3:ib*3+3]
    coordinates = np.indices(shape, dtype=np.int32).reshape(3, -1).T
    cols = np.empty((nnode, 27), dtype=np.int32)
    for tap, d in enumerate(offsets):
        v = (coordinates + d) % np.array(shape)
        cols[:, tap] = (v[:, 0] * shape[1] + v[:, 1]) * shape[2] + v[:, 2]
    k = sparse.bsr_matrix((data.reshape(nnode * 27, 3, 3), cols.ravel(),
                           np.arange(nnode + 1, dtype=np.int32) * 27), shape=(nnode * 3, nnode * 3))
    rhs = rhs.reshape(nnode * 3, 6)
    # Choose actual mean stiffness as homogeneous discrete FE preconditioner.
    counts = np.bincount(ids.ravel(), minlength=len(ctable))
    cmean = np.einsum("t,tij->ij", counts / nnode, ctable)
    _, keref, _ = q1_tables(shape, geom, cmean[None])
    kernel = np.zeros(shape + (3, 3))
    for ia, ca in enumerate(corners):
        for ib, cb in enumerate(corners):
            d = (ca - cb) % np.array(shape)
            kernel[tuple(d)] += keref[0, ia*3:ia*3+3, ib*3:ib*3+3]
    symbol = np.fft.rfftn(kernel, axes=(0, 1, 2))
    symbol[0, 0, 0] = np.eye(3)
    inverse = np.linalg.inv(symbol)
    inverse[0, 0, 0] = 0
    del kernel, symbol, data, cols, coordinates

    def precond(v):
        f = np.fft.rfftn(v.reshape(shape + (3,)), axes=(0, 1, 2))
        z = np.einsum("...ij,...j->...i", inverse, f)
        return np.fft.irfftn(z, s=shape, axes=(0, 1, 2)).ravel()

    pre = LinearOperator(k.shape, matvec=precond)
    solution = np.zeros_like(rhs)
    iterations, residuals = [], []
    assembly = time.perf_counter() - started
    for load in range(6):
        b = rhs[:, load]
        if np.linalg.norm(b) < 1e-13:
            iterations.append(0)
            residuals.append(0.0)
            continue
        count = [0]

        def callback(_):
            count[0] += 1

        solution[:, load], info = cg(k, b, M=pre, rtol=rtol, atol=0, maxiter=1000, callback=callback)
        residuals.append(float(np.linalg.norm(k @ solution[:, load] - b) / np.linalg.norm(b)))
        iterations.append(count[0])
        if info or residuals[-1] > 5 * rtol:
            raise RuntimeError(f"Reference CG failed: info={info}, residual={residuals[-1]}")
    c = cmean - rhs.T @ solution / geom.height
    avg_compliance = np.einsum("t,tij->ij", counts / nnode, np.linalg.inv(ctable))
    return c, {"shape": list(shape), "unknowns": nnode * 3 - 3, "voxel_fraction": vf,
               "relative_linear_residual": max(residuals), "iterations": iterations,
               "assembly_seconds": assembly, "total_seconds": time.perf_counter() - started,
               "bsr_bytes": k.data.nbytes + k.indices.nbytes + k.indptr.nbytes,
               "voigt": cmean.tolist(), "reuss": np.linalg.inv(avg_compliance).tolist()}


def validate(geom):
    checks = geom.validate()
    rng = np.random.default_rng(20260922)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    t = mandel_rotation(q)
    checks["mandel_orthogonality_max_error"] = float(np.max(np.abs(t @ t.T - np.eye(6))))
    checks["isotropic_rotation_max_error"] = float(np.max(np.abs(t @ stiffness() @ t.T - stiffness())))
    checks["TI_isotropic_limit_max_error"] = float(np.max(np.abs(local_ti(**MATERIALS["iso5"]) - stiffness(5))))
    # Independent 4th-order contraction, not a second call to Mandel rotation.
    c_local = local_ti(**MATERIALS["ti40"])
    tensor = np.einsum("ab,aij,bkl->ijkl", c_local, MANDEL, MANDEL)
    rotated = np.einsum("ia,jb,kc,ld,abcd->ijkl", q, q, q, q, tensor, optimize=True)
    independent = np.einsum("aij,ijkl,bkl->ab", MANDEL, rotated, MANDEL)
    checks["rotation_vs_fourth_order_max_error"] = float(np.max(np.abs(independent - t @ c_local @ t.T)))
    # Brute force 3D parametric quadrature verifies sinusoidal Fourier coefficients.
    degree, order = 2, 2
    ks = np.array([[0, 0, 0], [1, 2, 1], [-2, 1, -3], [3, -2, 2], [0, 0, 5]])
    radial, wr = np.polynomial.legendre.leggauss(20)
    radial, wr = (radial + 1) / 2, wr / 2
    theta = np.arange(48) * 2 * np.pi / 48
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    w_cross = np.broadcast_to((wr * radial)[:, None] * 2 * np.pi / 48, rr.shape).ravel()
    cross = basis_values(rr.ravel(), tt.ravel(), modes(degree))
    uv = np.column_stack([geom.width * rr.ravel() * np.cos(tt.ravel()),
                          geom.thickness * rr.ravel() * np.sin(tt.ravel())])
    ns = 128
    s = np.arange(ns) / ns
    axial = axial_values(s, order)
    exact = geometry_fourier(ks, geom, degree, order)
    numerical = []
    for yarn in YARNS:
        points = np.zeros((ns, len(uv), 3))
        points[:, :, yarn.axis] = s[:, None]
        points[:, :, 1-yarn.axis] = yarn.center + uv[None, :, 0]
        points[:, :, 2] = yarn.sign * geom.amplitude * np.cos(2*np.pi*s[:, None]) + uv[None, :, 1]
        phase = np.exp(-2j*np.pi * np.einsum("kd,sud->ksu", ks / np.array([1, 1, geom.height]), points))
        integral = np.einsum("ksu,ui,sl,u->kil", phase, cross, axial, w_cross, optimize=True)
        numerical.append((integral * geom.width * geom.thickness / (geom.height * ns)).reshape(len(ks), -1))
    checks["Fourier_vs_direct_3D_quadrature_max_error"] = float(np.max(np.abs(exact - np.concatenate(numerical, axis=1))))
    m1, r1, _ = material_blocks(geom, 2, 4, "ti100", 128)
    m2, r2, _ = material_blocks(geom, 2, 4, "ti100", 256)
    checks["material_quadrature_relative_change"] = float(np.linalg.norm(m1 - m2) / np.linalg.norm(m2))
    g = green3d(ks, geom, stiffness())
    checks["green_projector_max_error"] = float(np.max(np.abs(g @ stiffness() @ g - g)))
    # Independent anisotropic uniform FE check and two exact laminate shears.
    c_uniform = t @ c_local @ t.T
    cf, _ = voxel_reference((4, 4, 4), geom, "ti40", uniform=c_uniform)
    checks["uniform_anisotropic_FE_max_error"] = float(np.max(np.abs(cf - c_uniform)))
    cl, _ = voxel_reference((8, 8, 4), geom, "iso5", laminate=True)
    exact_shear = np.array([stiffness()[3, 3] * 5.5, stiffness()[4, 4] / 0.55])
    checks["exact_laminate_shear_relative_error"] = float(np.max(np.abs(np.diag(cl)[3:5] / exact_shear - 1)))
    for key, value in checks.items():
        if "error" in key or "quadrature_relative" in key:
            assert value < 1e-9, (key, value)
    return checks


def record(c, info, **parameters):
    return {**parameters, **info, "C_mandel": c.tolist(), "metrics": metrics(c)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "woven_results")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[8, 12, 20, 32])
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--grids", nargs="+", type=int, default=[24, 32, 48, 64])
    parser.add_argument("--materials", nargs="+", choices=list(MATERIALS), default=["iso5", "ti40", "ti100"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--modal-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "kernels"
    cache.mkdir(exist_ok=True)
    geom = Weave()
    path = args.output / "results.json"
    if args.quick:
        args.cutoffs, args.max_degree, args.max_order, args.grids = [4, 8], 1, 2, [12, 16]
        args.materials = ["ti40"]
    if path.exists():
        result = json.loads(path.read_text())
        if result["geometry"] != asdict(geom):
            raise ValueError("Existing result geometry does not match")
        if result["materials"] != MATERIALS:
            raise ValueError("Existing results use different material parameters")
    else:
        result = {"metadata": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "platform": platform.platform(),
                               "blas_threads": os.environ["OPENBLAS_NUM_THREADS"],
                               "geometry_type": "sinusoidal 2x2 plain weave with vertical elliptical sections",
                               "scope": "full 3D periodic linear elasticity; synthetic parameters; no TexGen import"},
                  "geometry": asdict(geom), "yarns": [asdict(y) for y in YARNS],
                  "materials": MATERIALS, "validation": validate(geom), "modal": [], "voxel": []}
        path.write_text(json.dumps(result, indent=2) + "\n")
    print("Validation:", result["validation"], flush=True)
    if not args.reference_only:
        configs = [(degree, order) for degree in range(args.max_degree + 1)
                   for order in sorted(set(range(0, args.max_order + 1, 2)) | {args.max_order})]
        for cutoff, h, seconds in kernel_scan(geom, args.max_degree, args.max_order, args.cutoffs, stiffness(), cache):
            for material in args.materials:
                for degree, order in configs:
                    for follow in ([True, False] if material == "ti40" and (degree, order) == configs[-1] else [True]):
                        if any(v["material"] == material and v["degree"] == degree and v["order"] == order
                               and v["cutoff"] == cutoff and v["follow_tangent"] == follow for v in result["modal"]):
                            continue
                        c, info = solve_modal(h, geom, args.max_degree, args.max_order, degree, order, material, follow)
                        result["modal"].append(record(c, info, material=material, degree=degree, order=order,
                                                      cutoff=cutoff, follow_tangent=follow,
                                                      kernel_build_seconds=seconds, kernel_unknowns=h.shape[0]))
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Modal cutoff {cutoff}: kernel {seconds:.2f}s, max unknowns {h.shape[0]}", flush=True)
    if not args.modal_only:
        for material in args.materials:
            for n in args.grids:
                for follow in ([True, False] if material == "ti40" and n == max(args.grids) else [True]):
                    shape = [n, n, max(4, n // 2)]
                    if any(v["material"] == material and v["shape"] == shape and v["follow_tangent"] == follow
                           for v in result["voxel"]):
                        continue
                    c, info = voxel_reference(shape, geom, material, follow)
                    result["voxel"].append(record(c, info, material=material, follow_tangent=follow))
                    path.write_text(json.dumps(result, indent=2) + "\n")
                    print(f"FE {shape} {material} tangent={follow}: {info['total_seconds']:.2f}s; "
                          f"iterations={info['iterations']}; Ex={metrics(c)['Ex']:.6f}", flush=True)
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()
