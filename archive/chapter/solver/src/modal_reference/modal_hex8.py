"""Exact HG/FI modal CPU prototype, periodic matrix-free assembly and PCG.

Material/shape conventions match the FEAD paper. No GPU dependency is required.
Array order is (z,y,x), local corners use binary x + 2*y + 4*z.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np
from numba import njit, prange, get_num_threads
from scipy import sparse
from scipy.sparse.linalg import splu

import generated_kernels as kernels


FEAD = Path(__file__).resolve().parents[2]
PAIRS = [(i, j) for i in range(6) for j in range(i, 6)]
SIGNS = np.array([[1.0 if a & (1 << d) else -1.0 for d in range(3)] for a in range(8)])
OFFSETS = ((SIGNS + 1) / 2).astype(np.int64)
SOURCE_GEOMETRY = FEAD / "repro/source/femlib/geometry.py"
SOURCE_MATERIALS = FEAD / "repro/source/femlib/materials.py"


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = source_module("frozen_fead_geometry", SOURCE_GEOMETRY)
materials = source_module("frozen_fead_materials", SOURCE_MATERIALS)


def b_matrices(sizes, kind):
    b, weight = geometry.make_B_matrices(*sizes, element_type="C3D8" if kind == "fi" else "C3D8R")
    permutation = [0, 1, 3, 2, 4, 5, 7, 6]
    columns = [3 * node + d for node in permutation for d in range(3)]
    return np.ascontiguousarray(b[:, :, columns]), weight


def unpack(c):
    matrix = np.empty(c.shape[:-1] + (6, 6))
    for n, (i, j) in enumerate(PAIRS):
        matrix[..., i, j] = matrix[..., j, i] = c[..., n]
    return matrix


def pack(c):
    return np.ascontiguousarray(np.stack([c[..., i, j] for i, j in PAIRS], axis=-1))


@njit(cache=True, parallel=True, fastmath=False)
def gather(forces, shape, output):
    nz, ny, nx = shape
    for node_index in prange(nz * ny * nx):
        node = np.int64(node_index)
        z, rem = divmod(node, ny * nx)
        y, x = divmod(rem, nx)
        for d in range(3):
            value = 0.0
            for a in range(8):
                ex, ey, ez = (x-(a&1)) % nx, (y-((a>>1)&1)) % ny, (z-((a>>2)&1)) % nz
                element = (ez * ny + ey) * nx + ex
                value += forces[element, 3*a+d]
            output[node, d] = value


@njit(cache=True, parallel=True, fastmath=False)
def prepare_rhs_and_diagonal(c, scales, shape, b, b0, weight, volume, macro,
                             rhs, diagonal, is_hg, compute_diagonal):
    """Node-owned setup: no global N*24 RHS or diagonal intermediates."""
    nz, ny, nx = shape
    active = np.array([[0, 4, 5], [1, 3, 5], [2, 3, 4]], dtype=np.int64)
    for node_index in prange(len(c)):
        node = np.int64(node_index)
        z, rem = divmod(node, ny*nx)
        y, x = divmod(rem, nx)
        for d in range(3):
            rhs[node, d] = 0.0
            if compute_diagonal:
                diagonal[node, d] = 0.0
        material = np.empty((6, 6))
        stress = np.empty(6)
        for corner in range(8):
            ex, ey, ez = (x-(corner&1)) % nx, (y-((corner>>1)&1)) % ny, (z-((corner>>2)&1)) % nz
            element = (ez*ny+ey)*nx+ex
            offset = 0
            for i in range(6):
                for j in range(i, 6):
                    material[i, j] = material[j, i] = c[element, offset]
                    offset += 1
            for i in range(6):
                stress[i] = 0.0
                for j in range(6):
                    stress[i] += material[i, j]*macro[j]
            for d in range(3):
                column = 3*corner+d
                for i in range(3):
                    row = active[d, i]
                    rhs[node, d] -= volume*b0[row, column]*stress[row]
                if compute_diagonal:
                    value = 0.0
                    for gp in range(len(b)):
                        for i in range(3):
                            row = active[d, i]
                            for j in range(3):
                                other = active[d, j]
                                value += weight*b[gp, row, column]*material[row, other]*b[gp, other, column]
                    diagonal[node, d] += value+(0.5*scales[element] if is_hg else 0.0)


@njit(cache=True, parallel=True, fastmath=False)
def average_stress(c, u, shape, b0, macro, stresses):
    nz, ny, nx = shape
    for element_index in prange(len(c)):
        e = np.int64(element_index)
        z, rem = divmod(e, ny * nx)
        y, x = divmod(rem, nx)
        strain = macro.copy()
        for a in range(8):
            node = (((z+((a>>2)&1)) % nz) * ny + ((y+((a>>1)&1)) % ny)) * nx + ((x+(a&1)) % nx)
            for row in range(6):
                for d in range(3):
                    strain[row] += b0[row, 3*a+d] * u[node, d]
        offset = 0
        for row in range(6):
            stresses[e, row] = 0.0
        for i in range(6):
            for j in range(i, 6):
                stresses[e, i] += c[e, offset] * strain[j]
                if j != i:
                    stresses[e, j] += c[e, offset] * strain[i]
                offset += 1


def zero_mean(u):
    return u - u.mean(axis=0, keepdims=True)


@dataclass
class PeriodicHex8Operator:
    c: np.ndarray
    shape: tuple[int, int, int]
    sizes: tuple[float, float, float]
    kind: str = "fi"
    arithmetic: str = "modal"
    assembly: str = "twostage"
    alpha: float = 0.1
    tile: int = 8

    def __post_init__(self):
        if self.kind not in ("hg", "fi") or self.arithmetic not in ("reference", "modal"):
            raise ValueError((self.kind, self.arithmetic))
        if self.assembly not in ("twostage", "tiled") or self.tile < 1:
            raise ValueError((self.assembly, self.tile))
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.shape = tuple(int(n) for n in self.shape)
        self.sizes = tuple(float(h) for h in self.sizes)
        if (len(self.shape) != 3 or min(self.shape) < 2 or len(self.sizes) != 3
                or not np.isfinite(self.sizes).all() or min(self.sizes) <= 0):
            raise ValueError("Require three periodic grid sizes >= 2 and positive voxel sizes")
        self.c = np.ascontiguousarray(self.c, dtype=np.float64)
        if self.c.shape != (int(np.prod(self.shape)), 21) or not np.isfinite(self.c).all():
            raise ValueError("C21 must be finite and have shape (N_voxel,21)")
        self.g = np.array([1/(4*h) for h in self.sizes] + [np.prod(self.sizes)])
        diagonal_sum = self.c[:, 0] + self.c[:, 6] + self.c[:, 11]
        invariant1 = diagonal_sum + 2*(self.c[:, 1] + self.c[:, 2] + self.c[:, 7])
        invariant2 = diagonal_sum + 2*(self.c[:, 15] + self.c[:, 18] + self.c[:, 20])
        self.scales = np.ascontiguousarray(self.alpha * np.maximum((invariant1+7*invariant2)/60, 1e-12)
                                         * np.prod(self.sizes) / np.mean(self.sizes)**2)
        self.forces = np.empty((len(self.c), 24)) if self.assembly == "twostage" else None
        prefix = f"{self.kind}_{self.arithmetic}"
        self.element_pass = getattr(kernels, prefix + "_element_pass")
        self.tile_pass = getattr(kernels, prefix + "_tile_pass")
        self.b, self.weight = b_matrices(self.sizes, self.kind)
        self.b0 = b_matrices(self.sizes, "hg")[0][0]
        self.diagonal = None

    @property
    def name(self):
        return f"{self.kind}_{self.arithmetic}_{self.assembly}"

    def apply(self, u, out=None):
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (len(self.c), 3) or not u.flags.c_contiguous:
            raise ValueError("u must be contiguous (N_voxel,3) FP64")
        if out is None:
            out = np.empty_like(u)
        if out.shape != u.shape or out.dtype != np.float64 or not out.flags.c_contiguous or np.shares_memory(out, u):
            raise ValueError("out must be a distinct contiguous FP64 array with the same shape")
        if self.assembly == "twostage":
            self.element_pass(u, self.c, self.scales, self.shape, self.g, self.forces)
            gather(self.forces, self.shape, out)
        else:
            self.tile_pass(u, self.c, self.scales, self.shape, self.g, out, self.tile)
        return out

    def rhs(self, macro):
        result = np.empty((len(self.c), 3))
        compute_diagonal = self.diagonal is None
        if compute_diagonal:
            self.diagonal = np.empty_like(result)
        prepare_rhs_and_diagonal(self.c, self.scales, self.shape, self.b, self.b0,
                                 self.weight, self.g[3], np.ascontiguousarray(macro, dtype=float),
                                 result, self.diagonal, self.kind == "hg", compute_diagonal)
        if np.min(self.diagonal) <= 0:
            raise ValueError("Nonpositive Jacobi diagonal")
        return np.ascontiguousarray(zero_mean(result))

    def recover(self, u, macro):
        stresses = np.empty((len(self.c), 6))
        average_stress(self.c, u, self.shape, self.b0, np.ascontiguousarray(macro), stresses)
        return stresses.mean(axis=0)

    def storage(self, threads=None):
        threads = get_num_threads() if threads is None else threads
        b = min(self.tile, max(self.shape))
        return dict(material_bytes=self.c.nbytes, scale_bytes=self.scales.nbytes,
                    global_force_buffer_bytes=0 if self.forces is None else self.forces.nbytes,
                    tile_scratch_upper_bound_bytes=(threads*((b+1)**3*24+24)*8 if self.forces is None else 0),
                    excludes="input/output/PCG vectors, setup temporaries, JIT/runtime, allocator retention")


def pcg(op, rhs, tolerance=1e-10, max_iterations=5000):
    if op.diagonal is None:
        raise ValueError("Call rhs() to build the common Jacobi preconditioner first")
    x = np.zeros_like(rhs)
    norm = np.linalg.norm(rhs)
    if norm == 0:
        return x, dict(iterations=0, true_residual=0.0)
    r = rhs.copy()
    z = zero_mean(r/op.diagonal)
    p = z.copy()
    rho = float(np.vdot(r, z))
    q = np.empty_like(rhs)
    for iteration in range(1, max_iterations+1):
        op.apply(p, out=q)
        denominator = float(np.vdot(p, q))
        if denominator <= 0:
            raise RuntimeError("PCG detected a nonpositive search curvature")
        alpha = rho/denominator
        x += alpha*p
        r -= alpha*q
        if np.linalg.norm(r) <= tolerance*norm:
            x = np.ascontiguousarray(zero_mean(x))
            true_r = zero_mean(rhs-op.apply(x))
            true = float(np.linalg.norm(true_r)/norm)
            if true <= tolerance:
                return x, dict(iterations=iteration, true_residual=true)
            r = true_r
            z = zero_mean(r/op.diagonal)
            p = z.copy()
            rho = float(np.vdot(r, z))
            continue
        z = zero_mean(r/op.diagonal)
        updated = float(np.vdot(r, z))
        p *= updated/rho
        p += z
        rho = updated
    raise RuntimeError("PCG did not meet the recomputed true residual threshold")


def weave_material(n):
    """Paper appendix C geometry, strong material_control.STRONG constants (MPa)."""
    height, width, thickness, amplitude = 0.4, 0.18, 0.035, 0.09
    matrix = materials.isotropic_to_voigt(3500.0, 0.35)
    yarn = materials.engineering_to_voigt(135000.0, 9000.0, 9000.0, 0.30, 0.30, 0.35, 5000.0, 5000.0, 3500.0)
    field = np.empty((n, n, n, 21))
    phase = np.full((n, n, n), -1, dtype=np.int8)
    y, x = np.meshgrid((np.arange(n)+0.5)/n, (np.arange(n)+0.5)/n, indexing="ij")
    for k in range(n):
        z = height*((k+0.5)/n-0.5)
        field[k] = pack(matrix)
        for yarn_id, (axis, center, sign) in enumerate([(0, 0.0, 1), (0, 0.5, -1), (1, 0.0, -1), (1, 0.5, 1)]):
            axial, transverse = (x, y) if axis == 0 else (y, x)
            dt = transverse-center
            dt -= np.floor(dt+0.5)
            dz = z-sign*amplitude*np.cos(2*np.pi*axial)
            dz -= height*np.floor(dz/height+0.5)
            mask = (dt/width)**2+(dz/thickness)**2 < 1
            if np.any(phase[k][mask] >= 0):
                raise ValueError("Overlapping yarn voxels")
            count = int(mask.sum())
            if count:
                direction = np.zeros((count, 3))
                direction[:, axis] = 1
                direction[:, 2] = -2*np.pi*sign*amplitude*np.sin(2*np.pi*axial[mask])
                transverse_direction = np.zeros_like(direction)
                transverse_direction[:, 1-axis] = 1 if axis == 0 else -1
                rotation = materials.build_rotation_matrices(direction, transverse_direction)
                field[k][mask] = pack(materials.rotate_voigt_batch(yarn, rotation))
                phase[k][mask] = yarn_id
    c = field.reshape(-1, 21)
    return c, (1/n, 1/n, height/n), dict(
        geometry="sections_structured_zh/appendix_resolution_geometry.tex",
        material="repro/extension_20261006/material_control.py:STRONG",
        domain_size_xyz=[1.0, 1.0, height], voxel_size_xyz=[1/n, 1/n, height/n],
        phase_sha256=hashlib.sha256(phase.tobytes()).hexdigest(),
        c21_sha256=hashlib.sha256(memoryview(c).cast("B")).hexdigest(),
        yarn_voxel_fraction=float((phase >= 0).mean()))


def independent_csr(c, shape, sizes, kind, scales):
    """Trusted tiny-grid oracle from frozen conventional B matrices, not modal code."""
    n = int(np.prod(shape))
    b, weight = b_matrices(sizes, kind)
    high = np.array([np.prod(SIGNS[:, [d for d in range(3) if m & (1 << d)]], axis=1)
                     for m in (3, 5, 6, 7)]).T/np.sqrt(8.0)
    projector = np.kron(high@high.T, np.eye(3))
    rows, cols, vals = [], [], []
    rhs = np.zeros((3*n, 6))
    for element in range(n):
        z, rem = divmod(element, shape[1]*shape[2])
        y, x = divmod(rem, shape[2])
        nodes = [((((z+oz) % shape[0])*shape[1]+((y+oy) % shape[1]))*shape[2]+((x+ox) % shape[2]))
                 for ox, oy, oz in OFFSETS]
        dofs = np.array([3*node+d for node in nodes for d in range(3)])
        ce = unpack(c[element])
        ke = sum(weight*bg.T@ce@bg for bg in b)
        if kind == "hg":
            ke += scales[element]*projector
        local_rhs = -sum(weight*bg.T@ce for bg in b)
        rhs[dofs] += local_rhs
        rows.extend(np.repeat(dofs, 24))
        cols.extend(np.tile(dofs, 24))
        vals.extend(ke.ravel())
    assembled = sparse.coo_matrix((vals, (rows, cols)), shape=(3*n, 3*n)).tocsr()
    factor = splu(assembled[3:, 3:].tocsc())
    solution = np.zeros((3*n, 6))
    solution[3:] = factor.solve(rhs[3:])
    solution = solution.reshape(n, 3, 6)
    solution -= solution.mean(axis=0, keepdims=True)
    return assembled, rhs, solution
