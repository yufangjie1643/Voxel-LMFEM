# -*- coding: utf-8 -*-
"""Low-level NumPy reference implementations for periodic element stencils.

This module is deliberately separate from the production CUDA/Triton path.  It
keeps the experiments small and inspectable while probing element families. Its
C3D8R hourglass path delegates to the shared stabilized C3D8R-HG definition;
this is reference code, not a high-level solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np

from ..c3d8r import make_c3d8r_element_stiffness, validate_hourglass_coefficient


Array = np.ndarray
GridShape = Tuple[int, int, int]
VoxelSize = Tuple[float, float, float]


_C3D8_REF = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

_C3D8_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int64,
)

_C3D20_REF = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [0.0, -1.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, -1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

@dataclass(frozen=True)
class PeriodicBlockStencil:
    """Sparse block-stencil for periodic structured grids.

    ``blocks[(row_node, col_node)]`` stores one 3x3 displacement-component
    block.  This is intentionally a prototype-friendly representation rather
    than a production storage layout.
    """

    element_type: str
    grid_shape: GridShape
    voxel_size: VoxelSize
    n_nodes: int
    n_dofs: int
    blocks: Dict[Tuple[int, int], Array]


def make_anisotropic_voigt_field(grid_shape: GridShape, seed: int = 0) -> Array:
    """Return a deterministic SPD anisotropic Voigt field.

    Shape is ``(Nz, Ny, Nx, 6, 6)``.  The matrices are symmetric positive
    definite and include normal-shear coupling, which is enough for stencil
    verification without relying on a specific material model.
    """

    Nz, Ny, Nx = map(int, grid_shape)
    rng = np.random.default_rng(seed)
    field = np.empty((Nz, Ny, Nx, 6, 6), dtype=np.float64)
    base_eigs = np.array([160.0, 120.0, 95.0, 42.0, 35.0, 28.0], dtype=np.float64)
    for z, y, x in product(range(Nz), range(Ny), range(Nx)):
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        scale = 1.0 + 0.04 * x + 0.03 * y + 0.02 * z
        jitter = 1.0 + 0.05 * rng.normal(size=6)
        eigs = base_eigs * scale * np.clip(jitter, 0.75, 1.25)
        C = q @ np.diag(eigs) @ q.T
        field[z, y, x] = 0.5 * (C + C.T)
    return field


def element_stiffness(
    element_type: str,
    C_voigt: Array,
    voxel_size: VoxelSize = (1.0, 1.0, 1.0),
    *,
    hourglass: bool = True,
    hourglass_coefficient: float = 0.1,
) -> Array:
    """Build one low-level reference stiffness matrix.

    For C3D8R, ``hourglass=False`` explicitly selects the raw reduced-
    integration reference matrix. ``hourglass=True`` selects stabilized
    C3D8R-HG and requires a finite positive coefficient.
    """

    element_type = element_type.upper()
    C = _as_voigt_matrix(C_voigt)
    if element_type == "C3D8R":
        K = _c3d8r_stiffness(C, voxel_size)
        if hourglass:
            coefficient = validate_hourglass_coefficient(hourglass_coefficient)
            K = K + _c3d8r_hourglass_stiffness(C, voxel_size, coefficient)
        return _sym(K)
    if element_type == "C3D8I":
        return _c3d8i_stiffness(C, voxel_size)
    if element_type == "C3D20":
        return _c3d20_stiffness(C, voxel_size)
    raise ValueError(f"unsupported element type: {element_type!r}")


def build_periodic_block_stencil(
    element_type: str,
    C_field: Array,
    *,
    voxel_size: VoxelSize = (1.0, 1.0, 1.0),
    hourglass_coefficient: float = 0.1,
) -> PeriodicBlockStencil:
    """Build a periodic block-stencil by accumulating element blocks."""

    element_type = element_type.upper()
    C = _as_field(C_field)
    grid_shape = tuple(map(int, C.shape[:3]))
    n_nodes = _node_count(element_type, grid_shape)
    blocks: Dict[Tuple[int, int], Array] = {}

    for z, y, x in _iter_cells(grid_shape):
        ids = _local_node_ids(element_type, z, y, x, grid_shape)
        Ke = element_stiffness(
            element_type,
            C[z, y, x],
            voxel_size,
            hourglass=True,
            hourglass_coefficient=hourglass_coefficient,
        )
        _accumulate_blocks(blocks, ids, Ke)

    return PeriodicBlockStencil(
        element_type=element_type,
        grid_shape=grid_shape,
        voxel_size=tuple(map(float, voxel_size)),
        n_nodes=n_nodes,
        n_dofs=3 * n_nodes,
        blocks=blocks,
    )


def stencil_apply(stencil: PeriodicBlockStencil, u: Array) -> Array:
    """Apply the block-stencil without assembling a dense matrix."""

    u_arr = np.asarray(u, dtype=np.float64)
    if u_arr.shape != (stencil.n_dofs,):
        raise ValueError(f"u must have shape {(stencil.n_dofs,)}, got {u_arr.shape}")
    out = np.zeros_like(u_arr)
    for (row, col), block in stencil.blocks.items():
        out[3 * row : 3 * row + 3] += block @ u_arr[3 * col : 3 * col + 3]
    return out


def element_apply(
    element_type: str,
    C_field: Array,
    u: Array,
    *,
    voxel_size: VoxelSize = (1.0, 1.0, 1.0),
    hourglass_coefficient: float = 0.1,
) -> Array:
    """Reference matrix-free apply using direct element gather/scatter."""

    element_type = element_type.upper()
    C = _as_field(C_field)
    grid_shape = tuple(map(int, C.shape[:3]))
    n_nodes = _node_count(element_type, grid_shape)
    n_dofs = 3 * n_nodes
    u_arr = np.asarray(u, dtype=np.float64)
    if u_arr.shape != (n_dofs,):
        raise ValueError(f"u must have shape {(n_dofs,)}, got {u_arr.shape}")

    out = np.zeros_like(u_arr)
    for z, y, x in _iter_cells(grid_shape):
        ids = _local_node_ids(element_type, z, y, x, grid_shape)
        Ke = element_stiffness(
            element_type,
            C[z, y, x],
            voxel_size,
            hourglass=True,
            hourglass_coefficient=hourglass_coefficient,
        )
        dofs = _node_ids_to_dofs(ids)
        out[dofs] += Ke @ u_arr[dofs]
    return out


def _c3d8r_stiffness(C: Array, voxel_size: VoxelSize) -> Array:
    return make_c3d8r_element_stiffness(C, voxel_size, hourglass_coefficient=0.0)


def _c3d8r_hourglass_stiffness(
    C: Array,
    voxel_size: VoxelSize,
    coefficient: float,
) -> Array:
    stabilized = make_c3d8r_element_stiffness(
        C, voxel_size, hourglass_coefficient=coefficient,
    )
    return stabilized - _c3d8r_stiffness(C, voxel_size)


def _c3d8i_stiffness(C: Array, voxel_size: VoxelSize) -> Array:
    Kuu = np.zeros((24, 24), dtype=np.float64)
    Kua = np.zeros((24, 9), dtype=np.float64)
    Kaa = np.zeros((9, 9), dtype=np.float64)
    detJ = _volume(voxel_size) / 8.0
    for gp, weight in _gauss_points(2):
        Bu = _B_from_ref_derivs(_shape8_derivs(gp), voxel_size)
        Ba = _B_internal_bubbles(gp, voxel_size)
        wdet = weight * detJ
        Kuu += Bu.T @ C @ Bu * wdet
        Kua += Bu.T @ C @ Ba * wdet
        Kaa += Ba.T @ C @ Ba * wdet

    reg = 1e-12 * max(float(np.trace(Kaa)) / Kaa.shape[0], 1.0)
    correction = Kua @ np.linalg.solve(Kaa + reg * np.eye(9), Kua.T)
    return _sym(Kuu - correction)


def _c3d20_stiffness(C: Array, voxel_size: VoxelSize) -> Array:
    K = np.zeros((60, 60), dtype=np.float64)
    detJ = _volume(voxel_size) / 8.0
    for gp, weight in _gauss_points(3):
        B = _B_from_ref_derivs(_shape20_derivs(gp), voxel_size)
        K += B.T @ C @ B * weight * detJ
    return _sym(K)


def _B_from_ref_derivs(dN_ref: Array, voxel_size: VoxelSize) -> Array:
    dN = np.asarray(dN_ref, dtype=np.float64)
    nnode = dN.shape[0]
    hx, hy, hz = map(float, voxel_size)
    dNx = dN[:, 0] * (2.0 / hx)
    dNy = dN[:, 1] * (2.0 / hy)
    dNz = dN[:, 2] * (2.0 / hz)
    B = np.zeros((6, 3 * nnode), dtype=np.float64)
    for node in range(nnode):
        c = 3 * node
        B[0, c + 0] = dNx[node]
        B[1, c + 1] = dNy[node]
        B[2, c + 2] = dNz[node]
        B[3, c + 1] = dNz[node]
        B[3, c + 2] = dNy[node]
        B[4, c + 0] = dNz[node]
        B[4, c + 2] = dNx[node]
        B[5, c + 0] = dNy[node]
        B[5, c + 1] = dNx[node]
    return B


def _B_internal_bubbles(gp: Iterable[float], voxel_size: VoxelSize) -> Array:
    xi, eta, zeta = map(float, gp)
    hx, hy, hz = map(float, voxel_size)
    dpsi = np.array(
        [
            [-2.0 * xi * (2.0 / hx), 0.0, 0.0],
            [0.0, -2.0 * eta * (2.0 / hy), 0.0],
            [0.0, 0.0, -2.0 * zeta * (2.0 / hz)],
        ],
        dtype=np.float64,
    )
    B = np.zeros((6, 9), dtype=np.float64)
    for comp in range(3):
        for mode in range(3):
            col = 3 * comp + mode
            dx, dy, dz = dpsi[mode]
            if comp == 0:
                B[0, col] = dx
                B[4, col] = dz
                B[5, col] = dy
            elif comp == 1:
                B[1, col] = dy
                B[3, col] = dz
                B[5, col] = dx
            else:
                B[2, col] = dz
                B[3, col] = dy
                B[4, col] = dx
    return B


def _shape8_derivs(gp: Iterable[float]) -> Array:
    xi, eta, zeta = map(float, gp)
    out = np.zeros((8, 3), dtype=np.float64)
    for node, (rx, ry, rz) in enumerate(_C3D8_REF):
        out[node, 0] = 0.125 * rx * (1.0 + ry * eta) * (1.0 + rz * zeta)
        out[node, 1] = 0.125 * ry * (1.0 + rx * xi) * (1.0 + rz * zeta)
        out[node, 2] = 0.125 * rz * (1.0 + rx * xi) * (1.0 + ry * eta)
    return out


def _shape20_derivs(gp: Iterable[float]) -> Array:
    xi, eta, zeta = map(float, gp)
    out = np.zeros((20, 3), dtype=np.float64)

    for node, (rx, ry, rz) in enumerate(_C3D20_REF[:8]):
        A = 1.0 + rx * xi
        B = 1.0 + ry * eta
        C = 1.0 + rz * zeta
        D = rx * xi + ry * eta + rz * zeta - 2.0
        out[node, 0] = 0.125 * rx * B * C * (D + A)
        out[node, 1] = 0.125 * ry * A * C * (D + B)
        out[node, 2] = 0.125 * rz * A * B * (D + C)

    for node, (rx, ry, rz) in enumerate(_C3D20_REF[8:], start=8):
        if rx == 0.0:
            out[node, 0] = -0.5 * xi * (1.0 + ry * eta) * (1.0 + rz * zeta)
            out[node, 1] = 0.25 * (1.0 - xi * xi) * ry * (1.0 + rz * zeta)
            out[node, 2] = 0.25 * (1.0 - xi * xi) * (1.0 + ry * eta) * rz
        elif ry == 0.0:
            out[node, 0] = 0.25 * rx * (1.0 - eta * eta) * (1.0 + rz * zeta)
            out[node, 1] = -0.5 * eta * (1.0 + rx * xi) * (1.0 + rz * zeta)
            out[node, 2] = 0.25 * (1.0 + rx * xi) * (1.0 - eta * eta) * rz
        else:
            out[node, 0] = 0.25 * rx * (1.0 + ry * eta) * (1.0 - zeta * zeta)
            out[node, 1] = 0.25 * (1.0 + rx * xi) * ry * (1.0 - zeta * zeta)
            out[node, 2] = -0.5 * zeta * (1.0 + rx * xi) * (1.0 + ry * eta)
    return out


def _gauss_points(order: int) -> Iterable[Tuple[Array, float]]:
    if order == 2:
        pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
        weights = np.ones(2, dtype=np.float64)
    elif order == 3:
        pts = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=np.float64)
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    else:
        raise ValueError(f"unsupported Gauss order: {order}")
    for ix, iy, iz in product(range(order), range(order), range(order)):
        gp = np.array([pts[ix], pts[iy], pts[iz]], dtype=np.float64)
        yield gp, float(weights[ix] * weights[iy] * weights[iz])


def _local_node_ids(
    element_type: str,
    z: int,
    y: int,
    x: int,
    grid_shape: GridShape,
) -> Array:
    if element_type in {"C3D8R", "C3D8I"}:
        return np.array(
            [_corner_id(z + dz, y + dy, x + dx, grid_shape) for dx, dy, dz in _C3D8_OFFSETS],
            dtype=np.int64,
        )
    if element_type == "C3D20":
        ids = [
            _corner_id(z, y, x, grid_shape),
            _corner_id(z, y, x + 1, grid_shape),
            _corner_id(z, y + 1, x + 1, grid_shape),
            _corner_id(z, y + 1, x, grid_shape),
            _corner_id(z + 1, y, x, grid_shape),
            _corner_id(z + 1, y, x + 1, grid_shape),
            _corner_id(z + 1, y + 1, x + 1, grid_shape),
            _corner_id(z + 1, y + 1, x, grid_shape),
            _edge_id(1, z, y, x, grid_shape),
            _edge_id(2, z, y, x + 1, grid_shape),
            _edge_id(1, z, y + 1, x, grid_shape),
            _edge_id(2, z, y, x, grid_shape),
            _edge_id(1, z + 1, y, x, grid_shape),
            _edge_id(2, z + 1, y, x + 1, grid_shape),
            _edge_id(1, z + 1, y + 1, x, grid_shape),
            _edge_id(2, z + 1, y, x, grid_shape),
            _edge_id(3, z, y, x, grid_shape),
            _edge_id(3, z, y, x + 1, grid_shape),
            _edge_id(3, z, y + 1, x + 1, grid_shape),
            _edge_id(3, z, y + 1, x, grid_shape),
        ]
        return np.array(ids, dtype=np.int64)
    raise ValueError(f"unsupported element type: {element_type!r}")


def _corner_id(z: int, y: int, x: int, grid_shape: GridShape) -> int:
    Nz, Ny, Nx = grid_shape
    return ((z % Nz) * Ny + (y % Ny)) * Nx + (x % Nx)


def _edge_id(family: int, z: int, y: int, x: int, grid_shape: GridShape) -> int:
    Nz, Ny, Nx = grid_shape
    n = Nz * Ny * Nx
    return family * n + _corner_id(z, y, x, grid_shape)


def _node_count(element_type: str, grid_shape: GridShape) -> int:
    Nz, Ny, Nx = grid_shape
    n_elem = Nz * Ny * Nx
    if element_type in {"C3D8R", "C3D8I"}:
        return n_elem
    if element_type == "C3D20":
        return 4 * n_elem
    raise ValueError(f"unsupported element type: {element_type!r}")


def _accumulate_blocks(blocks: Dict[Tuple[int, int], Array], ids: Array, Ke: Array) -> None:
    n = len(ids)
    for a in range(n):
        a_slice = slice(3 * a, 3 * a + 3)
        for b in range(n):
            b_slice = slice(3 * b, 3 * b + 3)
            key = (int(ids[a]), int(ids[b]))
            if key not in blocks:
                blocks[key] = np.zeros((3, 3), dtype=np.float64)
            blocks[key] += Ke[a_slice, b_slice]


def _node_ids_to_dofs(ids: Array) -> Array:
    dofs = np.empty(3 * len(ids), dtype=np.int64)
    for i, node in enumerate(ids):
        dofs[3 * i : 3 * i + 3] = [3 * int(node), 3 * int(node) + 1, 3 * int(node) + 2]
    return dofs


def _iter_cells(grid_shape: GridShape) -> Iterable[Tuple[int, int, int]]:
    Nz, Ny, Nx = grid_shape
    return product(range(Nz), range(Ny), range(Nx))


def _as_voigt_matrix(C: Array) -> Array:
    arr = np.asarray(C, dtype=np.float64)
    if arr.shape != (6, 6):
        raise ValueError(f"C_voigt must have shape (6, 6), got {arr.shape}")
    return 0.5 * (arr + arr.T)


def _as_field(C_field: Array) -> Array:
    arr = np.asarray(C_field, dtype=np.float64)
    if arr.ndim != 5 or arr.shape[-2:] != (6, 6):
        raise ValueError(f"C_field must have shape (Nz, Ny, Nx, 6, 6), got {arr.shape}")
    return arr


def _volume(voxel_size: VoxelSize) -> float:
    hx, hy, hz = map(float, voxel_size)
    return hx * hy * hz


def _sym(K: Array) -> Array:
    return 0.5 * (K + K.T)
