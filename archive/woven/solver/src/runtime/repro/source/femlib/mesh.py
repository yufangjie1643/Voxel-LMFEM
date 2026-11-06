# -*- coding: utf-8 -*-
"""
TexGen voxel-mesh reader and Voigt stiffness-field assembly for FEM backends.
"""

import time
import numpy as np

from .element_types import merge_input_element_type, parse_abaqus_element_type
from .materials import (
    isotropic_to_voigt,
    engineering_to_voigt,
    rotate_voigt_batch,
    build_rotation_matrices,
)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Upper-triangular storage order for a symmetric 6x6 Voigt matrix.  The
# packed field is the sole persistent constitutive representation used by the
# HG candidate; kernels reconstruct lower-triangular entries from these 21
# values.  Keeping this order explicit also makes the layout auditable.
_PACKED_VOIGT_PAIRS = tuple(
    (row, col) for row in range(6) for col in range(row, 6)
)
_PACKED_VOIGT_INDEX = {
    pair: index for index, pair in enumerate(_PACKED_VOIGT_PAIRS)
}


# ============================================================
# TexGen mesh reader
# ============================================================
class TexGenMesh:
    """Reader for a TexGen voxel mesh (.inp / .eld / .ori)."""

    def __init__(self, inp_path, eld_path, ori_path, verbose=True):
        self.verbose = verbose
        if verbose:
            print("[Mesh] Reading TexGen mesh ...")
        t0 = time.time()
        self._read_inp(inp_path)
        self._read_eld(eld_path)
        self._read_ori(ori_path)
        if verbose:
            print(f"[Mesh] grid: {self.Nx} x {self.Ny} x {self.Nz} = {self.n_elements} elements")
            print(f"[Mesh] domain size (Lx, Ly, Lz): {self.domain_size}")
            print(f"[Mesh] voxel size (dx, dy, dz): {self.voxel_size}")
            print(f"[Mesh] read time: {time.time() - t0:.1f}s")

    def _read_inp(self, path):
        n_elements = 0
        element_type = None
        all_x, all_y, all_z = set(), set(), set()
        in_nodes = False
        with open(path, 'r') as f:
            for line in f:
                ls = line.strip()
                declared_type = parse_abaqus_element_type(ls)
                if declared_type is not None:
                    element_type = merge_input_element_type(element_type, declared_type)
                if ls.lower() == '*node':
                    in_nodes = True
                    continue
                if in_nodes:
                    if ls.startswith('*'):
                        in_nodes = False
                    else:
                        parts = ls.split(',')
                        all_x.add(round(float(parts[1]), 8))
                        all_y.add(round(float(parts[2]), 8))
                        all_z.add(round(float(parts[3]), 8))
                        continue
                if 'ElSet=AllElements, Generate' in ls:
                    nl = next(f).strip()
                    n_elements = int(nl.split(',')[1].strip())
        xs, ys, zs = sorted(all_x), sorted(all_y), sorted(all_z)
        self.element_type = element_type or "C3D8"
        self.Nx, self.Ny, self.Nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
        self.n_elements = n_elements
        Lx, Ly, Lz = xs[-1] - xs[0], ys[-1] - ys[0], zs[-1] - zs[0]
        self.domain_size = (Lx, Ly, Lz)
        self.voxel_size = (Lx / self.Nx, Ly / self.Ny, Lz / self.Nz)
        # Array layout: axis 0 = z, axis 1 = y, axis 2 = x
        self.grid_shape = (self.Nz, self.Ny, self.Nx)
        if self.Nx * self.Ny * self.Nz != n_elements:
            raise ValueError(
                f"[Mesh] grid dimensions disagree with element count: {self.Nx}×{self.Ny}×{self.Nz} != {n_elements}"
            )

    def _read_eld(self, path):
        yarn_ids = []
        with open(path, 'r') as f:
            for line in f:
                ls = line.strip()
                if ls.startswith('*') or ls.startswith('**') or len(ls) == 0:
                    continue
                parts = ls.split(',')
                try:
                    int(parts[0].strip())
                    yarn_ids.append(int(parts[1].strip()))
                except (ValueError, IndexError):
                    continue
        if len(yarn_ids) != self.n_elements:
            raise ValueError(
                f"[Mesh] .eld element count {len(yarn_ids)} disagrees with .inp element count {self.n_elements}"
            )
        self.yarn_id = np.array(yarn_ids, dtype=np.int32).reshape(self.grid_shape)
        if self.verbose:
            n_mat = int(np.sum(self.yarn_id == -1))
            n_yarn = int(np.sum(self.yarn_id >= 0))
            unique_yarns = sorted(set(np.unique(self.yarn_id).tolist()) - {-1})
            print(f"[Mesh] matrix elements: {n_mat}, yarn elements: {n_yarn}, yarn ids: {unique_yarns}")

    def _read_ori(self, path):
        orient1, orient2 = [], []
        self.default_orient = (np.array([1., 0., 0.]), np.array([0., 1., 0.]))
        with open(path, 'r') as f:
            for line in f:
                ls = line.strip()
                if ls.startswith('*') or len(ls) == 0:
                    continue
                parts = [p.strip() for p in ls.split(',')]
                try:
                    if parts[0] == '':
                        self.default_orient = (
                            np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                            np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                        )
                    else:
                        int(parts[0])
                        orient1.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        orient2.append([float(parts[4]), float(parts[5]), float(parts[6])])
                except (ValueError, IndexError):
                    continue
        if len(orient1) != self.n_elements:
            raise ValueError(
                f"[Mesh] .ori element count {len(orient1)} disagrees with .inp element count {self.n_elements}"
            )
        self.orient1 = np.array(orient1).reshape(self.grid_shape + (3,))
        self.orient2 = np.array(orient2).reshape(self.grid_shape + (3,))


# ============================================================
# Stiffness-field assembly (Voigt, for FEM)
# ============================================================
def build_voigt_stiffness_field(mesh, mat0_params, mat1_params,
                                use_orientations=True,
                                force_isotropic_yarn=False,
                                verbose=True):
    """
    Counterpart of the FFT-side build_stiffness_field, but returning Voigt stiffness (not Mandel).
    Returns
    -------
    C_voigt_field : (6, 6, Nz, Ny, Nx) numpy array
    """
    Nz, Ny, Nx = mesh.grid_shape

    C_mat = isotropic_to_voigt(mat0_params['E'], mat0_params['Nu'])
    if force_isotropic_yarn or 'E11' not in mat1_params:
        C_yarn_local = isotropic_to_voigt(mat1_params['E'], mat1_params['Nu'])
    else:
        C_yarn_local = engineering_to_voigt(
            mat1_params['E11'], mat1_params['E22'], mat1_params['E33'],
            mat1_params['Nu12'], mat1_params['Nu13'], mat1_params['Nu23'],
            mat1_params['G12'], mat1_params['G13'], mat1_params['G23'],
        )

    C_field = np.empty((6, 6, Nz, Ny, Nx), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            C_field[i, j].fill(C_mat[i, j])

    yarn_mask = mesh.yarn_id >= 0
    if not np.any(yarn_mask):
        if verbose:
            print("[FEM/Stiffness] no yarn elements found in mesh")
        return C_field

    if use_orientations and not force_isotropic_yarn and 'E11' in mat1_params:
        idx = np.where(yarn_mask)
        o1 = mesh.orient1[idx]
        o2 = mesh.orient2[idx]
        R = build_rotation_matrices(o1, o2)
        C_yarn = rotate_voigt_batch(C_yarn_local, R)
        zz, yy, xx = idx
        for I in range(6):
            for J in range(6):
                C_field[I, J, zz, yy, xx] = C_yarn[:, I, J]
    else:
        for I in range(6):
            for J in range(6):
                C_field[I, J][yarn_mask] = C_yarn_local[I, J]

    if verbose:
        print(f"[FEM/Stiffness] matrix elements: {int(np.sum(~yarn_mask))}, "
              f"yarn elements: {int(np.sum(yarn_mask))}, "
              f"use_orientations={use_orientations}, force_iso={force_isotropic_yarn}")
    return C_field


# ============================================================
# Batched GPU Voigt stiffness-field assembly
# ============================================================
def build_voigt_stiffness_fields_gpu(meshes, mat0_params, mat1_params, *,
                                     use_orientations=True,
                                     force_isotropic_yarn=False,
                                     device='cuda',
                                     dtype=None,
                                     verbose=False):
    """Batched GPU version of `build_voigt_stiffness_field`.

    Parameters
    ----------
    meshes : list of objects with attrs `yarn_id (Nz,Ny,Nx)`,
             `orient1 (Nz,Ny,Nx,3)`, `orient2 (Nz,Ny,Nx,3)`, `grid_shape`.
             Accepts numpy arrays or tensors already on GPU.
    mat0_params, mat1_params : same as the single-sample version (matrix / yarn).
    dtype  : torch dtype, default torch.float64.

    Returns
    -------
    C_field_gpu : torch.Tensor (B, N, 21)  upper-triangular symmetric Voigt
        stiffness, on `device`.  The 21 entries are ordered by
        ``(0,0),(0,1),...,(0,5),(1,1),...,(5,5)``.  This packed form is
        intentionally canonical: the HG operator and its CUDA kernels consume
        it directly rather than retaining a dense 36-entry view.
    """
    import torch  # optional dependency
    if dtype is None:
        dtype = torch.float64

    device = torch.device(device)
    Bsz = len(meshes)
    Nz, Ny, Nx = tuple(int(v) for v in meshes[0].grid_shape)

    # Keep the material mask on the host while assembling the packed field.
    # Selecting yarn orientations before transfer avoids the large dense
    # all-voxel orientation/rotation intermediates used by the legacy path.
    yarn_ids_np = np.stack(
        [np.asarray(getattr(mesh, 'yarn_id')) for mesh in meshes], axis=0,
    ).astype(np.int32, copy=False)
    yarn_mask_np = yarn_ids_np >= 0
    yarn_flat_np = yarn_mask_np.reshape(-1)
    yarn_flat_indices = np.flatnonzero(yarn_flat_np).astype(np.int64, copy=False)

    # ---- 2) matrix / yarn local stiffness (6,6) ----
    C_mat_np = isotropic_to_voigt(mat0_params['E'], mat0_params['Nu'])
    if force_isotropic_yarn or 'E11' not in mat1_params:
        C_yarn_local_np = isotropic_to_voigt(mat1_params['E'], mat1_params['Nu'])
        is_orthotropic = False
    else:
        C_yarn_local_np = engineering_to_voigt(
            mat1_params['E11'], mat1_params['E22'], mat1_params['E33'],
            mat1_params['Nu12'], mat1_params['Nu13'], mat1_params['Nu23'],
            mat1_params['G12'], mat1_params['G13'], mat1_params['G23'],
        )
        is_orthotropic = True
    C_mat = torch.tensor(C_mat_np, device=device, dtype=dtype)                   # (6,6)
    C_yarn_local = torch.tensor(C_yarn_local_np, device=device, dtype=dtype)     # (6,6)

    def _pack_dense(C):
        return torch.stack(
            [C[..., row, col] for row, col in _PACKED_VOIGT_PAIRS], dim=-1,
        )

    # ---- 3) initialise the one canonical field with matrix stiffness ----
    n_total = Bsz * Nz * Ny * Nx
    C_field_flat = C_mat.new_empty((n_total, len(_PACKED_VOIGT_PAIRS)))
    C_field_flat.copy_(C_mat.new_tensor(
        [C_mat_np[row, col] for row, col in _PACKED_VOIGT_PAIRS],
    ).expand(n_total, -1))

    # ---- 4) replace only yarn positions, without making a dense 6x6 field ----
    if yarn_flat_indices.size and use_orientations and is_orthotropic:
        def _yarn_orientation(attr):
            chunks = []
            for mesh, mask in zip(meshes, yarn_mask_np):
                values = np.asarray(getattr(mesh, attr))
                chunks.append(np.ascontiguousarray(values[mask]))
            return torch.from_numpy(np.concatenate(chunks, axis=0)).to(
                device=device, dtype=dtype,
            )

        orient1 = _yarn_orientation('orient1')
        orient2 = _yarn_orientation('orient2')
        C_yarn_packed = _rotate_voigt_packed_yarn_gpu(
            C_yarn_local, orient1, orient2,
        )
        yarn_indices_gpu = torch.from_numpy(yarn_flat_indices).to(device=device)
        C_field_flat.index_copy_(0, yarn_indices_gpu, C_yarn_packed)
    elif yarn_flat_indices.size:
        yarn_indices_gpu = torch.from_numpy(yarn_flat_indices).to(device=device)
        C_field_flat.index_copy_(
            0, yarn_indices_gpu,
            C_yarn_local.new_tensor(
                [C_yarn_local_np[row, col] for row, col in _PACKED_VOIGT_PAIRS],
            ).expand(yarn_flat_indices.size, -1),
        )

    C_field = C_field_flat.view(Bsz, Nz * Ny * Nx, len(_PACKED_VOIGT_PAIRS)).contiguous()

    if verbose:
        n_yarn = int(yarn_flat_indices.size)
        n_mat = Bsz * Nz * Ny * Nx - n_yarn
        print(f"[FEM/Stiffness-GPU] B={Bsz}, yarn voxels: {n_yarn}, matrix voxels: {n_mat}, "
              f"use_orientations={use_orientations}, force_iso={force_isotropic_yarn}, "
              f"layout=packed_symmetric_voigt21")
    return C_field


def _rotate_voigt_field_gpu(C_local, orient1, orient2, *, eps=1e-14):
    """Apply a Bond rotation to every voxel of the (B,Nz,Ny,Nx) grid.

    Parameters
    ----------
    C_local : (6, 6) torch   local Voigt stiffness
    orient1, orient2 : (B, Nz, Ny, Nx, 3) torch   local x axis / reference direction

    Returns
    -------
    C_rot : (B, Nz, Ny, Nx, 6, 6)
    """
    import torch
    e1 = orient1 / orient1.norm(dim=-1, keepdim=True).clamp(min=eps)
    e3 = torch.cross(e1, orient2, dim=-1)
    e3 = e3 / e3.norm(dim=-1, keepdim=True).clamp(min=eps)
    e2 = torch.cross(e3, e1, dim=-1)
    # R[..., :, k] = global components of the k-th local axis
    R = torch.stack([e1, e2, e3], dim=-1)                # (B,Nz,Ny,Nx,3,3)

    # Bond matrix T (..., 6, 6). Build the four quadrants fully vectorized.
    # SHEAR[K] = (i1, i2), K ∈ {0,1,2} → (1,2), (0,2), (0,1)
    i1 = torch.tensor([1, 0, 0], device=R.device)
    i2 = torch.tensor([2, 2, 1], device=R.device)

    # (B,Nz,Ny,Nx,3,3)
    TL = R * R                                            # T[I,J] = R[I,J]^2

    # Gather rows/cols in one shot. R_i1row[..., K, j] = R[..., i1[K], j]
    R_i1row = R.index_select(-2, i1)                      # (..., 3, 3)
    R_i2row = R.index_select(-2, i2)
    # R_i1col[..., i, K] = R[..., i, i1[K]]
    R_i1col = R.index_select(-1, i1)
    R_i2col = R.index_select(-1, i2)

    TR = 2.0 * R_i1col * R_i2col                          # T[I, 3+K]
    BL = R_i1row * R_i2row                                # T[3+K, J]

    # BR[..., K, L] = R[..., i1[K], i1[L]] * R[..., i2[K], i2[L]]
    #               + R[..., i1[K], i2[L]] * R[..., i2[K], i1[L]]
    R_i1i1 = R_i1row.index_select(-1, i1)
    R_i2i2 = R_i2row.index_select(-1, i2)
    R_i1i2 = R_i1row.index_select(-1, i2)
    R_i2i1 = R_i2row.index_select(-1, i1)
    BR = R_i1i1 * R_i2i2 + R_i1i2 * R_i2i1

    T = torch.cat([torch.cat([TL, TR], dim=-1),
                   torch.cat([BL, BR], dim=-1)], dim=-2)  # (..., 6, 6)

    # C_global = T @ C_local @ T.T
    return torch.einsum('...ij,jk,...lk->...il', T, C_local, T)


def _rotate_voigt_packed_yarn_gpu(C_local, orient1, orient2, *, eps=1e-14):
    """Rotate only yarn voxels and return their 21-entry symmetric field.

    Unlike ``_rotate_voigt_field_gpu``, this helper does not materialize a
    dense constitutive tensor for every voxel.  The temporary transformation
    matrix is restricted to yarn entries, and the final result is assembled
    directly in the canonical upper-triangular order.
    """
    import torch

    e1 = orient1 / orient1.norm(dim=-1, keepdim=True).clamp(min=eps)
    e3 = torch.cross(e1, orient2, dim=-1)
    e3 = e3 / e3.norm(dim=-1, keepdim=True).clamp(min=eps)
    e2 = torch.cross(e3, e1, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)

    i1 = torch.tensor([1, 0, 0], device=R.device)
    i2 = torch.tensor([2, 2, 1], device=R.device)
    TL = R * R
    R_i1row = R.index_select(-2, i1)
    R_i2row = R.index_select(-2, i2)
    R_i1col = R.index_select(-1, i1)
    R_i2col = R.index_select(-1, i2)
    TR = 2.0 * R_i1col * R_i2col
    BL = R_i1row * R_i2row
    R_i1i1 = R_i1row.index_select(-1, i1)
    R_i2i2 = R_i2row.index_select(-1, i2)
    R_i1i2 = R_i1row.index_select(-1, i2)
    R_i2i1 = R_i2row.index_select(-1, i1)
    BR = R_i1i1 * R_i2i2 + R_i1i2 * R_i2i1
    T = torch.cat([
        torch.cat([TL, TR], dim=-1),
        torch.cat([BL, BR], dim=-1),
    ], dim=-2)

    # Form one dense intermediate only over yarn entries, then immediately
    # reduce it into the packed output.  No dense field survives this helper.
    TC = torch.matmul(T, C_local)
    packed = torch.stack(
        [
            (TC[..., row, :] * T[..., col, :]).sum(dim=-1)
            for row, col in _PACKED_VOIGT_PAIRS
        ],
        dim=-1,
    )
    return packed.contiguous()


def _rotate_voigt_field_gpu_legacy(C_local, orient1, orient2, *, eps=1e-14):
    """Legacy nested-loop implementation, kept for A/B diff only."""
    import torch
    e1 = orient1 / orient1.norm(dim=-1, keepdim=True).clamp(min=eps)
    e3 = torch.cross(e1, orient2, dim=-1)
    e3 = e3 / e3.norm(dim=-1, keepdim=True).clamp(min=eps)
    e2 = torch.cross(e3, e1, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)

    SHEAR = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long, device=R.device)
    shape = R.shape[:-2]
    T = torch.zeros(shape + (6, 6), dtype=R.dtype, device=R.device)

    for I in range(3):
        for J in range(3):
            T[..., I, J] = R[..., I, J] ** 2
    for I in range(3):
        for K in range(3):
            j1, j2 = int(SHEAR[K, 0]), int(SHEAR[K, 1])
            T[..., I, 3 + K] = 2.0 * R[..., I, j1] * R[..., I, j2]
    for K in range(3):
        i1, i2 = int(SHEAR[K, 0]), int(SHEAR[K, 1])
        for J in range(3):
            T[..., 3 + K, J] = R[..., i1, J] * R[..., i2, J]
    for K in range(3):
        i1, i2 = int(SHEAR[K, 0]), int(SHEAR[K, 1])
        for L in range(3):
            j1, j2 = int(SHEAR[L, 0]), int(SHEAR[L, 1])
            T[..., 3 + K, 3 + L] = (R[..., i1, j1] * R[..., i2, j2]
                                    + R[..., i1, j2] * R[..., i2, j1])

    return torch.einsum('...ij,jk,...lk->...il', T, C_local, T)
