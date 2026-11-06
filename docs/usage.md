# Using Voxel-LMFEM

Run the following commands from the repository root after creating the uv environment described in docs/installation.md.

## First result without a GPU

    uv run --no-sync voxel-lmfem example --out runs/example
    uv run --no-sync voxel-lmfem solve --input runs/example/homogeneous.npz --materials runs/example/materials.json --backend cpu --output runs/example/result.json

The example uses the matrix properties from Table 1 and an illustrative unit cube. Its exact effective stiffness equals the constituent stiffness. Use a new output path for each run. The equivalent Python example is examples/homogeneous.py.

## Python interface

    import numpy as np
    from voxel_lmfem import homogenize

    phase = np.zeros((4, 4, 4), dtype=np.uint8)
    materials = {0: {"kind": "isotropic", "E": 3500.0, "nu": 0.35}}
    result = homogenize(phase, materials, [1.0, 1.0, 1.0], backend="cpu")
    C_H = np.array(result["CH"])

Use backend="cuda" for the GPU solver. warmup=True performs a full same-grid warm-up before the returned workflow time. The CPU backend is a small independent verification implementation. It uses the exact zero-fluctuation solution for homogeneous global stiffness and independent eight-point assembly with sparse direct solution otherwise.

## Input contract

| Field | Meaning |
|---|---|
| phase_id | Integer array (Nz, Ny, Nx); every dimension is at least 2 |
| domain_size | Positive physical lengths (Lx, Ly, Lz) |
| materials | Phase-label dictionary; JSON keys are converted to integers |
| orientation | Optional constant or spatial rotation matrices ending in (3,3) |
| axis1, axis2 | Alternative orientation representation in an NPZ file |

Arrays use **z, y, x** ordering; domain lengths use **x, y, z**. Rotation-matrix columns are local material axes in global coordinates. Only active non-isotropic phases declared with frame="local" require orientation. Rotations must be orthonormal and right-handed.

Material definitions:

- Isotropic: kind="isotropic", E, nu.
- Orthotropic: kind="orthotropic", E1, E2, E3, nu12, nu13, nu23, G12, G13, G23.
- General anisotropic: kind="anisotropic", symmetric positive-definite C with shape (6,6).

The engineering-Voigt order is **11, 22, 33, 23, 13, 12**. Engineering shear strains are doubled; shear stresses are not. Use consistent units; output stiffness has the same units as input moduli. Archived paper values use MPa.

The mesh uses periodic, axis-aligned rectangular HEX8 elements with full integration and elementwise constant stiffness. Six unit macroscopic engineering-strain loads define the columns of CH.

## GPU solve

    uv run --no-sync voxel-lmfem solve --input model.npz --materials materials.json --backend cuda --warmup --output runs/model/result.json

Output records the backend, stiffness, residual checks and elapsed time. The GPU backend uses FP64 stiffness actions and PCG vectors with FP32 FFT preconditioning.

## Benchmark campaigns

Prepare canonical isotropic inputs in a repository checkout:

    uv run --no-sync voxel-lmfem prepare-inputs
    uv run --no-sync voxel-lmfem chapter-study --phase launch --cases sphere,grosmont,woven --grid 192 --variants stencil27_jacobi,execution_fp32 --repeats 3 --out runs/table3
    uv run --no-sync voxel-lmfem woven-fraction --phase launch --grid 192 --fractions all --repeats 3 --out runs/woven_fraction

Launch phases create fresh worker processes through the Unix launcher; run GPU campaigns on Linux. Use uv run --no-sync voxel-lmfem COMMAND --help for driver options. Benchmark inputs are read from repro/inputs/.

Each process owns a configured numerical backend. Use separate processes for concurrent solves; each process owns its solver configuration.
