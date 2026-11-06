# Validation

## Numerical and interface tests

```bash
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest -q
```

The tests cover homogeneous stiffness, layered materials with an analytical traction-continuity solution, rotated anisotropy, input validation and the command-line interface.

## Finite element and material-state checks

```bash
uv pip install --torch-backend=cpu --reinstall-package torch -e ".[cpu-checks]"
uv run --no-sync voxel-lmfem check fi --output runs/checks/fi.json
uv run --no-sync voxel-lmfem check material-state --output runs/checks/material_state.json
uv run --no-sync voxel-lmfem woven-fraction --phase geometry --grid 32 --out runs/checks/geometry
```

These checks exercise the independently assembled FI operator, five material-history cases and 21 yarn-fraction geometries. Use a new output path for each run.

## GPU integration

After installing the GPU and development dependencies on Linux:

```bash
VOXEL_LMFEM_TEST_GPU=1 uv run --no-sync python -m pytest tests/test_gpu.py -q
```

The integration test compares a heterogeneous, rotated-anisotropic 4³ cell with the independent CPU solution and checks the final residual. CPU CI runs the remaining tests.

## Model and execution scope

The solver implements periodic, small-strain linear elasticity with positive-definite stiffness tensors and fully integrated rectangular HEX8 elements. Each call solves six macroscopic engineering-strain loads. The CPU reference supports up to 4096 voxels; the CUDA backend handles larger cells. The material-state fixture is a separate CPU example.

Paper-data checks regenerate archived observations. Fresh GPU measurements use the benchmark commands and protocols in [Paper reproduction](../repro/README.md).
