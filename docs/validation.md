# Validation

Create the uv environment following docs/installation.md, then run these commands from the repository root.

## Automated regression tests

    uv pip install -e ".[dev]"
    uv run --no-sync python -m pytest -q
    uv run --no-sync python -m ruff check src/voxel_lmfem tests examples run.py project_paths.py

Tests compare homogeneous stiffness with the constitutive tensor, layered cells with an independent traction-continuity solution, and rotated anisotropy with the expected axis permutation. Input checks reject invalid labels, non-positive materials, missing orientation and left-handed rotations. CLI tests run outside the repository and verify output preservation.

Continuous integration runs CPU tests, source checks, wheel construction and paper-data regeneration on Linux and Windows with Python 3.12. It also tests a wheel outside the checkout in a separate environment.

## Original independent checks

    uv run --no-sync python run.py check fi --output runs/checks/fi.json
    uv run --no-sync python run.py check material-state --output runs/checks/material_state.json
    uv run --no-sync python run.py woven-fraction --phase geometry --grid 32 --out runs/checks/geometry

The FI check needs PyTorch, but not CUDA. The material-state example uses NumPy/SciPy. The geometry check covers 21 fraction constructions.

## GPU integration test

With a configured Linux CUDA environment:

    VOXEL_LMFEM_TEST_GPU=1 uv run --no-sync python -m pytest tests/test_gpu.py -q

This compares a heterogeneous, rotated anisotropic 4³ cell against the independent CPU solution and checks the true residual. Ordinary CPU CI skips it. Passing CPU tests does not certify a GPU installation.

## Archived observations

    uv run --no-sync python run.py reproduce

This validates checksums, residuals, repetition counts, table rounding and tensor comparisons from measured records. It does not rerun GPU timings. See repro/REPRODUCING.md for measurement conventions and external-reference scope.

## Scope and limits

- Small-strain linear elasticity with symmetric positive-definite tensors.
- Axis-aligned rectangular periodic HEX8 grids; elementwise constant stiffness.
- One cell per public API call; six engineering-strain loads.
- CPU reference limited to 4096 voxels.
- GPU workflow requires a CUDA toolchain and native extension compilation.
- Material-state example is a separate CPU fixture, not a nonlinear GPU solver.
- Performance depends on the recorded hardware, software and timing protocol.
