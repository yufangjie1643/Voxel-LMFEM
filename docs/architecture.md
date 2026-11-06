# Project structure

```text
src/voxel_lmfem/   Solver, material interface and command line
examples/         Example scripts and material configurations
tests/            Numerical tests and material-state checks
docs/             Installation, usage and developer documentation
repro/            Paper datasets, benchmark drivers and generated tables
.github/          Continuous integration and project checks
```

The package entry point is `voxel-lmfem`. It provides solves, examples, validation and reproduction commands. Python applications use `voxel_lmfem.homogenize`.

## Solver components

| Component | Role |
|---|---|
| materials | Constitutive definitions, orientations and material storage |
| hex8_modal, constitutive_basis | Element action and polynomial representation |
| hybrid_operator | Background and material-contrast evaluation |
| spectral, reference_medium, modal_reference | FFT preconditioning and reference construction |
| solver_chain, assembly_pipeline, compact_operator | PCG execution |
| runtime | Finite element implementation dependencies |

Numerical components live in `src/voxel_lmfem/_core/`. Public input validation and interfaces live in `inputs.py`, `solver.py` and `cli.py`.

## Reproduction files

`repro/bench/` contains experiment drivers, `repro/baselines/` contains reference solvers, and `repro/raw/` contains measured results and source snapshots. Prepared input arrays go in `repro/inputs/`; new computation results go in `runs/`.

`repro/table_manifest.json` maps paper results to generated files. `repro/path_map.json` maps original record identifiers to the datasets' current locations.
