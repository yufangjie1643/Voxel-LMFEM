# Voxel-LMFEM

**Memory-efficient GPU finite element homogenization.**

Voxel-LMFEM solves periodic small-strain elasticity on fully integrated rectangular HEX8 voxel grids. It supports spatially varying anisotropic materials, hybrid material storage, background/contrast evaluation and FFT-preconditioned PCG. LM denotes low memory; FEM denotes finite element method.

[Installation](docs/installation.md) · [Usage](docs/usage.md) · [Architecture](docs/architecture.md) · [Validation](docs/validation.md) · [Paper tables](repro/generated/TABLES.md) · [中文](README_zh.md)

## Quick start: CPU example

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

    git -c core.longpaths=true clone https://github.com/yufangjie1643/Voxel-LMFEM.git
    cd Voxel-LMFEM
    uv venv --python 3.12
    uv pip install -e .
    uv run --no-sync voxel-lmfem example --out runs/example
    uv run --no-sync voxel-lmfem solve --input runs/example/homogeneous.npz --materials runs/example/materials.json --backend cpu --output runs/example/result.json

Run the commands from the repository root. uv manages the local .venv; uv run --no-sync uses the installed environment without changing the selected CPU/CUDA packages. The base installation needs NumPy and SciPy. The small CPU reference is limited to 4096 voxels; it verifies inputs and results without requiring a GPU.

## GPU solve

In the same uv environment on Linux, install the GPU extra with a driver-compatible PyTorch backend as described in the installation guide:

    uv pip install --torch-backend=auto --reinstall-package torch -e ".[gpu]"
    uv run --no-sync voxel-lmfem doctor --require-cuda
    uv run --no-sync voxel-lmfem solve --input model.npz --materials materials.json --backend cuda --warmup --output runs/model/result.json

The GPU implementation uses FP64 physical stiffness and PCG vectors, with FP32 FFT preconditioning. Native extensions are compiled on first use. GPU benchmarks run on Linux.

## Python API

    import numpy as np
    from voxel_lmfem import homogenize

    result = homogenize(
        np.zeros((4, 4, 4), dtype=np.uint8),
        {0: {"kind": "isotropic", "E": 3500.0, "nu": 0.35}},
        [1.0, 1.0, 1.0], backend="cpu",
    )
    C_H = np.array(result["CH"])

Phase arrays use (z,y,x); domain lengths use (x,y,z). The engineering-Voigt order is (11,22,33,23,13,12). See the usage guide for orthotropic/general-anisotropic materials and local orientations.

## Repository

| Directory | Contents |
|---|---|
| src/voxel_lmfem | Maintained package and one canonical numerical core |
| experiments | Benchmark and timing protocols |
| examples, configs | Runnable examples and material definitions |
| tests, validation | Regression tests and independent numerical checks |
| data | Input guide and prepared benchmark files |
| baselines | Third-party solver source and licenses |
| docs | Installation, use, architecture and development |
| repro | Paper map, checksums, scripts and generated tables |
| campaigns | Frozen source/result evidence used by the paper |
| runs | New local outputs, ignored by Git |

Historical snapshots are retained for provenance. New solves use the maintained source through the public API or root runner.

## Reproduce the manuscript

Manuscript: **Memory-efficient GPU finite element homogenization of microstructures with spatially varying anisotropy**, snapshot 25 October 2026.

    uv run --no-sync python run.py reproduce

This CPU command verifies archived file hashes and regenerates the current three tables and numerical result-figure data, including the six-row performance table, eleven-grid convergence through 480³, 21 yarn fractions and nine active fractions. It validates saved measurements rather than rerunning GPU timings.

See [repro/table_manifest.json](repro/table_manifest.json), [repro/generated/TABLES.md](repro/generated/TABLES.md) and [repro/REPRODUCING.md](repro/REPRODUCING.md). The direct entry uv run --no-sync python repro/scripts/reproduce_paper_tables.py remains supported. Fixed maintained release: **v0.1.0**; the original review-package tags remain available.

To prepare canonical benchmark inputs and launch fresh comparisons:

    uv run --no-sync python run.py prepare-inputs
    uv run --no-sync python run.py chapter-study --phase launch --cases sphere,grosmont,woven --grid 192 --variants stencil27_jacobi,execution_fp32 --repeats 3 --out runs/table3

## Tests and development

    uv pip install -e ".[dev]"
    uv run --no-sync python -m pytest -q
    uv build

Linux/Windows CPU CI tests the API, numerical reference, input preparation, table regeneration and installed wheel. GPU tests are opt-in; see the validation guide. The original GPU measurements used RTX 5090 / PyTorch 2.12.0+cu130 / CUDA 13.0.

## Citation and terms

Voxel-LMFEM code and accompanying software documentation are released under the [MIT License](LICENSE). Third-party code retains its original copyright and license notices; dataset attribution and terms are described in [NOTICE.md](NOTICE.md). Please use [CITATION.cff](CITATION.cff) to cite this software and the accompanying manuscript when using its methods or data.
