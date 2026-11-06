# Voxel-LMFEM

GPU finite element homogenization of voxel microstructures with spatially varying anisotropy. Voxel-LMFEM combines fully integrated HEX8 elements, compact material storage and FFT-preconditioned conjugate gradients.

This README contains installation, external-model specifications, solver usage and paper reproduction instructions. The repository uses one branch, `main`; published versions are identified by release tags.

[Installation](#installation) · [CUDA preparation](#prepare-cuda-extensions) · [External models](#external-models) · [Input specification](#input-specification) · [Python API](#python-api) · [Paper reproduction](#paper-reproduction) · [Development](#development) · [中文速查](#中文速查)

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and create a Python 3.12 environment:

```bash
git clone --single-branch --branch main https://github.com/yufangjie1643/Voxel-LMFEM.git
cd Voxel-LMFEM
uv venv --python 3.12
uv pip install -e .
uv run --no-sync voxel-lmfem doctor
```

uv manages `.venv`; `uv run --no-sync` uses its installed packages. The base dependencies are NumPy and SciPy. Python 3.10 and newer are supported.

Run commands from the repository root. A first CPU example is:

```bash
uv run --no-sync voxel-lmfem example --out runs/example
uv run --no-sync voxel-lmfem solve --input runs/example/homogeneous.npz --materials runs/example/materials.json --backend cpu --output runs/example/result.json
```

The example computes a homogeneous cell's effective stiffness, which equals its constituent stiffness. Use a new output file for each solve.

### GPU environment

Use Linux with an NVIDIA GPU, driver, CUDA toolkit and C++ compiler:

```bash
uv pip install --torch-backend=auto --reinstall-package torch -e ".[gpu]"
uv run --no-sync voxel-lmfem doctor --require-cuda
```

For CUDA 13.0, select `--torch-backend=cu130`. See the [uv/PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for supported backends. CUDA extensions compile on first use. The GPU extra includes Ninja, packaging and Numba; the CSR baseline also needs [CuPy](https://docs.cupy.dev/en/stable/install.html).

The paper's environment is recorded in `repro/environment.txt`: RTX 5090, Python 3.12, PyTorch 2.12.0+cu130 and CUDA 13.0.

Compiled extensions are cached in `~/.cache/voxel-lmfem/torch_extensions`. Set `TORCH_EXTENSIONS_DIR` to choose another cache location.

### Prepare CUDA extensions

The first GPU run compiles C++/CUDA extensions for the selected solver and can take several minutes. From a repository checkout, prepare the default paper solver before running benchmarks:

```bash
uv run --no-sync voxel-lmfem build
```

This prepares `execution_fp32` and runs an 8³ woven-cell check. Output shows the GPU and toolkit, each extension, live Ninja/compiler messages, elapsed time and a completion summary. A running extension reports its elapsed time every 15 seconds. Each invocation saves `build.log` and the check result in a new directory under `runs/build/`.

Prepare another chapter-study configuration or choose the compiler's parallelism with:

```bash
uv run --no-sync voxel-lmfem build --variant stencil27_jacobi
uv run --no-sync voxel-lmfem build --variant execution_fp32 --jobs 4
```

The default compiler parallelism is `MAX_JOBS`, or 2 when unset. More jobs use more host memory. Repeating the command reuses the extension cache. Changes to extension sources, build flags, GPU architecture or the PyTorch/CUDA environment can require rebuilding.

The shell's elapsed time includes compilation, initialization, warm-up and checks. The time on a chapter-study `PASS` line is the measured solver workflow after warm-up.

To match the paper's recorded Python dependencies and CUDA 13.0 backend:

```bash
uv pip install --torch-backend=cu130 -c repro/environment.txt -e ".[gpu]"
```

The [release wheel](https://github.com/yufangjie1643/Voxel-LMFEM/releases) provides the Python API, model importer and solver CLI. Install its downloaded filename with `uv pip install`, then use `uv run --no-project python -m voxel_lmfem --help` outside a checkout. Paper benchmarks, their precompilation command and reproduction data use the full repository. `VOXEL_LMFEM_ROOT` selects a separate checkout when needed.

## External models

Use `import-model` to convert **NPY/NPZ arrays of material labels** into validated solver inputs. It preserves voxel values, normalizes array-axis order, checks materials and directions, and records source hashes. Importing uses the CPU.

### Complete working example

```bash
uv run --no-sync python examples/import_model.py --out runs/import_demo
```

This creates an external NPY model in x,y,z array order, imports it, and solves it with the CPU reference. The converted model is in `runs/import_demo/model/`; the result is `runs/import_demo/result.json`.

### Import an NPY volume

Prepare `phase.npy` and a `materials.json` file following the specification below. For source dimensions z,y,x and voxel spacings 0.01, 0.02 and 0.03 in the physical x,y,z directions:

```bash
uv run --no-sync voxel-lmfem import-model --input phase.npy --materials materials.json --axes zyx --spacing 0.01 0.02 0.03 --out runs/model
```

Use `--axes xyz` when the source dimensions mean x,y,z. To specify total cell lengths instead of voxel spacing, replace `--spacing ...` with `--domain-size Lx Ly Lz`.

### Import an NPZ volume

The default label-array key is `phase_id`. An NPZ may also contain `domain_size` and orientation arrays:

```bash
uv run --no-sync voxel-lmfem import-model --input external.npz --materials materials.json --out runs/model
```

For labels stored under `labels`, use:

```bash
uv run --no-sync voxel-lmfem import-model --input external.npz --key labels --materials materials.json --axes xyz --domain-size 1 2 3 --out runs/model
```

Supply either spacing or total lengths when the NPZ has no `domain_size`. An explicit `--spacing` or `--domain-size` overrides the source's physical dimensions. Arrays are reordered without interpolation or resampling.

### Import directions

For local-frame anisotropic materials, store directions inside the source NPZ or supply a separate NPY/NPZ:

```bash
uv run --no-sync voxel-lmfem import-model --input phase.npy --materials materials.json --axes zyx --domain-size 1 1 1 --orientation rotations.npy --out runs/oriented_model
```

An orientation NPY contains rotation matrices. An orientation NPZ contains either `orientation` or the pair `axis1`, `axis2`. Use one source of directions per model. Spatial direction fields follow the source label array's dimension order; vector and matrix components remain in global x,y,z coordinates.

### Imported files and solving

The output must be a new directory. Successful import creates:

| File | Contents |
|---|---|
| `model.npz` | `phase_id` in z,y,x order, `domain_size` in x,y,z order, and supplied directions |
| `materials.json` | Validated phase-to-material definitions |
| `import.json` | Source paths and SHA-256 hashes, original and normalized shapes, physical dimensions, voxel counts and volume fractions per phase |

Run the imported model:

```bash
uv run --no-sync voxel-lmfem solve --input runs/model/model.npz --materials runs/model/materials.json --backend cuda --warmup --output runs/model/result.json
```

Use `--backend cpu` for reference cells of up to 4096 voxels. Results include the 6×6 effective stiffness `CH`, residual checks and elapsed time. The CUDA solver uses FP64 stiffness actions and PCG vectors with FP32 FFT preconditioning. `--warmup` performs a full same-grid warm-up before the measured call.

## Input specification

### Geometry, axes and units

| Item | Required convention |
|---|---|
| Geometry | A regular, axis-aligned rectangular voxel grid; at least two cells per axis |
| Labels | A 3D integer array; every used label has a material definition. Boolean masks become IDs 0 and 1 |
| Source axes | `--axes` names the three source dimensions; default `zyx`. All six permutations of `xyz` are supported |
| Solver array | `phase_id.shape == (Nz, Ny, Nx)` |
| Physical size | `domain_size == (Lx, Ly, Lz)`, finite positive values |
| Voxel spacing | `(dx, dy, dz)`; `Lx = Nx*dx`, `Ly = Ny*dy`, `Lz = Nz*dz` |
| Units | One consistent length unit and one consistent stress/modulus unit. Output stiffness uses the input modulus unit |
| Material IDs | Integer keys in Python or integer strings in JSON; IDs retain their values and fit signed 64-bit integers |
| Physics | Periodic, small-strain linear elasticity; positive-definite material stiffness; fully integrated rectangular HEX8 elements |

An axis permutation changes storage order; physical coordinates and direction components keep their meaning. NPY/NPZ preserve their numeric data types and storage layout in their file headers; the importer preserves label values while normalizing byte order and axis order. Convert image intensities to material labels before import. For CAD or mesh models, first export a voxelized material-label volume as NPY/NPZ.

Label 0 is an ordinary material ID. All used labels, including any phase representing pores, follow the material definitions supplied by the user.

### Materials JSON

The following matrix and yarn properties are those in Table 1, in MPa:

```json
{
  "0": {"kind": "isotropic", "E": 3500.0, "nu": 0.35},
  "1": {
    "kind": "orthotropic", "frame": "local",
    "E1": 135000.0, "E2": 9000.0, "E3": 9000.0,
    "nu12": 0.30, "nu13": 0.30, "nu23": 0.35,
    "G12": 5000.0, "G13": 5000.0, "G23": 3500.0
  }
}
```

| `kind` | Properties |
|---|---|
| `isotropic` | `E`, `nu`, with `E > 0` and `-1 < nu < 0.5` |
| `orthotropic` | `E1`, `E2`, `E3`, `nu12`, `nu13`, `nu23`, `G12`, `G13`, `G23`; the resulting stiffness must be positive definite |
| `anisotropic` | `C`: a finite, symmetric, positive-definite 6×6 stiffness matrix |

`frame` is `global` by default. Set `frame: "local"` when non-isotropic coefficients are given in material axes; directions are required wherever those phases occur. Isotropic phases use the same stiffness in every orientation.

The engineering-Voigt order is **11, 22, 33, 23, 13, 12**. Strain components are `(ε11, ε22, ε33, 2ε23, 2ε13, 2ε12)`; stress components are `(σ11, σ22, σ33, σ23, σ13, σ12)`. The six columns of `CH` correspond to six unit macroscopic engineering-strain loads. Paper values use MPa; generated CSV tables state their units in the column names.

### Orientation arrays

Let `S` denote the source label array's shape. The importer accepts:

| Representation | Array shape | Meaning |
|---|---|---|
| Constant rotation | `(3, 3)` | One material frame for all applicable voxels |
| Spatial rotations | `S + (3, 3)` | One frame per source voxel |
| Constant axes | `axis1` and `axis2`, each `(3,)` | One pair of material directions |
| Spatial axes | `axis1` and `axis2`, each `S + (3,)` | One pair per source voxel |

Rotation-matrix **columns** are local material axes in global x,y,z coordinates. Matrices at active local-frame phases must be finite, orthonormal and right-handed (`R.T @ R = I`, `det(R) = +1`). For the two-axis representation, axis1 and axis2 must be finite, nonzero and noncollinear. The solver normalizes axis1, obtains axis3 from the normalized cross product, and completes the frame with axis2 = axis3 × axis1.

Example of a 90° rotation about global z:

```python
import numpy as np

R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
np.save("rotations.npy", R)
```

## Python API

```python
import numpy as np
from voxel_lmfem import homogenize, import_model

import_model("phase.npy", "materials.json", "runs/imported",
             axes="zyx", spacing=[0.01, 0.01, 0.01])

result = homogenize(
    np.zeros((4, 4, 4), dtype=np.uint8),
    {0: {"kind": "isotropic", "E": 3500.0, "nu": 0.35}},
    [1.0, 1.0, 1.0], backend="cpu")
C_H = np.asarray(result["CH"])
```

Use `backend="cuda"` for the GPU solver and `orientation=...` for local-frame materials. The CPU backend uses independent eight-point assembly and sparse direct solution; homogeneous global stiffness has the exact zero-fluctuation solution. Each process owns its solver configuration; use separate processes for concurrent solves.

## Paper reproduction

Commands run from the repository root using the uv environment above. New GPU measurements are written to runs/; the reproduction command reads the included paper measurements.

### CPU regeneration of published values

    uv run --no-sync voxel-lmfem reproduce

Outputs go to `repro/generated/`; use `--out runs/paper_tables` to choose another directory. `TABLES.md` indexes the results and `verification.json` records the checks. [repro/table_manifest.json](repro/table_manifest.json) maps sections and labels to outputs, [repro/file_manifest.json](repro/file_manifest.json) records checksums, and [repro/path_map.json](repro/path_map.json) resolves original record identifiers.

| Manuscript item | Generated file |
|---|---|
| Table 1: constituent properties | `table_1_constituent_properties.csv` |
| Table 2: stiffness verification | `table_2_verification.csv` |
| Table 3: solver comparison | `table_3_solver_summary.csv` |
| Figure 8: convergence | `figure_8_grid_convergence.csv` |
| Figure 9: optimization chain | `figure_9_optimization_chain.csv` |
| Figure 10: grid scaling/cost | `figure_10_grid_cost.csv` |
| Figure 11: yarn fractions/savings | `figure_11_woven_fraction.csv`, `figure_11_gains.csv` |
| Figure D.1: active fractions | `figure_D1_active_fraction.csv` |
| Appendix D: component timings | `appendix_D_component_comparisons.csv` |
| Summary checks | `headline_checks.json`, `verification.json` |

Run Python with assertions enabled; the numerical checks use them to validate the generated values.

Table 3 selects stencil27_jacobi and execution_fp32 for each 192³ cell. Time is the median of three independent processes; memory is the maximum measured process peak. All speedups in this table use the stencil Jacobi–PCG baseline. Table 2 compares proposed tensors with the analytical laminate, chfem for the two isotropic cells, and stencil Jacobi–PCG for woven material.

Figure 8 reads three raw results per grid and computes discrepancies against 480³. Figure 10 retains the original cost study's configurations and reference media, as described in Section 5.3. Figure 11 corresponds to Section 5.4, with 21 fractions × 3 configurations × 3 repetitions. Figure D.1 uses nine active fractions × 2 configurations × 3 repetitions.

### Run independent CPU checks

The material-state example needs NumPy and SciPy:

    uv run --no-sync voxel-lmfem check material-state --output runs/checks/material_state.json

Output: runs/checks/material_state.json. This checks five frozen material-history cases, independent FI assembly, equilibrium residuals and the loading/unloading path in Appendix C. It is a CPU prototype.

The independent FI operator checks additionally require PyTorch (a CPU build is sufficient):

    uv pip install --torch-backend=cpu --reinstall-package torch -e ".[cpu-checks]"
    uv run --no-sync voxel-lmfem check fi --output runs/checks/fi_cpu.json

Use a new output path for each check.

For the yarn geometry, from the repository root:

    uv run --no-sync voxel-lmfem woven-fraction --phase geometry --grid 32 --out runs/checks/geometry

This checks all 21 analytical constructions on a small grid. The saved 192³ validation is included with the measured data. The optional verify-stencils-cpu command also imports Triton through the stored-stencil backend; use a Linux environment with those GPU software dependencies.

### Rerun chapter benchmarks

From the repository root on Linux (run uv run --no-sync voxel-lmfem prepare-inputs first):

    uv run --no-sync voxel-lmfem chapter-study --phase launch --cases sphere,grosmont,woven --grid 192 --variants stencil27_jacobi,execution_fp32 --repeats 3 --out runs/table3

For the controlled woven optimization chain:

    uv run --no-sync voxel-lmfem chapter-study --phase launch --cases woven --grid 192 --variants stencil27_fft,modal_fft,hybrid_fft,background_fft,execution_fp64,execution_fp32 --repeats 3 --out runs/optimization

For a small worker smoke test before a large campaign:

    uv run --no-sync voxel-lmfem chapter-study --phase worker --case woven --grid 32 --variant execution_fp32 --repeat 0 --out runs/smoke

For all 11 convergence grids, using Linux shell syntax:

```bash
for n in 64 96 128 160 192 224 256 320 384 448 480; do
  uv run --no-sync voxel-lmfem chapter-study --phase launch --cases woven --grid "$n" --variants execution_fp32 --repeats 3 --out "runs/convergence/n$n"
done
```

The exact older grid-cost jobs, method choices and reference media are recorded under repro/raw/paper/scaling/remote/plan.json and repro/raw/paper/stencil/formal/. Paths inside saved plans describe the original machine; use the included drivers and select a new runs/ destination. The table-generation command uses the archived measurements.

For Appendix D's 54-run active-fraction sweep:

    uv run --no-sync voxel-lmfem active-sweep

It creates its plan and results in runs/active_fraction_128_20261024.

### Rerun yarn-fraction benchmarks

From the repository root:

    uv run --no-sync voxel-lmfem woven-fraction --phase launch --grid 192 --fractions all --repeats 3 --out runs/woven_fraction

The driver executes all three configurations at each fraction and rotates their order. The corresponding analyzer is repro/raw/woven/analysis/analyze.py. Its input directory must contain raw/results and raw/logs; the saved campaign demonstrates that layout.

### Measurement conventions

Each full workflow solves six unit engineering-strain loads under periodic zero-mean constraints. Stiffness actions and PCG vectors use FP64. The proposed implementation uses FP32 FFT preconditioning and verifies the final FP64 Euclidean true residual against 1e-8. The workflow includes material construction, operator preparation, right-hand sides, solves, residual checks, stress recovery and cleanup. A full same-grid warm-up precedes measurement.

Each configuration has three fresh-process repetitions. Report median complete-call time and its min/max range. Peak memory is the maximum 20-ms NVML process sample, including CUDA context, libraries and retained allocations. Kernel timings in Appendix D use CUDA events with resident data, outside the full-workflow window.

Optimization-chain configurations share a phase-based reference and 327 total iterations. The cost-scaling campaign uses sampled references for stored stencils and phase-based references for modal solvers. Interpret these cost comparisons as complete solver configurations. Yarn-fraction configurations share the same fixed matrix-based reference and iteration sequence at each fraction.

### Reference solver and geometry

The frozen campaigns include chfem result JSON, sampled-memory logs and canonical isotropic input arrays. Its maintained source copy and license are in repro/baselines/. The recorded reference tensors are transformed into the project's Voigt convention before comparison. Rebuilding the native chfem executable is platform dependent; follow its included build documentation. The comparison tables use the included chfem outputs.

The Grosmont input is the paper's 400³ crop beginning at zero-based (z,y,x)=(550,0,0) and its nearest-neighbor 192³ resampling. The distributed geometry consists of these two arrays. The manuscript cites Andrä et al.'s digital-rock dataset; original dataset attribution remains applicable.

Keep the GPU exclusive during timing. Large stencil cases need substantially more memory than the proposed solver. Geometry generation, JIT compilation, warm-up and I/O are outside the measurement window. Chapter launch outputs include `results/`, `logs/` and a completion record; choose a new output directory for each campaign.

<details>
<summary>Additional chapter-study controls and CSR baseline</summary>

`chapter-study` also exposes `csr_jacobi`, `stencil14_fft` and `direct_fft`. The direct eight-point operator controls modal evaluation; the 14-point stencil compares symmetry-based storage with the full 27-point stencil. These controls share the eight-point HEX8 discretization, periodic grid, six strains, FP64 physical operators and final residual criterion. FFT stages share a normalized phase-based reference; `execution_fp32` selects FP32 FFT arithmetic.

```bash
uv run --no-sync voxel-lmfem chapter-study --phase validate --out runs/csr_validation
uv run --no-sync voxel-lmfem chapter-study --phase launch --grid 16 --repeats 1 --out runs/chapter_chain_smoke
uv run --no-sync voxel-lmfem chapter-study --phase launch --grid 192 --cases sphere,grosmont --variants csr_jacobi,stencil27_jacobi,stencil27_fft,execution_fp32 --out runs/chapter_isotropic
```

### CSR baseline
CuPy 14.2.0 (cupy-cuda13x), CUDA 13.0, FP64 values, int32 indices.
Element stiffness contributions are computed from eight Gauss points and
assembled into the full periodic global CSR matrix. Two bounded passes count
and fill the exact nonzeros, without a simultaneous full COO/stencil cache.
The stored matrix is sorted and has unique column indices in each row.
CSR SpMV and CG are CuPy library operations. Jacobi is the assembled scalar
diagonal with a zero-mean projection to fix the three translation modes.
Material preparation is shared with the full-stencil implementation; its
element-wise tensors are passed to CuPy without copying through DLPack.
RHS and mean stress use the exact element-average strain map for cellwise
constant material, equivalent to eight-point quadrature for these integrals.

The proposed isotropic variant enters through the native global-frame phase
interface: both phases share their tensors and the pool has exactly two rows.
The woven variant combines one shared matrix row with local yarn tensors.

CuPy 14.2.0 csr_matrix.diagonal() computes a binary-search midpoint by adding
two int32 offsets. That sum can overflow when nnz exceeds 2^30 even though
the individual offsets fit int32. This baseline extracts the diagonal with
64-bit midpoint arithmetic. It does not replace CuPy SpMV or CG.
The assembly is currently restricted to at least three cells per axis and
nnz < 2^31. The tested 192^3 cases satisfy these requirements.

### CSR verification
A 4x5x6 grid with unequal cell lengths and random fully anisotropic SPD
materials is independently assembled in NumPy. Tests compare the full CSR
matrix, symmetry, the three rigid translations, all six RHS vectors,
solutions from a CPU direct solve, and mean stresses. A small woven campaign
also compares the assembled and matrix-free implementations end to end.


</details>

## Development

Create the environment following [Installation](#installation), then install the development tools:

```bash
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest -q
uv run --no-sync python -m ruff check src tests examples
uv build
```

Build outputs are written to `dist/`. The wheel contains the Python API and numerical implementation; the full repository also contains the paper's experiments and data.

### Working directories

- `src/`: solver implementation and public interfaces.
- `tests/`: numerical and interface tests.
- `examples/`: small examples and material configurations.
- `repro/bench/`: benchmark protocols and measurement drivers.
- `runs/`: local computation results.

Use analytical solutions, mechanics invariants and independent numerical references to verify solver changes. The [verification section](#verification) describes the available checks.

### Names and paths

Use descriptive file names and group files by purpose. Repository-relative paths are limited to 120 characters, individual names to 64 characters, and depth to 12 components. Run the project checks with:

```bash
uv run --no-sync python .github/tools/check_paths.py
```

CI runs path checks, numerical tests, input preparation, table generation and package installation on Windows and Linux.

## Verification

### Numerical and interface tests

```bash
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest -q
```

The tests cover homogeneous stiffness, layered materials with an analytical traction-continuity solution, rotated anisotropy, input validation, NPY/NPZ model import, the command-line interface and build-progress reporting.

### Finite element and material-state checks

```bash
uv pip install --torch-backend=cpu --reinstall-package torch -e ".[cpu-checks]"
uv run --no-sync voxel-lmfem check fi --output runs/checks/fi.json
uv run --no-sync voxel-lmfem check material-state --output runs/checks/material_state.json
uv run --no-sync voxel-lmfem woven-fraction --phase geometry --grid 32 --out runs/checks/geometry
```

These checks exercise the independently assembled FI operator, five material-history cases and 21 yarn-fraction geometries. Use a new output path for each run.

### GPU integration

After installing the GPU and development dependencies on Linux:

```bash
VOXEL_LMFEM_TEST_GPU=1 uv run --no-sync python -m pytest tests/test_gpu.py -q
```

The integration test compares a heterogeneous, rotated-anisotropic 4³ cell with the independent CPU solution and checks the final residual. CPU CI runs the remaining tests.

### Model and execution scope

The solver implements periodic, small-strain linear elasticity with positive-definite stiffness tensors and fully integrated rectangular HEX8 elements. Each call solves six macroscopic engineering-strain loads. The CPU reference supports up to 4096 voxels; the CUDA backend handles larger cells. The material-state fixture is a separate CPU example.

Paper-data checks regenerate archived observations. Fresh GPU measurements use the benchmark commands and protocols in [Paper reproduction](#paper-reproduction).

## Project structure

```text
src/voxel_lmfem/   Solver, importer, material interface and command line
examples/         Example scripts and material configurations
tests/            Numerical tests and material-state checks
docs/             Changelog and third-party notices
repro/            Paper datasets, benchmark drivers and generated tables
.github/          Continuous integration and project checks
```

The package entry point is `voxel-lmfem`. It provides solves, examples, validation and reproduction commands. Python applications use `voxel_lmfem.homogenize`.

### Solver components

| Component | Role |
|---|---|
| materials | Constitutive definitions, orientations and material storage |
| hex8_modal, constitutive_basis | Element action and polynomial representation |
| hybrid_operator | Background and material-contrast evaluation |
| spectral, reference_medium, modal_reference | FFT preconditioning and reference construction |
| solver_chain, assembly_pipeline, compact_operator | PCG execution |
| runtime | Finite element implementation dependencies |

Numerical components live in `src/voxel_lmfem/_core/`. Public input validation and interfaces live in `inputs.py`, `importers.py`, `solver.py` and `cli.py`.

### Reproduction files

`repro/bench/` contains experiment drivers, `repro/baselines/` contains reference solvers, and `repro/raw/` contains measured results and source snapshots. Prepared input arrays go in `repro/inputs/`; new computation results go in `runs/`.

`repro/table_manifest.json` maps paper results to generated files. `repro/path_map.json` maps original record identifiers to the datasets' current locations.

Under `repro/raw/`, `paper/` stores comparisons, convergence and performance; `chapter/` stores canonical inputs and solver snapshots; `woven/` stores yarn-fraction records; and `sources/` stores numerical source snapshots. Project user instructions belong in this README. Source snapshots and third-party packages retain their original documentation.

## 中文速查

本仓库仅维护 `main` 分支，安装、输入规范、预编译、求解和论文复现说明统一在本 README 中。

- **外部模型入口**：`voxel-lmfem import-model`，支持 NPY/NPZ 三维材料编号数组。完整示例：`uv run --no-sync python examples/import_model.py --out runs/import_demo`。
- **轴顺序**：`--axes` 指源数组三个维度的含义，默认 `zyx`；求解器内部统一为 `(Nz, Ny, Nx)`。`--spacing` 和 `--domain-size` 始终按物理 `(x, y, z)` 填写。改变存储轴顺序时，方向矢量的物理分量保持不变。
- **材料**：JSON 的键是相编号，每个实际出现的编号都要有定义。支持各向同性、正交各向异性和一般各向异性，刚度须正定。编号 0 同样需要材料定义。模量和输出刚度使用相同单位。
- **材料方向**：`frame: "local"` 的各向异性相需要方向；旋转矩阵的列表示材料局部轴在全局坐标系中的方向。空间方向场与源体素数组使用相同的网格维度顺序。也可用 `axis1`、`axis2` 两个不共线方向描述材料坐标系。
- **导入输出**：新目录中的 `model.npz`、`materials.json` 和 `import.json`。随后使用 `solve --input .../model.npz --materials .../materials.json` 求解。
- **首次编译**：`uv run --no-sync voxel-lmfem build` 显示实时编译日志、每 15 秒更新耗时，并进行 8³ 检查。日志保存在 `runs/build/`。首次编译可能需要几分钟，后续复用缓存。
- **论文数据**：`uv run --no-sync voxel-lmfem reproduce` 从随附测量记录生成表格和图数据；上面的算例命令会重新进行 GPU 计算。表格输出在 `repro/generated/`，新计算结果在 `runs/`。

## License and citation

[MIT License](LICENSE). [Third-party credits](docs/NOTICE.md). Use [CITATION.cff](CITATION.cff) to cite the software and accompanying paper. See [Changelog](docs/changelog.md) for updates.
