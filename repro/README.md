# Paper reproduction

Commands run from the repository root using the uv environment described in [Installation](../docs/installation.md).

## 1. CPU regeneration of published values

    uv run --no-sync voxel-lmfem reproduce

Use --out to choose an output directory. Run Python with assertions enabled; the numerical checks use them to validate the generated values.

Table 3 selects stencil27_jacobi and execution_fp32 for each 192³ cell. Time is the median of three independent processes; memory is the maximum measured process peak. All speedups in this table use the stencil Jacobi–PCG baseline. Table 2 compares proposed tensors with the analytical laminate, chfem for the two isotropic cells, and stencil Jacobi–PCG for woven material.

Figure 8 reads three raw results per grid and computes discrepancies against 480³. Figure 10 retains the original cost study's configurations and reference media, as described in Section 5.3. Figure 11 corresponds to Section 5.4, with 21 fractions × 3 configurations × 3 repetitions. Figure D.1 uses nine active fractions × 2 configurations × 3 repetitions.

## 2. Run independent CPU checks

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

## 3. GPU environment

Use Linux, an NVIDIA GPU, a matching driver/CUDA toolkit, a working C++/CUDA compiler, and a CUDA-enabled PyTorch build. Runtime extensions are compiled on first use. The measured environment is recorded in each result and repro/environment.txt:

- GPU: NVIDIA GeForce RTX 5090.
- Python: 3.12.
- PyTorch: 2.12.0+cu130; CUDA: 13.0.
- NumPy, SciPy, Ninja, packaging and Numba: see the recorded environment file.
- The CSR comparison also requires CuPy; chfem uses its separate native build.

Install the CUDA-enabled PyTorch build and dependencies following [Installation](../docs/installation.md). The benchmark workers inherit the uv environment.

First use compiles CUDA extensions and can take several minutes. Prepare the default paper solver with live compiler output before launching a campaign:

    uv run --no-sync voxel-lmfem build

This compiles or loads the cached extensions and runs an 8³ woven-cell check. Logs are saved under runs/build/. Use --variant stencil27_jacobi to prepare the Table 3 baseline. The shell's elapsed time includes compilation and warm-up; the PASS line reports the measured solver workflow.

The large stencil cases need substantially more GPU memory than the proposed solver. Keep the GPU exclusive during timing. Geometry generation, JIT compilation, warm-up and I/O are outside the measurement window.

## 4. Rerun chapter benchmarks

From the repository root on Linux (run uv run --no-sync voxel-lmfem prepare-inputs first):

    uv run --no-sync voxel-lmfem chapter-study --phase launch --cases sphere,grosmont,woven --grid 192 --variants stencil27_jacobi,execution_fp32 --repeats 3 --out runs/table3

For the controlled woven optimization chain:

    uv run --no-sync voxel-lmfem chapter-study --phase launch --cases woven --grid 192 --variants stencil27_fft,modal_fft,hybrid_fft,background_fft,execution_fp64,execution_fp32 --repeats 3 --out runs/optimization

For a small worker smoke test before a large campaign:

    uv run --no-sync voxel-lmfem chapter-study --phase worker --case woven --grid 32 --variant execution_fp32 --repeat 0 --out runs/smoke

For a fresh high-resolution convergence point:

    uv run --no-sync voxel-lmfem chapter-study --phase launch --cases woven --grid 480 --variants execution_fp32 --repeats 3 --out runs/convergence480

The exact older grid-cost jobs, method choices and reference media are recorded under repro/raw/paper/scaling/remote/plan.json and repro/raw/paper/stencil/formal/. Paths inside saved plans describe the original machine; use the included drivers and select a new runs/ destination. The table-generation command uses the archived measurements.

For Appendix D's 54-run active-fraction sweep:

    uv run --no-sync voxel-lmfem active-sweep

It creates its plan and results in runs/active_fraction_128_20261024.

## 5. Rerun yarn-fraction benchmarks

From the repository root:

    uv run --no-sync voxel-lmfem woven-fraction --phase launch --grid 192 --fractions all --repeats 3 --out runs/woven_fraction

The driver executes all three configurations at each fraction and rotates their order. The corresponding analyzer is repro/raw/woven/analysis/analyze.py. Its input directory must contain raw/results and raw/logs; the saved campaign demonstrates that layout.

## 6. Measurement conventions

Each full workflow solves six unit engineering-strain loads under periodic zero-mean constraints. Stiffness actions and PCG vectors use FP64. The proposed implementation uses FP32 FFT preconditioning and verifies the final FP64 Euclidean true residual against 1e-8. The workflow includes material construction, operator preparation, right-hand sides, solves, residual checks, stress recovery and cleanup. A full same-grid warm-up precedes measurement.

Each configuration has three fresh-process repetitions. Report median complete-call time and its min/max range. Peak memory is the maximum 20-ms NVML process sample, including CUDA context, libraries and retained allocations. Kernel timings in Appendix D use CUDA events with resident data, outside the full-workflow window.

Optimization-chain configurations share a phase-based reference and 327 total iterations. The cost-scaling campaign uses sampled references for stored stencils and phase-based references for modal solvers. Interpret these cost comparisons as complete solver configurations. Yarn-fraction configurations share the same fixed matrix-based reference and iteration sequence at each fraction.

## 7. Reference solver and geometry

The frozen campaigns include chfem result JSON, sampled-memory logs and canonical isotropic input arrays. Its maintained source copy and license are in repro/baselines/. The recorded reference tensors are transformed into the project's Voigt convention before comparison. Rebuilding the native chfem executable is platform dependent; follow its included build documentation. The comparison tables use the included chfem outputs.

The Grosmont input is the paper's 400³ crop beginning at zero-based (z,y,x)=(550,0,0) and its nearest-neighbor 192³ resampling. The distributed geometry consists of these two arrays. The manuscript cites Andrä et al.'s digital-rock dataset; original dataset attribution remains applicable.
