# Reproduction guide

Maintained release: v0.1.0. Use the root run.py entry and the source under src/voxel_lmfem/. The campaigns directory is a frozen evidence archive. See docs/installation.md and docs/usage.md for the installed API.

## 1. CPU regeneration of published values

Install uv as described in docs/installation.md. From a fresh repository checkout:

    uv venv --python 3.12
    uv pip install -e .
    uv run --no-sync python repro/scripts/reproduce_paper_tables.py

An alternative output directory can be supplied with --out. Do not run Python with -O; assertions implement numerical validation. The script never changes the frozen campaigns. Original analyzers that write summaries execute in temporary copies.

Table 3 selects stencil27_jacobi and execution_fp32 for each 192³ cell. Time is the median of three independent processes; memory is the maximum measured process peak. All speedups in this table use the stencil Jacobi–PCG baseline. Table 2 compares proposed tensors with the analytical laminate, chfem for the two isotropic cells, and stencil Jacobi–PCG for woven material.

Figure 8 reads three raw results per grid and computes discrepancies against 480³. Figure 10 retains the original cost study's configurations and reference media, as described in Section 5.3. Figure 11 corresponds to Section 5.4, with 21 fractions × 3 configurations × 3 repetitions. Figure D.1 uses nine active fractions × 2 configurations × 3 repetitions.

The active manifest checks frozen campaign bytes and the regeneration script; it is not an independent certification of experimental truth. It excludes maintained source/docs, generated outputs, runtime caches, new runs and itself. The original v2 inventory remains in repro/provenance/v2_file_manifest.json. The Git release fixes the manifest version.

## 2. Run independent CPU checks

The material-state example needs NumPy and SciPy:

    uv run --no-sync python run.py check material-state --output runs/checks/material_state.json

Output: runs/checks/material_state.json. This checks five frozen material-history cases, independent FI assembly, equilibrium residuals and the loading/unloading path in Appendix C. It is a CPU prototype.

The independent FI operator checks additionally require PyTorch (a CPU build is sufficient):

    uv pip install --torch-backend=cpu --reinstall-package torch -e ".[cpu-checks]"
    uv run --no-sync python run.py check fi --output runs/checks/fi_cpu.json

Use a new output filename when repeating this check; the driver refuses to overwrite an existing result. Both checks passed on Windows/Python 3.12 in the release verification. The tested CPU PyTorch version is recorded in repro/provenance/release_checks.json. This CPU environment is separate from the original GPU measurement environment.

For the yarn geometry, from the repository root:

    uv run --no-sync python -B run.py woven-fraction --phase geometry --grid 32 --out runs/checks/geometry

This checks all 21 analytical constructions on a small grid. The saved 192³ validation is included with the measured data. The optional verify-stencils-cpu command also imports Triton through the stored-stencil backend; use a Linux environment with those GPU software dependencies. It was not part of the successful Windows release checks.

## 3. GPU environment

Use Linux, an NVIDIA GPU, a matching driver/CUDA toolkit, a working C++/CUDA compiler, and a CUDA-enabled PyTorch build. Runtime extensions are compiled on first use. The measured environment is recorded in each result and requirements/measurement-environment.txt:

- GPU: NVIDIA GeForce RTX 5090.
- Python: 3.12.
- PyTorch: 2.12.0+cu130; CUDA: 13.0.
- NumPy, SciPy, Ninja, packaging and Numba: see the recorded lock file.
- The CSR comparison also requires CuPy; chfem uses its separate native build.

The lock file records the measurement environment, rather than a replacement for installation of the correct CUDA PyTorch build. Install that build and its Triton dependency according to the PyTorch distribution for your platform. Install the remaining pinned packages separately. Use the uv run --no-sync commands below so worker processes inherit the installed environment; VOXEL_PYTHON may explicitly select its Python executable.

The large stencil cases need substantially more GPU memory than the proposed solver. Keep the GPU exclusive during timing. Geometry generation, JIT compilation, warm-up and I/O are outside the measurement window.

## 4. Rerun chapter benchmarks

From the repository root on Linux (run uv run --no-sync python run.py prepare-inputs first):

    chmod +x run
    uv run --no-sync ./run chapter-study --phase launch --cases sphere,grosmont,woven --grid 192 --variants stencil27_jacobi,execution_fp32 --repeats 3 --out runs/table3

For the controlled woven optimization chain:

    uv run --no-sync ./run chapter-study --phase launch --cases woven --grid 192 --variants stencil27_fft,modal_fft,hybrid_fft,background_fft,execution_fp64,execution_fp32 --repeats 3 --out runs/optimization

For a small worker smoke test before a large campaign:

    uv run --no-sync ./run chapter-study --phase worker --case woven --grid 32 --variant execution_fp32 --repeat 0 --out runs/smoke

For a fresh high-resolution convergence point:

    uv run --no-sync ./run chapter-study --phase launch --cases woven --grid 480 --variants execution_fp32 --repeats 3 --out runs/convergence480

The exact older grid-cost jobs, method choices and reference media are recorded under campaigns/chapter/manuscript_evidence/review_results/grid_refinement_20261023/remote/plan.json and campaigns/chapter/manuscript_evidence/review_results/stencil27_baseline_20261023/formal/. Paths inside saved plans describe the original machine; use the included drivers and select a new runs/ destination. The currently regenerated CSVs always use the archived campaign, not results from a new run.

For Appendix D's 54-run active-fraction sweep:

    uv run --no-sync ./run active-sweep

It writes runs/active_fraction_128_20261024 and refuses to overwrite an existing plan.

## 5. Rerun yarn-fraction benchmarks

From the repository root:

    chmod +x run
    uv run --no-sync ./run woven-fraction --phase launch --grid 192 --fractions all --repeats 3 --out runs/woven_fraction

The driver executes all three configurations at each fraction and rotates their order. The corresponding analyzer is campaigns/woven_fraction/revision/experiments/woven_fraction_20261025/analyze.py. Its input directory must contain raw/results and raw/logs; the saved campaign demonstrates that layout.

## 6. Measurement conventions

Each full workflow solves six unit engineering-strain loads under periodic zero-mean constraints. Stiffness actions and PCG vectors use FP64. The proposed implementation uses FP32 FFT preconditioning and verifies the final FP64 Euclidean true residual against 1e-8. The workflow includes material construction, operator preparation, right-hand sides, solves, residual checks, stress recovery and cleanup. A full same-grid warm-up precedes measurement.

Each configuration has three fresh-process repetitions. Report median complete-call time and its min/max range. Peak memory is the maximum 20-ms NVML process sample, including CUDA context, libraries and retained allocations. Kernel timings in Appendix D use CUDA events with resident data, outside the full-workflow window.

Optimization-chain configurations share a phase-based reference and 327 total iterations. The cost-scaling campaign uses sampled references for stored stencils and phase-based references for modal solvers; it does not isolate representation changes alone. Yarn-fraction configurations share the same fixed matrix-based reference and iteration sequence at each fraction.

## 7. External-reference scope

The frozen campaigns include chfem result JSON, sampled-memory logs and canonical isotropic input arrays. Its maintained source copy and license are in baselines/. The recorded reference tensors are transformed into the project's Voigt convention before comparison. Rebuilding the native chfem executable is platform dependent; follow its included build documentation. The release check regenerates comparisons from the archived external outputs and does not claim a fresh chfem build or GPU rerun.

The Grosmont input is the paper's 400³ crop beginning at zero-based (z,y,x)=(550,0,0) and its nearest-neighbor 192³ resampling. The original full 1024³ image is not included. The manuscript cites Andrä et al.'s digital-rock dataset; original dataset attribution remains applicable.

## 8. What the release verification covers

The release checks numerical aggregation, source/data checksums, the CPU FI check, the CPU material-state example, and 21 small-grid geometry checks. It does not rerun the original GPU timing campaigns. Full-figure artwork and the manuscript LaTeX are maintained separately; this package regenerates the numerical data for the paper's tables and result plots.
