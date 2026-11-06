# Voxel-LMFEM

Memory-efficient GPU finite element homogenization. LM denotes low memory; FEM denotes finite element method.

Code and measured evidence for **Memory-efficient GPU finite element homogenization of microstructures with spatially varying anisotropy**, submission snapshot **25 October 2026**.

Fixed version: **fead-2026-10-25-v2**. This public package is the reproducibility entry for the manuscript. It contains the fully integrated HEX8 solver, benchmark definitions and inputs, measured six-load tensors, timing records, NVML telemetry, and CPU aggregation scripts.

## Regenerate the paper's numbers

Python 3.12 was used for the release check. Clone the fixed version into a short path (for example C:/work/Voxel-LMFEM on Windows). Git for Windows needs long-path support for preserved historical source directories:

    git -c core.longpaths=true clone --branch fead-2026-10-25-v2 https://github.com/yufangjie1643/Voxel-LMFEM.git
    cd Voxel-LMFEM

From the repository root:

    python -m venv .venv
    # Activate .venv using your operating system's usual command.
    python -m pip install -r repro/requirements-cpu.txt
    python repro/scripts/reproduce_paper_tables.py

No GPU is needed. The script verifies SHA-256 checksums, checks residuals and repeat counts, recomputes discrepancies and statistics from the saved measurements, and verifies the printed table values. It writes:

- [Current table and figure index](repro/generated/TABLES.md)
- [Six-row Table 3](repro/generated/table_3_solver_summary.csv)
- [480³-reference convergence](repro/generated/figure_8_grid_convergence.csv)
- [Headline checks](repro/generated/headline_checks.json)
- [Verification report](repro/generated/verification.json)

The machine-readable correspondence is [repro/table_manifest.json](repro/table_manifest.json). Use this map for the submitted paper. Historical summaries inside the campaigns retain their original labels; the former 19-row comparison and 256³ convergence reference are superseded.

## What is included

| Location | Contents |
|---|---|
| campaigns/chapter/solver | FI solver, stored-stencil baselines, chapter/grid/active-fraction drivers, Grosmont inputs, material-state fixture |
| campaigns/chapter/manuscript_evidence/review_results | Main comparisons, reference outputs, convergence through 480³, cost scaling, nine active fractions |
| campaigns/woven_fraction/solver | Source snapshot for the 21 yarn-fraction experiment |
| campaigns/woven_fraction/revision/review_results/woven_fraction_20261025 | 189 measured runs, telemetry, geometry and small-grid checks |
| repro/scripts | Current CPU regeneration entry |
| repro/file_manifest.json | SHA-256 inventory of the frozen payload and reproduction scripts |
| repro/provenance | Source-to-package mapping and release verification |

The two campaigns preserve their corresponding numerical sources. The chapter snapshot includes the later isotropic-input extension of its driver; the exact earlier woven driver is preserved under solver/provenance/chapter_woven_20261024/. Its differences concern isotropic input dispatch. The 480³ and active-fraction records match the included chapter numerical source.

The portable launchers use python3, or the VOXEL_PYTHON environment variable. Their menus contain paper-related commands only. Unrelated FE²/CMC applications and private development history are excluded. The material-law fixture under src/multiscale/progressive is included only for Appendix C's CPU state-update example.

## Numerical checks and fresh GPU experiments

See [repro/REPRODUCING.md](repro/REPRODUCING.md) for tested CPU checks, GPU commands, environment details, timing definitions and external-reference scope.

Regeneration reads archived measurements; it does not remeasure GPU performance. Fresh GPU timings depend on hardware, drivers and the software environment. The released performance records were obtained with an RTX 5090, PyTorch 2.12.0+cu130 and CUDA 13.0.

## Attribution and terms

See [NOTICE.md](NOTICE.md). Third-party license files are preserved. This release does not assign a new license to author-owned code. The measured data and source provenance identify the benchmark inputs and external references used by the paper.
