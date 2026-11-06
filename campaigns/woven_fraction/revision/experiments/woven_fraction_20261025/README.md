# Woven volume-fraction experiment

This experiment supports Figure 11 in Section 5.4 of the 25 October 2026 submission. It measures the storage and runtime effects of Section 3.2 on geometrically resolved woven RVEs with spatially varying yarn orientations. Three configurations differ only in material storage and background decomposition. All use the same modal stiffness formulation and optimized FFT–PCG execution.

## Inputs

`geometry.py` is the deterministic geometry and material definition. The 192³ grid spans 1 × 1 × 0.4. Four periodic interlaced yarns have sinusoidal centre lines, transverse semiaxis 0.18 and undulation amplitude 0.14. The vertical semiaxis changes with the prescribed yarn fraction. The 21 nominal fractions run from 5% to 33% in 1.4-percentage-point increments. The stored actual voxel fractions are used in figures.

All matrix voxels share an isotropic tensor. Yarn tensors are rotated individually using the local centre-line tangent. Yarn volume in the RVE is distinct from the fiber fraction within a homogenized yarn. The construction is an analytical woven geometry, not an experimental image.

Geometry validation checks same-family separation, crossing-yarn separation, vertical periodic separation and disjoint voxel membership. The fixed material parameters match the manuscript table. Only geometry changes across fractions.

## Configuration controls

- `elementwise`: one tensor per voxel; direct addressing.
- `hybrid`: shared matrix tensor, individually oriented yarn tensors and an int32 element-to-row map.
- `background`: the same hybrid material pool plus the homogeneous-background/contrast action.

All configurations use the same normalized matrix tensor as the FFT reference. Physical operators, PCG vectors, reductions and residual checks use FP64; FFT arrays and cached coefficients use FP32. Six unit engineering-strain loads are solved to a recomputed relative residual of 10⁻⁸. No forced tensor symmetrization is applied to the reported effective stiffness.

One independent process provides one measured six-load solution after a complete six-load warmup at the same grid size. There are three repetitions per fraction/configuration. The runner rotates configuration order and reverses fraction order on the second repetition. It rejects concurrent GPU workloads and samples process GPU memory with NVML every 20 ms during the marked measurement window.

Complete-call time includes material construction, operator/reference preparation, RHS construction, PCG, true-residual checks, stress recovery and cleanup. It excludes geometry generation, compilation, warmup and file I/O. Aggregation uses median time, min–max time range and maximum sampled process-memory peak across repetitions. Material-array bytes are retained separately, not used in place of NVML memory.

## Reproduction

In this public package, use campaigns/woven_fraction/solver as the working directory and its run entry point. Select a new output directory when reproducing the experiment; the runner preserves completed records.

```sh
./run woven-fraction --phase geometry --grid 192 --out runs/woven_fraction_reproduce
./run woven-fraction --phase launch --grid 32 --fractions 0.05,0.19,0.33 --repeats 1 --out runs/woven_fraction_reproduce/validation
./run woven-fraction --phase launch --grid 192 --repeats 3 --out runs/woven_fraction_reproduce
```

The 32³ validation compares all three configurations at the low, middle and high fractions before the full study. Inspect their iteration sequences and relative Frobenius differences of the unsymmetrized effective stiffness. Actual validation results are distributed alongside the raw evidence.

For local aggregation, place the captured run directory at `review_results/woven_fraction_20261025/raw/` under the manuscript revision, then run:

```sh
python experiments/woven_fraction_20261025/analyze.py review_results/woven_fraction_20261025
```

For the current manuscript's numerical figure data, use the top-level repro/scripts/reproduce_paper_tables.py entry. It writes figure_11_woven_fraction.csv and figure_11_gains.csv. Manuscript artwork is maintained separately.

## Evidence mapping

- `raw/results/`: full timing, stage records, iteration sequences, material storage, raw effective stiffness and source hashes.
- `raw/logs/*_telemetry.json`: NVML samples and measurement markers.
- `raw/geometry_validation.json`: all 21 geometric admissibility checks.
- `raw/validation/paired_validation.json`: the 32³ paired solver check.
- `summary.json`, `summary.csv`: aggregate figure data and cross-configuration checks.
- `images/manifest.json`: exact voxel render provenance.

Figure 11 reports complete solver memory and runtime over woven fractions. The layered-cell active-fraction test appears as Figure D.1 in Appendix D and measures background-kernel sensitivity over a wider fraction range.
