# Woven-fraction experiment and inclusion decision

## Purpose and placement

The experiment tests the Section 3.2 claim that combining shared isotropic material data with individually oriented yarn tensors reduces storage, and that background reuse reduces repeated matrix-phase work. Include the measured complete-call time and NVML memory curves in a new Section 5.2. Move the existing grid-scaling subsection to 5.3 by insertion, without changing its evidence. English is authoritative; synchronize Chinese.

The previous 128³ layered active-fraction test is a supplementary operator sensitivity test, not evidence for fraction-dependent woven-memory savings. Retain it in Appendix D because it reaches higher active fractions and identifies the background-action crossover. Avoid repeating those curves in the main text.

## Geometry and material

Use the existing four-yarn periodic analytical weave, with sinusoidal centre lines, vertical elliptical sections and spatially varying tangent-based orthotropy. This is a geometrically resolved woven RVE, not an imaged experimental specimen. The variable is yarn volume in the RVE, not fiber volume within each homogenized yarn.

- Domain: 1 × 1 × 0.4; grid: 192³.
- Fixed transverse semiaxis w = 0.18 and undulation amplitude a = 0.14.
- Change the vertical semiaxis b = f × 0.4 / (4πw).
- Twenty-one nominal fractions, 5.0–33.0% in 1.4 percentage-point increments. Plot measured voxel fractions.
- The raised amplitude, compared with the fixed-grid benchmark, permits this thickness range without yarn intersection. Verify both a cos(2πw) > b and a + b < 0.2, and check every voxel for multiple yarn membership.
- Matrix and yarn properties are unchanged from the manuscript material table.

## Controlled comparison

Three configurations share the same modal formulation, material interface, normalized matrix-phase FFT reference, FP32 FFT execution, FP64 physical operator and PCG vectors, fused GPU execution, periodic zero-mean constraint, six unit engineering strains and recomputed FP64 residual tolerance 10⁻⁸:

1. Elementwise tensors, direct element indexing.
2. Hybrid material pool and element-to-row indices, full-domain modal action.
3. The same hybrid pool plus the uniform-background/contrast action.

Each point/configuration uses three fresh processes, each preceded by one complete six-load same-grid warmup. Rotate configuration order and reverse the fraction order in the second repetition. Measure the entire call, including material construction, operator/reference setup, all loads, PCG, residual certification, stress recovery and cleanup. Exclude geometry generation, JIT compilation, warmup and file I/O. Use a common external 20-ms NVML sampler, requiring an otherwise idle GPU. Report median time with min–max range, and the largest sampled NVML process-memory peak across repetitions. Record material-array bytes separately for interpretation; do not substitute them for total solver memory.

First validate all 21 geometries on CPU at 192³ and compare the three solvers at 32³ for nominal fractions 5%, 19% and 33%. Then run 189 measured GPU solves at 192³. Compare all six raw effective-stiffness columns, iteration counts, references and input hashes.

## Evidence and claim map before manuscript edits

| Proposed statement | Required evidence | Destination | Current strength |
| --- | --- | --- | --- |
| Hybrid storage reduces solver memory over the sampled woven fractions | Paired process NVML records at every fraction, same input and reference | Main 5.2, memory panel | Pending measurement |
| Sharing alone has a measurable runtime cost or benefit | Paired elementwise/hybrid complete-call medians and ranges | Main 5.2, time panel | Pending measurement; report either direction |
| Background reuse changes runtime as yarn fraction increases | Hybrid/background pairs, matched iterations and residuals | Main 5.2, time panel | Pending measurement; no monotonicity presumed |
| All configurations solve the same discretized problem | Effective-stiffness errors, input/reference equality, true residuals | Main 5.2 concise text | Pending validation |

## Figure contract

Python workflow, using the existing manuscript plotting/export helpers. One figure: small geometry views at low, middle and high fractions establish the changing yarn thickness; two linear-axis plots show complete-call time and NVML peak versus the measured yarn fraction. No third memory definition, decorative curves or repeated speedup panel. Short caption, all interpretation in正文/body text. Width exactly 140 mm; editable SVG text, PDF via Inkscape, 600-dpi TIFF; axis labels 9 pt and ticks/legend 7 pt. Plot only measured data, preserve the voxel geometry, use consistent method colours.

## Source locations

Canonical remote code: `/home/yufan/code/voxel-mechanics/experiments/woven_fraction/`.

Canonical remote evidence: `/home/yufan/code/voxel-mechanics/runs/woven_fraction_20261025/`.

Local protocol/source copy: `experiments/woven_fraction_20261025/`; curated records and analysis: this directory.
