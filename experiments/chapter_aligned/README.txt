Chapter-aligned solver comparison

Purpose
-------
Connect the periodic HEX8 / PCG formulation to three controlled changes:
modal constitutive action, shared material/background action, and compact GPU
execution. The conventional assembled baseline uses CuPy CSR and CuPy CG.

Canonical source and launch
---------------------------
Source: experiments/chapter_aligned/ in this checkout.
Entry: ./run chapter-study
New outputs belong in runs/<experiment-name>/; existing outputs are not replaced.

./run chapter-study --phase validate --out runs/chapter_csr_validation
./run chapter-study --phase launch --grid 16 --repeats 1 --out runs/chapter_chain_smoke
./run chapter-study --phase launch --grid 192 --cases woven --out runs/chapter_chain_192
./run chapter-study --phase launch --grid 192 --cases sphere,grosmont --variants csr_jacobi,stencil27_jacobi,stencil27_fft,execution_fp32 --out runs/chapter_isotropic_192

Controls
--------
All stages use the same eight-point HEX8 discretization, periodic grid,
six unit engineering strains, FP64 physical stiffness and solver vectors,
and final recomputed Euclidean relative residual <= 1e-8.
FFT stages use one normalized phase-based isotropic reference and FP64
FFT arithmetic; only execution_fp32 changes the FFT precision.
All measurements follow a complete same-grid warmup in a fresh process.
There are three repetitions per configuration. Complete-call and individual
stage durations are both recorded. The external NVML process-memory sampler
is shared with the existing solver study, at a target interval of 20 ms.

CSR baseline
------------
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

Verification
------------
A 4x5x6 grid with unequal cell lengths and random fully anisotropic SPD
materials is independently assembled in NumPy. Tests compare the full CSR
matrix, symmetry, the three rigid translations, all six RHS vectors,
solutions from a CPU direct solve, and mean stresses. A small woven campaign
also compares the assembled and matrix-free implementations end to end.

Data-to-manuscript mapping
-------------------------
The root reproduction command regenerates all published table and plot data.
See repro/REPRODUCING.md and repro/table_manifest.json for exact inputs,
aggregation conventions and the current manuscript mapping.

The direct eight-point operator is an isolated control for modal evaluation.
The 14-point stencil is a separate symmetry comparison against the 27-point
stencil, not an additional step after replacing the stencil with modal action.
