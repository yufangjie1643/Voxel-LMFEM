# FFT / PCG formula verification, 2026-10-20

This experiment compares new formulas with the measured fastest pipeline:
HG `combined` and FI `modal`. The frozen runtime, original modal experiment,
and pipeline experiment are dependencies; they are not modified here.

Variants:

- FI `analytic`: the exact FI Fourier symbol is diagonal plus rank one.
  Sherman–Morrison inverse action caches four real arrays and three 1D tables.
- HG `contrast64`: split the actual isotropic matrix background from active
  yarn contrast. Use the same Zeman reference as the baseline, whiten the
  system, and keep CG states in the real FFT half spectrum. All Krylov
  calculations use FP64/complex128. Only active contrast elements are stored
  and evaluated. The physical true residual is retained.
- HG `single`: single-reduction CG recurrence using the baseline FP32 FFT
  preconditioner. This is tested as a candidate, including its extra vector
  storage and finite-precision effects.
- `baseline`: the previously fastest full workflow for the selected element.

The mathematical derivation and CPU checks are in
`../formula_reduction_20261020/check_reductions.py` and
`../../review_results/formula_reduction_20261020/DERIVATION_zh.md`.
CUDA implementations are in `formula_kernels.cu`; the integration and CG
recurrences are in `formula_adapter.py`.

## Environment and reproducible commands

The recorded GPU host is `yufan-MS-7E47`, NVIDIA RTX 5090. Python is
`/home/yufan/uv-torch-py312/bin/python` (PyTorch 2.12.0+cu130).
The experimental working directory is `/home/yufan/code/reference_medium`.
Dependencies can be selected with:

```sh
export FEAD_RUNTIME=/home/yufan/code/runtime
export FEAD_MODAL=/home/yufan/code/hex8_modal
export FEAD_PIPELINE=/home/yufan/code/assembly_pipeline
```

Check the GPU is idle before running. Each command below uses a fresh process.
The orchestrator also checks for foreign GPU jobs and stops its own job if
another job appears. A completed evidence directory is protected against
accidental reruns: use a new directory for a new full experiment.

```sh
python formula_study.py --phase validate --kind fi --variant analytic
python formula_study.py --phase validate --kind hg --variant contrast64
python formula_study.py --phase validate --kind hg --variant single
python formula_study.py --phase validate --kind fi --variant analytic --even
python formula_study.py --phase validate --kind hg --variant contrast64 --even
python launch.py
```

`launch.py` starts the 192³ orchestrator in the background. Five pilots are
followed by all five configurations in three interleaved repetitions. Every
measurement is a complete six-load homogenization after same-size full
warmup. CPU geometry, disk IO, compilation, and warmup are excluded; GPU
material construction, operator/reference setup, RHS, solving, final FP64
true-residual certification, and stress averaging are included.

The independent small-model check uses conventional CPU CSR assembly and
direct solution, anisotropic SPD materials, rectangular voxels, odd and even
grids, and 20% active contrast elements. It tests Ku, RHS, displacements,
local stresses, average stiffness, the whitened operator, spectral energy,
and the physical residual norm via half-spectrum Parseval weights.

New HG solvers check recursive residuals every four executed iterations;
the baseline retains its frozen latched logical iteration reporting. Final
FP64 physical residuals and the original outer restart mechanism are used
for all configurations. Iteration counts therefore require that qualification.

After downloading the complete directory into
`../../review_results/formula_gpu_20261020/remote/`, regenerate all tables:

```sh
python summarize.py
```

The summary rejects missing runs, failed validation, differing input hashes,
changes to shared frozen sources, excessive true residuals, and mismatched
full unsymmetrized effective stiffness. It saves `recommended.json` from
measured full-task medians. This only selects this experiment's entry;
the previous production defaults are unchanged.

Run a selected candidate explicitly (fresh process, refuses an existing file):

```sh
python run_rve.py --kind fi --variant analytic --grid 192 --out custom_fi.json
python run_rve.py --kind hg --variant contrast64 --grid 192 --out custom_hg.json
python run_rve.py --kind hg --variant single --grid 192 --out custom_single.json
python run_rve.py --kind fi --grid 192 --out recommended_fi.json
```

`pack_evidence.py` archives completed raw JSON, logs, source files and a
per-file SHA-256 manifest. Numerical and performance conclusions belong in
the generated `REPORT_zh.md`, including slower candidates.

After formal measurement, the missing `os` import in the foreign-job stop
branch was repaired. The numerical code is unchanged. The byte-exact
measured `formula_study.py` is retained under `remote/measured_sources/`
in the evidence package and verified against recorded provenance.
