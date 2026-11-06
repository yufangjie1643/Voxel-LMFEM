# Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and create a Python 3.12 environment:

```bash
git clone https://github.com/yufangjie1643/Voxel-LMFEM.git
cd Voxel-LMFEM
uv venv --python 3.12
uv pip install -e .
uv run --no-sync voxel-lmfem doctor
```

uv manages `.venv`; `uv run --no-sync` uses its installed packages. The base dependencies are NumPy and SciPy. Python 3.10 and newer are supported.

## GPU environment

Use Linux with an NVIDIA GPU, driver, CUDA toolkit and C++ compiler:

```bash
uv pip install --torch-backend=auto --reinstall-package torch -e ".[gpu]"
uv run --no-sync voxel-lmfem doctor --require-cuda
```

For CUDA 13.0, select `--torch-backend=cu130`. See the [uv/PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for supported backends. CUDA extensions compile on first use. The GPU extra includes Ninja, packaging and Numba; the CSR baseline also needs [CuPy](https://docs.cupy.dev/en/stable/install.html).

The paper's environment is recorded in `repro/environment.txt`: RTX 5090, Python 3.12, PyTorch 2.12.0+cu130 and CUDA 13.0.

Compiled extensions are cached in `~/.cache/voxel-lmfem/torch_extensions`. Set `TORCH_EXTENSIONS_DIR` to choose another cache location.

## Prepare CUDA extensions

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

## Development tools

```bash
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest -q
```

## Wheel installation

Download the wheel from [Releases](https://github.com/yufangjie1643/Voxel-LMFEM/releases), then:

```bash
uv venv --python 3.12
uv pip install voxel_lmfem-0.1.2-py3-none-any.whl
uv run --no-project python -m voxel_lmfem --help
```

The wheel provides the solver API and CLI. Benchmark campaigns, material-state fixtures and paper reproduction use the full repository checkout. `VOXEL_LMFEM_ROOT` selects a separate checkout when needed.
