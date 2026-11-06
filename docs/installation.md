# Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. Python 3.10 or newer is required; the commands below select Python 3.12, matching the release checks. Repository paths are checked for Windows compatibility; use the ordinary clone command below.

    git clone https://github.com/yufangjie1643/Voxel-LMFEM.git
    cd Voxel-LMFEM
    uv venv --python 3.12
    uv pip install -e .
    uv run --no-sync voxel-lmfem doctor

uv venv creates the local .venv, uv pip installs dependencies from pyproject.toml, and uv run --no-sync runs commands in that installed environment. Shell activation is optional. This workflow keeps the chosen PyTorch backend when running examples and benchmarks. See the [uv environment guide](https://docs.astral.sh/uv/pip/environments/).

The base installation uses NumPy and SciPy. Its independent CPU reference supports up to 4096 voxels. PyTorch and a GPU are not required for this reference, examples or table regeneration.

For development:

    uv pip install -e ".[dev]"
    uv run --no-sync python -m pytest -q

For the original independent FI validation, additionally install a CPU or CUDA build of PyTorch:

    uv pip install --torch-backend=cpu --reinstall-package torch -e ".[cpu-checks]"
    uv run --no-sync python run.py check fi --output runs/checks/fi.json

## GPU installation

Use Linux with a supported NVIDIA GPU, driver, CUDA toolkit and C++ compiler. In the .venv created above, let uv choose the PyTorch backend compatible with the installed driver:

    uv pip install --torch-backend=auto --reinstall-package torch -e ".[gpu]"
    uv run --no-sync voxel-lmfem doctor --require-cuda

For an explicitly selected CUDA 13.0 backend, use uv pip install --torch-backend=cu130 --reinstall-package torch -e ".[gpu]". The reinstall flag also replaces a previously installed CPU build when switching backends. Automatic selection falls back to CPU when no supported GPU is detected; doctor --require-cuda confirms CUDA availability. See the [official uv/PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection) and [PyTorch installation selector](https://pytorch.org/get-started/locally/) for backend compatibility.

The GPU numerical modules compile CUDA extensions on first use. Ninja, packaging and Numba are included in the gpu extra. The stored-stencil baseline also uses Triton from the Linux PyTorch environment. The optional CSR baseline requires [a compatible CuPy installation](https://docs.cupy.dev/en/stable/install.html).

The paper measurements used RTX 5090 / PyTorch 2.12.0+cu130 / CUDA 13.0. requirements/measurement-environment.txt preserves the recorded package versions. It is a provenance record, not a cross-platform environment lock.

Launch repository commands with uv run --no-sync python run.py COMMAND. On Linux, uv run --no-sync ./run COMMAND also passes the environment to worker processes. VOXEL_PYTHON can explicitly select the interpreter for the Unix ./run launcher. Compiled extensions default to ~/.cache/voxel-lmfem/torch_extensions; TORCH_EXTENSIONS_DIR overrides that location.

## Installation from a wheel

The release wheel includes the public API and numerical core. It supports solve, example, doctor, and the FI CPU check outside a checkout. Paper reproduction, benchmark campaigns and the material-state fixture also need the full repository.

    uv venv --python 3.12
    uv pip install voxel_lmfem-0.1.1-py3-none-any.whl
    uv run --no-project python -m voxel_lmfem --help

When using a wheel with a separate checkout, run checkout-specific commands from that checkout or set VOXEL_LMFEM_ROOT to its absolute path. GPU performance and native extension compatibility must be checked on the target machine.
