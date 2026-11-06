# Installation

Python 3.10 or newer is required; release checks use Python 3.12. Use a short checkout path on Windows because the retained archive contains long historical paths.

    git -c core.longpaths=true clone https://github.com/yufangjie1643/Voxel-LMFEM.git
    cd Voxel-LMFEM
    python -m venv .venv

Activate .venv with source .venv/bin/activate on Linux/macOS or .venv\Scripts\Activate.ps1 in Windows PowerShell, then:

    python -m pip install -e .
    voxel-lmfem doctor

The base installation uses NumPy and SciPy. Its independent CPU reference supports up to 4096 voxels. PyTorch and a GPU are not required for this reference, examples or table regeneration.

For development:

    python -m pip install -e ".[dev]"
    python -m pytest -q

For the original independent FI validation, additionally install a CPU or CUDA build of PyTorch:

    python -m pip install -e ".[cpu-checks]"
    python run.py check fi --output runs/checks/fi.json

## GPU installation

Use Linux with a supported NVIDIA GPU, driver, CUDA toolkit and C++ compiler. Install the CUDA-enabled PyTorch build appropriate for the system using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then:

    python -m pip install -e ".[gpu]"
    voxel-lmfem doctor --require-cuda

The GPU numerical modules compile CUDA extensions on first use. Ninja, packaging and Numba are included in the gpu extra. The stored-stencil baseline also uses Triton from the Linux PyTorch environment. The optional CSR baseline requires [a compatible CuPy installation](https://docs.cupy.dev/en/stable/install.html).

The paper measurements used RTX 5090 / PyTorch 2.12.0+cu130 / CUDA 13.0. requirements/measurement-environment.txt preserves the recorded package versions. It is a provenance record, not a cross-platform environment lock.

VOXEL_PYTHON selects the interpreter for the Unix ./run launcher. Compiled extensions default to ~/.cache/voxel-lmfem/torch_extensions; TORCH_EXTENSIONS_DIR overrides that location.

## Installation from a wheel

The release wheel includes the public API and numerical core. It supports solve, example, doctor, and the FI CPU check outside a checkout. Paper reproduction, benchmark campaigns and the material-state fixture also need the full repository.

    python -m pip install voxel_lmfem-0.1.0-py3-none-any.whl
    python -m voxel_lmfem --help

When using a wheel with a separate checkout, run checkout-specific commands from that checkout or set VOXEL_LMFEM_ROOT to its absolute path. GPU performance and native extension compatibility must be checked on the target machine.
