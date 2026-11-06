# Voxel-LMFEM

GPU finite element homogenization of voxel microstructures with spatially varying anisotropy. Voxel-LMFEM combines fully integrated HEX8 elements, compact material storage and FFT-preconditioned conjugate gradients.

[Installation](docs/installation.md) · [Usage](docs/usage.md) · [Paper reproduction](repro/README.md) · [中文](docs/README_zh.md)

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/yufangjie1643/Voxel-LMFEM.git
cd Voxel-LMFEM
uv venv --python 3.12
uv pip install -e .
uv run --no-sync voxel-lmfem example --out runs/example
uv run --no-sync voxel-lmfem solve --input runs/example/homogeneous.npz --materials runs/example/materials.json --backend cpu --output runs/example/result.json
```

The example computes the effective stiffness of a homogeneous cell. The [usage guide](docs/usage.md) describes the Python API, material definitions and input format.

## GPU execution

On Linux with an NVIDIA GPU and a CUDA toolkit:

```bash
uv pip install --torch-backend=auto --reinstall-package torch -e ".[gpu]"
uv run --no-sync voxel-lmfem doctor --require-cuda
uv run --no-sync voxel-lmfem solve --input model.npz --materials materials.json --backend cuda --output runs/model/result.json
```

The first GPU run compiles C++/CUDA extensions and can take several minutes. To prepare the paper solver with live compiler output, run:

```bash
uv run --no-sync voxel-lmfem build
```

The command also runs a small woven-cell check. Subsequent runs reuse the compiled extensions. See [CUDA preparation](docs/installation.md#prepare-cuda-extensions) for build options and logs.

## Paper reproduction

```bash
uv run --no-sync voxel-lmfem reproduce
```

This command regenerates the manuscript tables and figure data from the included measurements. See the [reproduction guide](repro/README.md) for datasets, benchmark commands and measurement protocols.

## Development

```bash
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest -q
```

[Project structure](docs/architecture.md) · [Development guide](docs/development.md) · [Changelog](docs/changelog.md)

## License and citation

[MIT License](LICENSE). [Third-party credits](docs/NOTICE.md). Use [CITATION.cff](CITATION.cff) to cite the software and accompanying paper.
