"""Voxel-LMFEM: periodic elastic homogenization on rectangular HEX8 grids."""

__version__ = "0.1.3"


def homogenize(phase_id, materials, domain_size, *, orientation=None, backend="cuda", warmup=False):
    """Solve six unit-strain loads; see README.md for input conventions."""
    from .solver import homogenize as solve
    return solve(phase_id, materials, domain_size, orientation=orientation, backend=backend, warmup=warmup)


def import_model(input_path, materials_path, out, **options):
    """Import an NPY/NPZ voxel model; see README.md for formats and options."""
    from .importers import import_model as convert
    return convert(input_path, materials_path, out, **options)


__all__ = ["homogenize", "import_model", "__version__"]
