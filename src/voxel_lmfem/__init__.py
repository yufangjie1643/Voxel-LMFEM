"""Voxel-LMFEM: periodic elastic homogenization on rectangular HEX8 grids."""

__version__ = "0.1.2"


def homogenize(phase_id, materials, domain_size, *, orientation=None, backend="cuda", warmup=False):
    """Solve six unit-strain loads; see docs/usage.md for input conventions."""
    from .solver import homogenize as solve
    return solve(phase_id, materials, domain_size, orientation=orientation, backend=backend, warmup=warmup)


__all__ = ["homogenize", "__version__"]
