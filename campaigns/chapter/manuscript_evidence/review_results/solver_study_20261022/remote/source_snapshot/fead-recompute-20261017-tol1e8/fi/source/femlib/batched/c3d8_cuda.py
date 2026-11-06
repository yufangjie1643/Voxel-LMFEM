"""Lazy loader and Python wrappers for the native CUDA C3D8 kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_VALID_BLOCK_SIZES = {64, 128, 256, 512}


def _validate_block_size(block_size: int) -> int:
    block_size = int(block_size)
    if block_size not in _VALID_BLOCK_SIZES:
        raise ValueError("block_size must be one of 64, 128, 256, or 512")
    return block_size


@lru_cache(maxsize=1)
def load_c3d8_cuda_extension(*, verbose: bool = False):
    """Build and load the local C3D8 CUDA extension on first use."""

    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / "csrc"
    return load(
        name="voxel_acdm_c3d8_cuda",
        sources=[
            str(source_dir / "c3d8.cpp"),
            str(source_dir / "c3d8_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )


def c3d8_stencil_apply_cuda(
    u_flat,
    S_flat,
    *,
    grid_shape: tuple[int, int, int],
    block_size: int = 256,
    verbose_build: bool = False,
):
    """Apply a batched periodic C3D8 stencil.

    ``u_flat`` and the returned tensor use the solver's node-major layout
    ``(B, 3*N)``. ``S_flat`` uses either the full ``(B, 27, 9, Nz, Ny, Nx)``
    layout or the symmetric compressed ``(B, 14, 9, Nz, Ny, Nx)`` layout.
    """

    extension = load_c3d8_cuda_extension(verbose=verbose_build)
    nz, ny, nx = map(int, grid_shape)
    return extension.apply(
        u_flat,
        S_flat,
        nz,
        ny,
        nx,
        _validate_block_size(block_size),
    )


def c3d8_rhs_cuda(
    stiffness,
    geometry,
    macro_strain,
    *,
    grid_shape: tuple[int, int, int],
    block_size: int = 128,
    variant: str = "thread",
    verbose_build: bool = False,
):
    """Assemble the periodic C3D8 right-hand side.

    ``geometry`` is ``(inv4dx, inv4dy, inv4dz, element_volume)``. The
    deterministic ``thread`` variant writes element vectors then gathers
    them at nodes. The explicitly selected ``atomic`` variant accumulates
    element contributions directly into the nodal result.
    """

    variants = {"thread": 0, "atomic": 1}
    if variant not in variants:
        raise ValueError(f"unknown CUDA kernel variant {variant!r}")
    extension = load_c3d8_cuda_extension(verbose=verbose_build)
    nz, ny, nx = map(int, grid_shape)
    return extension.rhs(
        stiffness,
        geometry,
        macro_strain,
        nz,
        ny,
        nx,
        _validate_block_size(block_size),
        variants[variant],
    )


def c3d8_avg_stress_cuda(
    u_flat,
    stiffness,
    geometry,
    macro_strain,
    *,
    grid_shape: tuple[int, int, int],
    block_size: int = 128,
    verbose_build: bool = False,
):
    """Recover the batch-average C3D8 stress with a two-stage reduction."""

    extension = load_c3d8_cuda_extension(verbose=verbose_build)
    nz, ny, nx = map(int, grid_shape)
    return extension.avg_stress(
        u_flat,
        stiffness,
        geometry,
        macro_strain,
        nz,
        ny,
        nx,
        _validate_block_size(block_size),
    )


def c3d8_fft_spectral_multiply_cuda(
    symbol_inverse,
    rhs_spectrum,
    *,
    block_size: int = 256,
    verbose_build: bool = False,
):
    """Apply a batched complex 3x3 FFT symbol in one native CUDA launch."""

    extension = load_c3d8_cuda_extension(verbose=verbose_build)
    return extension.fft_spectral_multiply(
        symbol_inverse,
        rhs_spectrum,
        _validate_block_size(block_size),
    )
