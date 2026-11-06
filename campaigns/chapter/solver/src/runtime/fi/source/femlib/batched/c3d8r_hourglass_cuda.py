"""Lazy loader for the native CUDA C3D8R hourglass comparison kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_c3d8r_hourglass_cuda_extension(*, verbose: bool = False):
    """Build and load the local CUDA extension on first use."""

    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / "csrc"
    return load(
        name="voxel_acdm_c3d8r_hourglass_cuda",
        sources=[
            str(source_dir / "c3d8r_hourglass.cpp"),
            str(source_dir / "c3d8r_hourglass_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )


def c3d8r_two_stage_apply_cuda(
    u_flat,
    stiffness,
    hourglass_scale,
    geometry,
    *,
    grid_shape: tuple[int, int, int],
    block_size: int = 128,
    variant: str = "thread",
    verbose_build: bool = False,
):
    """Apply the native CUDA element-force and node-gather kernels."""

    variants = {"thread": 0, "atomic": 1}
    if variant not in variants:
        raise ValueError(f"unknown CUDA kernel variant {variant!r}")
    extension = load_c3d8r_hourglass_cuda_extension(verbose=verbose_build)
    nz, ny, nx = map(int, grid_shape)
    return extension.apply(
        u_flat,
        stiffness,
        hourglass_scale,
        geometry,
        nz,
        ny,
        nx,
        int(block_size),
        variants[variant],
    )


def c3d8r_rhs_cuda(
    stiffness,
    geometry,
    macro_strain,
    *,
    grid_shape: tuple[int, int, int],
    block_size: int = 128,
    variant: str = "thread",
    verbose_build: bool = False,
):
    """Assemble the periodic C3D8R right-hand side without PyTorch intermediates."""

    variants = {"thread": 0, "atomic": 1}
    if variant not in variants:
        raise ValueError(f"unknown CUDA kernel variant {variant!r}")
    extension = load_c3d8r_hourglass_cuda_extension(verbose=verbose_build)
    nz, ny, nx = map(int, grid_shape)
    return extension.rhs(
        stiffness,
        geometry,
        macro_strain,
        nz,
        ny,
        nx,
        int(block_size),
        variants[variant],
    )


def c3d8r_avg_stress_cuda(
    u_flat,
    stiffness,
    geometry,
    macro_strain,
    *,
    grid_shape: tuple[int, int, int],
    block_size: int = 128,
    verbose_build: bool = False,
):
    """Recover batch-average center stress with a two-stage CUDA reduction."""

    extension = load_c3d8r_hourglass_cuda_extension(verbose=verbose_build)
    nz, ny, nx = map(int, grid_shape)
    return extension.avg_stress(
        u_flat,
        stiffness,
        geometry,
        macro_strain,
        nz,
        ny,
        nx,
        int(block_size),
    )


def c3d8r_fft_spectral_multiply_cuda(
    symbol_inverse,
    rhs_spectrum,
    *,
    block_size: int = 256,
    verbose_build: bool = False,
):
    """Apply a batched complex 3x3 FFT symbol in one native CUDA launch."""

    if block_size not in {64, 128, 256, 512}:
        raise ValueError("block_size must be one of 64, 128, 256, or 512")
    extension = load_c3d8r_hourglass_cuda_extension(verbose=verbose_build)
    return extension.fft_spectral_multiply(
        symbol_inverse,
        rhs_spectrum,
        int(block_size),
    )
