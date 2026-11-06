"""Lazy loader and thin wrapper for fused native CUDA PCG updates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_pcg_cuda_extension(*, verbose: bool = False):
    """Build and load the local Jacobi-PCG CUDA extension on first use."""

    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / "csrc"
    return load(
        name="voxel_acdm_pcg_cuda",
        sources=[
            str(source_dir / "pcg.cpp"),
            str(source_dir / "pcg_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )


def pcg_jacobi_step_cuda(
    u,
    r,
    p,
    Ap,
    M_inv,
    rz,
    rhs_norm,
    best_u,
    best_res,
    res,
    workspace,
    state,
    *,
    block_size: int = 256,
    verbose_build: bool = False,
):
    """Run one fused native CUDA Jacobi-PCG state update."""

    extension = load_pcg_cuda_extension(verbose=verbose_build)
    return extension.jacobi_step(
        u,
        r,
        p,
        Ap,
        M_inv,
        rz,
        rhs_norm,
        best_u,
        best_res,
        res,
        workspace,
        state,
        int(block_size),
    )
