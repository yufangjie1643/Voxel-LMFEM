"""Public adapters to the verified FI GPU chain and independent CPU reference."""
import time
import numpy as np

from .inputs import material_module, validate
from .paths import configure


def homogenize(phase_id, materials, domain_size, *, orientation=None, backend="cuda", warmup=False):
    """Return effective stiffness in engineering-Voigt order 11,22,33,23,13,12.

    The CPU backend is an independent sparse direct reference for <=4096 voxels.
    The CUDA backend uses the paper's FP64 physical operator and FP32 FFT.
    Numerical modules maintain process-level configuration; run concurrent jobs
    in separate processes.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda'")
    phase, table, domain, orientation, labels, tensors, local = validate(
        phase_id, materials, domain_size, orientation)
    if backend == "cpu" and phase.size > 4096:
        raise ValueError("The CPU reference is limited to 4096 voxels; use the CUDA backend for larger cells")
    configure()
    start = time.perf_counter()
    if backend == "cpu":
        from independent_validation import assemble, fourth_order_rotate
        full = tensors[np.searchsorted(labels, phase.ravel())].copy()
        mf = material_module()
        for index in np.flatnonzero(local):
            selected = np.flatnonzero(phase.ravel() == labels[index])
            if selected.size:
                rotations = mf._orientation_values(orientation, phase.shape, selected)
                for element, rotation in zip(selected, rotations):
                    full[element] = fourth_order_rotate(tensors[index], rotation)
        voxel = domain / np.array(phase.shape[::-1])
        if np.array_equal(full, np.broadcast_to(full[0], full.shape)):
            # A uniform global tensor gives an identically zero periodic RHS
            # and zero fluctuation. Use that exact solution instead of dividing
            # quadrature/assembly roundoff by a mathematically zero load.
            CH = full[0].copy()
            residuals = [0.] * 6
            solution_kind = "homogeneous-exact-zero-fluctuation"
        else:
            K, rhs, u, CH, _ = assemble(full, phase.shape, voxel)
            residuals = [float(np.linalg.norm(rhs[:, j]-K@u[..., j].ravel()) /
                               max(np.linalg.norm(rhs[:, j]), 1e-300)) for j in range(6)]
            solution_kind = "independent-sparse-direct"
        result = dict(CH=CH.tolist(), true_residuals=residuals, backend="cpu-reference",
                      grid_shape=list(phase.shape), domain_size=domain.tolist(),
                      solution_kind=solution_kind,
                      scope="Independent eight-point HEX8 reference; exact zero fluctuation for uniform stiffness")
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install a CUDA-enabled PyTorch build and the GPU dependencies; see docs/installation.md") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --backend cpu for a small reference cell, "
                               "or install/configure the CUDA runtime.")
        import chain_study as cs
        import operator_adapter as latest
        b = cs.s.load("fi")
        latest.prior.install(b, cs)
        cs.a.configure(b, "fi", "combined")
        b.TOL = 1e-8
        b.gpu_guard()
        latest.install(b, "split_delta", 64)
        data = dict(phase_id=phase[None], grid_shape=phase.shape, domain_size=domain[None])
        if orientation is not None:
            data["orientation"] = orientation
        if warmup:
            b.workload(data, table, "direct", "mixed")
        start = time.perf_counter()
        result, _ = b.workload(data, table, "direct", "mixed")
        result.update(backend="cuda", storage=b._storage, operator=b._operator_metadata,
                      scope="FP64 fully integrated stiffness/PCG; FP32 FFT; six unit engineering strains")
        residuals = [r for call in result["calls"] for r in call["true_residuals"]]
    if not np.isfinite(residuals).all() or max(residuals) > 1e-8:
        raise RuntimeError(f"True-residual check failed: {residuals}")
    if not np.isfinite(result["CH"]).all():
        raise RuntimeError("Effective stiffness contains nonfinite values")
    result["max_true_residual"] = max(residuals)
    result["elapsed_seconds"] = time.perf_counter()-start
    result["voigt_order"] = ["11", "22", "33", "23", "13", "12"]
    return result
