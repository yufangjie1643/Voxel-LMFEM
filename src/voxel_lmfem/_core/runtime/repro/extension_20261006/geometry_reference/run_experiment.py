# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9"]
# ///
"""Voxel-free polarization Galerkin prototype; independent periodic Q1 reference.

Unit square, straight z-invariant elliptical fibers, isotropic linear elasticity.
Mandel order: xx, yy, zz, sqrt(2) yz, sqrt(2) xz, sqrt(2) xy.
Run with uv run --script run_experiment.py. See README.md for assumptions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import time

# Set before importing BLAS, allow an explicit operator override.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import eval_jacobi, jv


@dataclass(frozen=True)
class Ellipse:
    a: float = 0.32
    b: float = 0.22
    cx: float = 0.0
    cy: float = 0.0

    @property
    def fraction(self):
        return math.pi * self.a * self.b


def stiffness(young=1.0, poisson=0.3):
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    out = np.eye(6) * (2 * mu)
    out[:3, :3] += lam
    return out


def modes(degree):
    """Real Zernike modes normalized to disk mean(phi_i phi_j) = delta_ij."""
    return [(n, m, kind) for n in range(degree + 1)
            for m in range(n % 2, n + 1, 2)
            for kind in (["cos"] if m == 0 else ["cos", "sin"])]


def basis_values(r, theta, basis):
    vals = []
    for n, m, kind in basis:
        radial = r**m * eval_jacobi((n - m) // 2, 0, m, 2 * r*r - 1)
        angular = np.cos(m * theta) if kind == "cos" else np.sin(m * theta)
        vals.append(np.sqrt((n + 1) * (1 if m == 0 else 2)) * radial * angular)
    return np.stack(vals, axis=-1)


def shape_transforms(k, ellipse, basis):
    """Exact unit-cell Fourier coefficients of supported Zernike functions.

    k is integer reciprocal-lattice coordinates, not physical wavenumbers.
    No indicator rasterization or quadrature is used here.
    """
    qx = 2 * np.pi * ellipse.a * k[:, 0]
    qy = 2 * np.pi * ellipse.b * k[:, 1]
    q = np.hypot(qx, qy)
    theta = np.arctan2(qy, qx)
    phase = np.exp(-2j * np.pi * (ellipse.cx * k[:, 0] + ellipse.cy * k[:, 1]))
    radial = {}
    for n in {v[0] for v in basis}:
        radial[n] = np.divide(jv(n + 1, q), q, out=np.zeros_like(q), where=q != 0)
        radial[n][q == 0] = 0.5 if n == 0 else 0.0
    out = []
    for n, m, kind in basis:
        angle = np.cos(m * theta) if kind == "cos" else np.sin(m * theta)
        norm = np.sqrt((n + 1) * (1 if m == 0 else 2))
        out.append(2 * ellipse.fraction * (-1j)**n * norm * radial[n] * angle * phase)
    return np.stack(out, axis=-1)


def green(k, c0):
    """Continuous strain Green tensor B (B^T C0 B)^-1 B^T, zero mean."""
    k = np.asarray(k, dtype=float)
    q = np.zeros((len(k), 3))
    q[:, :2] = k  # Overall 2*pi cancels in this order-zero operator.
    b = np.zeros((len(k), 6, 3))
    for d in range(3):
        b[:, d, d] = q[:, d]
    for row, (i, j) in enumerate([(1, 2), (0, 2), (0, 1)], 3):
        b[:, row, i] = q[:, j] / np.sqrt(2)
        b[:, row, j] = q[:, i] / np.sqrt(2)
    acoustic = np.einsum("kia,ij,kjb->kab", b, c0, b)
    zero = np.all(k == 0, axis=1)
    acoustic[zero] = np.eye(3)
    return b @ np.linalg.inv(acoustic) @ b.transpose(0, 2, 1)


def interaction_matrices(ellipses, degree, cutoffs, c0):
    """Accumulate Fourier shells, reusing sums for each cutoff.

    Unknown indexing: (fiber, local scalar mode, Mandel component).
    All six polarizations are retained, including odd local modes.
    """
    basis = modes(degree)
    p = len(basis) * len(ellipses)
    h = np.zeros((p, 6, p, 6))
    previous = 0
    started = time.perf_counter()
    for cutoff in sorted(cutoffs):
        # Row strips bound temporary memory without limiting the physical geometry.
        for lo in range(-cutoff, cutoff + 1, 16):
            ix, iy = np.meshgrid(np.arange(lo, min(lo + 16, cutoff + 1)),
                                 np.arange(-cutoff, cutoff + 1), indexing="ij")
            k = np.column_stack([ix.ravel(), iy.ravel()])
            k = k[np.max(np.abs(k), axis=1) > previous]
            if not len(k):
                continue
            f = np.concatenate([shape_transforms(k, e, basis) for e in ellipses], axis=1)
            g = green(k, c0)
            for a in range(6):
                for b in range(a, 6):
                    if not np.any(g[:, a, b]):
                        continue
                    weighted = g[:, a, b, None] * f
                    block = (f.conj().T @ weighted).real
                    h[:, a, :, b] += block
                    if a != b:
                        h[:, b, :, a] += block
        yield cutoff, h.reshape(p * 6, p * 6).copy(), time.perf_counter() - started
        previous = cutoff


def effective_modal(h, ellipses, max_degree, degree, c0, contrast):
    started = time.perf_counter()
    full_p, p = len(modes(max_degree)), len(modes(degree))
    scalar_indices = np.concatenate([np.arange(i * full_p, i * full_p + p)
                                     for i in range(len(ellipses))])
    indices = (scalar_indices[:, None] * 6 + np.arange(6)).ravel()
    a = h[np.ix_(indices, indices)].copy()
    rhs = np.zeros((len(indices), 6))
    if contrast == 1:
        return c0.copy(), {"solve_seconds": time.perf_counter() - started,
                           "relative_linear_residual": 0.0, "unknowns": len(indices)}
    if contrast <= 1:
        raise ValueError("Prototype requires a stiffer fiber: contrast > 1, or homogeneous 1.")
    delta_inv = np.linalg.inv((contrast - 1) * c0)
    for i, e in enumerate(ellipses):
        block = slice(i * p * 6, (i + 1) * p * 6)
        a[block, block] += np.kron(np.eye(p) * e.fraction, delta_inv)
        rhs[i * p * 6:i * p * 6 + 6] = e.fraction * np.eye(6)
    coeff = cho_solve(cho_factor(a), rhs)
    ceff = c0 + rhs.T @ coeff
    residual = np.linalg.norm(a @ coeff - rhs) / np.linalg.norm(rhs)
    return ceff, {"solve_seconds": time.perf_counter() - started,
                  "relative_linear_residual": float(residual), "unknowns": len(indices)}


def q1_element(n, c0):
    """Independent bilinear finite element, 3 displacement components, full integration."""
    ke, le = np.zeros((12, 12)), np.zeros((12, 6))
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for xi in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
        for eta in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            dx = corners[:, 0] * (1 + corners[:, 1] * eta) * n / 2
            dy = corners[:, 1] * (1 + corners[:, 0] * xi) * n / 2
            b = np.zeros((6, 12))
            b[0, 0::3], b[1, 1::3] = dx, dy
            b[3, 2::3], b[4, 2::3] = dy / np.sqrt(2), dx / np.sqrt(2)
            b[5, 0::3], b[5, 1::3] = dy / np.sqrt(2), dx / np.sqrt(2)
            weight = 1 / (4 * n*n)
            ke += (b.T @ c0 @ b) * weight
            le += (b.T @ c0) * weight
    return ke, le


def voxel_reference(n, ellipses, c0, contrast, rtol=1e-9, *, material_mask=None):
    started = time.perf_counter()
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, y = (ix + 0.5) / n - 0.5, (iy + 0.5) / n - 0.5
    inside = np.zeros((n, n), bool)
    for e in ellipses:
        dx = (x - e.cx + 0.5) % 1 - 0.5
        dy = (y - e.cy + 0.5) % 1 - 0.5
        inside |= (dx / e.a)**2 + (dy / e.b)**2 < 1
    if material_mask is not None:
        inside = np.asarray(material_mask, dtype=bool)
        if inside.shape != (n, n):
            raise ValueError("material_mask shape must match the reference grid")
    factors = np.where(inside, contrast, 1.0).ravel()
    nodes = np.stack([ix * n + iy, ((ix + 1) % n) * n + iy,
                      ((ix + 1) % n) * n + (iy + 1) % n,
                      ix * n + (iy + 1) % n], axis=-1).reshape(-1, 4)
    dofs = (nodes[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
    ke, le = q1_element(n, c0)
    rows = np.broadcast_to(dofs[:, :, None], (n*n, 12, 12)).ravel()
    cols = np.broadcast_to(dofs[:, None, :], (n*n, 12, 12)).ravel()
    data = (factors[:, None, None] * ke).ravel()
    size = 3 * n*n
    k = sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    k.eliminate_zeros()
    rhs = np.zeros((size, 6))
    for a in range(12):
        np.add.at(rhs, dofs[:, a], factors[:, None] * le[a])

    # Exact inverse of the *discrete homogeneous Q1* reference on mean-zero fields.
    # Build its block-circulant symbol from an impulse response, not the continuum Green tensor.
    offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    kernel = np.zeros((n, n, 3, 3))
    for a in range(4):
        for b in range(4):
            d = (offsets[a] - offsets[b]) % n
            kernel[d[0], d[1]] += ke[a*3:a*3+3, b*3:b*3+3]
    symbol = np.fft.rfft2(kernel, axes=(0, 1))
    symbol[0, 0] = np.eye(3)
    symbol_inv = np.linalg.inv(symbol)
    symbol_inv[0, 0] = 0

    def precondition(v):
        spectrum = np.fft.rfft2(v.reshape(n, n, 3), axes=(0, 1))
        z = np.einsum("xyij,xyj->xyi", symbol_inv, spectrum)
        return np.fft.irfft2(z, s=(n, n), axes=(0, 1)).ravel()

    precond = LinearOperator((size, size), matvec=precondition)
    solution = np.zeros_like(rhs)
    counts, residuals = [], []
    assembly_seconds = time.perf_counter() - started
    for load in range(6):
        b = rhs[:, load]
        if np.linalg.norm(b) < 1e-13:
            counts.append(0)
            residuals.append(0.0)
            continue
        count = [0]

        def callback(_):
            count[0] += 1

        solution[:, load], info = cg(k, b, M=precond, rtol=rtol, atol=0,
                                     maxiter=2000, callback=callback)
        if info:
            raise RuntimeError(f"Voxel reference CG failed: N={n}, contrast={contrast}, info={info}")
        counts.append(count[0])
        residuals.append(float(np.linalg.norm(k @ solution[:, load] - b) / np.linalg.norm(b)))
    ceff = factors.mean() * c0 - rhs.T @ solution
    return ceff, {"grid": n, "unknowns": size - 3, "voxel_fraction": float(inside.mean()),
                  "cg_iterations": counts, "relative_linear_residual": max(residuals),
                  "assembly_seconds": assembly_seconds,
                  "total_seconds": time.perf_counter() - started,
                  "csr_bytes": k.data.nbytes + k.indices.nbytes + k.indptr.nbytes}


def validate():
    """Independent quadrature, exact uniform solution and discrete-reference checks."""
    c0 = stiffness()
    e = Ellipse(cx=0.071, cy=-0.043)
    basis = modes(6)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    r = (nodes + 1) / 2
    theta = np.arange(256) * 2 * np.pi / 256
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    w = np.broadcast_to((weights / 2 * r)[:, None] * (2 * np.pi / 256) * e.a * e.b,
                        rr.shape).ravel()
    values = basis_values(rr.ravel(), tt.ravel(), basis)
    gram = values.T @ (w[:, None] * values)
    mass_error = float(np.max(np.abs(gram / e.fraction - np.eye(len(basis)))))
    ks = np.array([[0, 0], [1, 0], [0, 2], [3, -2], [-4, 5], [7, 3]])
    xy = np.column_stack([e.cx + e.a * rr.ravel() * np.cos(tt.ravel()),
                          e.cy + e.b * rr.ravel() * np.sin(tt.ravel())])
    numeric = (np.exp(-2j * np.pi * ks @ xy.T) * w) @ values
    transform_error = float(np.max(np.abs(shape_transforms(ks, e, basis) - numeric)))
    g = green(ks, c0)
    projector_error = float(np.max(np.abs(g @ c0 @ g - g)))
    cv, vm = voxel_reference(8, [e], c0, 1)
    uniform_error = float(np.max(np.abs(cv - c0)))
    # Independent exact laminate: shear across x layers is harmonic; along y is arithmetic.
    mask = np.broadcast_to((np.arange(16) < 8)[:, None], (16, 16))
    laminate, _ = voxel_reference(16, [], c0, 10, material_mask=mask)
    exact_shear = np.array([c0[3, 3] * 5.5, c0[4, 4] / 0.55])
    laminate_error = float(np.max(np.abs(np.diag(laminate)[3:5] / exact_shear - 1)))
    # Rigid translation of a continuous fiber changes Fourier phases, not its effective tensor.
    h0 = next(interaction_matrices([Ellipse()], 2, [16], c0))[1]
    h1 = next(interaction_matrices([e], 2, [16], c0))[1]
    translation_error = float(np.max(np.abs(h0 - h1)))
    ch, _ = effective_modal(h0, [Ellipse()], 2, 2, c0, 1)
    homogeneous_modal_error = float(np.max(np.abs(ch - c0)))
    cw, _ = effective_modal(h0, [Ellipse()], 2, 2, c0, 1 + 1e-5)
    weak_contrast_error = float(np.linalg.norm(cw - (1 + Ellipse().fraction * 1e-5) * c0)
                               / np.linalg.norm(c0))
    assert mass_error < 1e-11, mass_error
    assert transform_error < 1e-11, transform_error
    assert projector_error < 1e-11, projector_error
    assert uniform_error < 1e-11, uniform_error
    assert laminate_error < 1e-9, laminate_error
    assert translation_error < 1e-12, translation_error
    assert homogeneous_modal_error < 1e-12, homogeneous_modal_error
    assert weak_contrast_error < 1e-10, weak_contrast_error
    return {"quadrature_mass_max_error": mass_error,
            "analytic_fourier_vs_quadrature_max_error": transform_error,
            "green_projector_max_error": projector_error,
            "uniform_q1_max_error": uniform_error,
            "exact_laminate_shear_relative_error": laminate_error,
            "continuous_translation_max_error": translation_error,
            "homogeneous_modal_max_error": homogeneous_modal_error,
            "weak_contrast_first_order_relative_error": weak_contrast_error}


def metrics(c):
    compliance = np.linalg.inv(c)
    return {"C11": float(c[0, 0]), "C22": float(c[1, 1]), "C33": float(c[2, 2]),
            "Gyz": float(c[3, 3] / 2), "Gxz": float(c[4, 4] / 2),
            "Gxy": float(c[5, 5] / 2), "Ex": float(1 / compliance[0, 0]),
            "Ey": float(1 / compliance[1, 1]), "Ez": float(1 / compliance[2, 2])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--cutoffs", type=int, nargs="+")
    parser.add_argument("--degrees", type=int, nargs="+")
    parser.add_argument("--grids", type=int, nargs="+")
    parser.add_argument("--cases", nargs="+", choices=["ellipse", "two_fibers", "tight_gap"])
    parser.add_argument("--extend-voxel", action="store_true",
                        help="Append missing --grids to existing results without rebuilding modal kernels")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.extend_voxel:
        if not args.grids:
            parser.error("--extend-voxel requires --grids")
        path = args.output / "results.json"
        result = json.loads(path.read_text())
        for case in result["cases"]:
            if args.cases and case["name"] not in args.cases:
                continue
            ellipses = [Ellipse(**e) for e in case["ellipses"]]
            for contrast in sorted({v["contrast"] for v in case["modal"]}):
                for grid in args.grids:
                    if any(v["contrast"] == contrast and v["grid"] == grid for v in case["voxel"]):
                        continue
                    c, info = voxel_reference(grid, ellipses, stiffness(case["matrix_E"], case["matrix_nu"]), contrast)
                    case["voxel"].append({"contrast": contrast, "C_mandel": c.tolist(), "metrics": metrics(c), **info})
                    print(f"{case['name']}: appended voxel {grid}, contrast {contrast:g}, {info['total_seconds']:.2f}s", flush=True)
                    path.write_text(json.dumps(result, indent=2) + "\n")
        return
    cutoffs = args.cutoffs or ([16, 32] if args.quick else [16, 32, 64, 128, 256])
    degrees = args.degrees or ([0, 2, 4] if args.quick else [0, 2, 4, 6, 8])
    grids = args.grids or ([16, 32] if args.quick else [32, 64, 128, 256])
    result = {"metadata": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "platform": platform.platform(),
                           "blas_threads": os.environ["OPENBLAS_NUM_THREADS"],
                           "scope": "3D linear elasticity, z-invariant straight fibers; unit square periodic cell",
                           "timing": "single-run CPU wall time; not a production/GPU benchmark"},
              "validation": validate(), "cases": []}
    print("validation", result["validation"], flush=True)
    c0 = stiffness()
    geometries = {"ellipse": [Ellipse()],
                  "two_fibers": [Ellipse(0.18, 0.22, -0.21, 0), Ellipse(0.18, 0.22, 0.21, 0)],
                  "tight_gap": [Ellipse(0.22, 0.22, -0.225, 0), Ellipse(0.22, 0.22, 0.225, 0)]}
    selected = args.cases or (["ellipse"] if args.quick else ["ellipse", "two_fibers"])
    cases = [(name, geometries[name]) for name in selected]
    for name, ellipses in cases:
        case = {"name": name, "ellipses": [asdict(e) for e in ellipses],
                "exact_fraction": sum(e.fraction for e in ellipses),
                "matrix_E": 1.0, "matrix_nu": 0.3, "fiber_nu": 0.3,
                "modal": [], "voxel": []}
        result["cases"].append(case)
        for cutoff, h, build_seconds in interaction_matrices(ellipses, max(degrees), cutoffs, c0):
            for contrast in [10.0, 100.0]:
                for degree in degrees:
                    c, info = effective_modal(h, ellipses, max(degrees), degree, c0, contrast)
                    case["modal"].append({"contrast": contrast, "degree": degree, "cutoff": cutoff,
                                          "kernel_build_cumulative_seconds": build_seconds,
                                          "kernel_bytes": h.nbytes, "C_mandel": c.tolist(),
                                          "metrics": metrics(c), **info})
            print(f"{name}: Fourier cutoff {cutoff}, build {build_seconds:.2f}s", flush=True)
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        for contrast in [10.0, 100.0]:
            for grid in grids:
                c, info = voxel_reference(grid, ellipses, c0, contrast)
                case["voxel"].append({"contrast": contrast, "C_mandel": c.tolist(),
                                      "metrics": metrics(c), **info})
                print(f"{name}: voxel {grid}, contrast {contrast:g}, {info['total_seconds']:.2f}s, "
                      f"iterations {info['cg_iterations']}", flush=True)
                (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
