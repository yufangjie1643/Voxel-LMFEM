"""Independent CPU checks for generic material fields and small FI solutions.

Usage: python validate.py --output results/cpu.json [--gpu-fields DIR]
CPU mode never invokes CUDA. GPU field exports are checked by independently
assembled eight-point periodic Q1 matrices and sparse direct solves.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "repro/source"
if SOURCE.exists():
    sys.path.insert(0, str(SOURCE))
else:
    sys.path.insert(0, str(__import__('project_paths').source_path('runtime/repro/source')))
from adapter import load_material_module
mf = load_material_module()


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def fourth_order_rotate(C, R):
    """Independent C_ijkl rotation, not the production Bond implementation."""
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    tensor = np.empty((3, 3, 3, 3))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            for ii, jj in ((i, j), (j, i)):
                for kk, ll in ((k, l), (l, k)):
                    tensor[ii, jj, kk, ll] = C[a, b]
    rotated = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, tensor)
    return np.array([[rotated[i, j, k, l] for k, l in pairs] for i, j in pairs])


def fixture(kind, grid=6):
    z, y, x = np.indices((grid, grid, grid))
    labels = np.array([10, 40, 70])[(x + 2*y + z*z + x*y) % 3]
    table = {
        10: dict(kind="isotropic", E=3500., nu=.35),
        40: dict(kind="isotropic", E=70000., nu=.2),
        70: dict(kind="isotropic", E=12000., nu=.28),
    }
    orientation = None
    if kind == "mixed":
        table[70] = dict(kind="orthotropic", frame="local", E1=135000., E2=9000., E3=9000.,
                         nu12=.30, nu13=.30, nu23=.35, G12=5000., G13=5000., G23=3500.)
        angle = .2*x + .3*y - .15*z
        # Proper rotations Rz(angle) Ry(0.37), non-axis-aligned and spatially varying.
        ca, sa, cb, sb = np.cos(angle), np.sin(angle), np.cos(.37), np.sin(.37)
        R = np.zeros(labels.shape + (3, 3))
        R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = ca*cb, -sa, ca*sb
        R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = sa*cb, ca, sa*sb
        R[..., 2, 0], R[..., 2, 2] = -sb, cb
        orientation = R
    full = np.stack([mf.material_stiffness(table[int(k)]) for k in labels.ravel()])
    if orientation is not None:
        for e in np.flatnonzero(labels.ravel() == 70):
            full[e] = fourth_order_rotate(full[e], orientation.reshape(-1, 3, 3)[e])
    h = np.array([.3, .5, .7])
    data = dict(grid_shape=labels.shape, domain_size=(h*np.array(labels.shape[::-1]))[None],
                phase_id=labels[None])
    if orientation is not None:
        data["orientation"] = orientation[None]
    return data, table, full, h


def assemble(full, shape, h):
    """Independent eight-point Q1 stiffness/RHS/recovery; no solver B arrays."""
    nz, ny, nx = shape
    n = int(np.prod(shape))
    offsets = np.array([(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)])
    signs = 2 * offsets - 1
    Bs = []
    for point in itertools.product((-1/np.sqrt(3), 1/np.sqrt(3)), repeat=3):
        point = np.array(point)
        d = np.empty((8, 3))
        for j in range(3):
            others = [k for k in range(3) if k != j]
            d[:, j] = signs[:, j] * np.prod(1 + signs[:, others]*point[others], axis=1)/(4*h[j])
        B = np.zeros((6, 24))
        for a, (dx, dy, dz) in enumerate(d):
            B[0, 3*a], B[1, 3*a+1], B[2, 3*a+2] = dx, dy, dz
            B[3, 3*a+1], B[3, 3*a+2] = dz, dy
            B[4, 3*a], B[4, 3*a+2] = dz, dx
            B[5, 3*a], B[5, 3*a+1] = dy, dx
        Bs.append(B)
    Bs = np.array(Bs)
    Ks = np.einsum("gai,eab,gbj->eij", Bs, full, Bs)*np.prod(h)/8
    Fs = -np.einsum("gai,eab->eib", Bs, full)*np.prod(h)/8
    z, y, x = np.indices(shape)
    origins = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    coords = (origins[:, None] + offsets) % [nx, ny, nz]
    nodes = (coords[..., 2]*ny + coords[..., 1])*nx + coords[..., 0]
    dofs = (3*nodes[..., None] + np.arange(3)).reshape(n, 24)
    rows = np.broadcast_to(dofs[:, :, None], (n, 24, 24)).ravel()
    cols = np.broadcast_to(dofs[:, None, :], (n, 24, 24)).ravel()
    K = sparse.coo_matrix((Ks.ravel(), (rows, cols)), shape=(3*n, 3*n)).tocsr()
    rhs = np.zeros((3*n, 6))
    np.add.at(rhs, dofs, Fs)
    u = np.zeros_like(rhs)
    u[3:] = splu(K[3:, 3:].tocsc()).solve(rhs[3:])
    u = u.reshape(n, 3, 6)
    u -= u.mean(axis=0, keepdims=True)

    def recover(v, j):
        local = v.reshape(n, 3)[nodes].reshape(n, 24)
        strain = np.einsum("gij,ej->egi", Bs, local) + np.eye(6)[j]
        return np.einsum("eij,egj->egi", full, strain).mean(axis=(0, 1))

    CH = np.column_stack([recover(u[..., j], j) for j in range(6)])
    return K, rhs, u, CH, recover


def cpu_checks():
    import torch
    torch.set_num_threads(1)
    rows = []
    for kind in ("isotropic", "mixed"):
        data, table, expected, h = fixture(kind)
        start = time.perf_counter()
        actual = mf.build_material_field(data["phase_id"], table,
                    orientation=data.get("orientation"), device="cpu", chunk_size=17).numpy()[0]
        elapsed = time.perf_counter()-start
        target = expected[:, [i for i, j in mf.PAIRS], [j for i, j in mf.PAIRS]]
        err = relative(actual, target)
        assert err < 1e-14
        # Full batch, 3D input, different chunk boundaries and two-axis representation.
        orient = data.get("orientation")
        if orient is not None:
            orient = {"axis1": orient[..., :, 0]*3, "axis2": orient[..., :, 1]*2 + orient[..., :, 0]}
        batched = mf.build_material_field(np.repeat(data["phase_id"], 2, axis=0), table,
                      orientation=orient, device="cpu", chunk_size=31).numpy()
        assert relative(batched[0], target) < 1e-14 and relative(batched[1], target) < 1e-14
        K, rhs, u, CH, _ = assemble(expected, data["grid_shape"], h)
        res = max(np.linalg.norm(rhs[:, j]-K@u[..., j].ravel())/np.linalg.norm(rhs[:, j]) for j in range(6))
        assert res < 1e-12
        rows.append(dict(case=kind, grid=list(data["grid_shape"]), C21_relative=err,
                         construction_cpu_seconds=elapsed, direct_residual=float(res), CPU_CH=CH.tolist()))
    # A general SPD tensor, global frame without orientations, and fixed local rotation.
    rng = np.random.default_rng(20261021)
    X = rng.normal(size=(6, 6)); C = X@X.T + np.eye(6)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0: R[:, 0] *= -1
    label = np.full((2, 3, 4), -8, dtype=int)
    packed = lambda c: c[tuple(i for i, j in mf.PAIRS), tuple(j for i, j in mf.PAIRS)]
    global_C = mf.build_material_field(label, {-8: dict(kind="anisotropic", C=C)}, device="cpu").numpy()
    assert relative(global_C, np.broadcast_to(packed(C), global_C.shape)) < 1e-14
    local_C = mf.build_material_field(label, {-8: dict(kind="anisotropic", C=C, frame="local")},
                                     orientation=R, device="cpu", chunk_size=5).numpy()
    assert relative(local_C, np.broadcast_to(packed(fourth_order_rotate(C, R)), local_C.shape)) < 1e-14
    iso = {0: dict(kind="isotropic", E=5., nu=.2)}
    # Isotropic data do not inspect invalid orientation placeholders.
    mf.build_material_field(np.zeros((2, 2, 2), int), iso, orientation=np.nan, device="cpu")
    invalid = [
        (np.ones((2, 2, 2), int), iso, {}),
        (label, {-8: dict(kind="anisotropic", C=C, frame="local")}, {}),
        (label, {-8: dict(kind="anisotropic", C=C, frame="local")}, {"orientation": np.ones((3, 3))}),
        (label, {-8: dict(kind="anisotropic", C=C, frame="local")}, {"orientation": {"axis1": [1, 0, 0], "axis2": [2, 0, 0]}}),
        (label, {-8: dict(kind="anisotropic", C=-C)}, {}),
        (label.astype(float), {-8: dict(kind="anisotropic", C=C)}, {}),
        (label, {-8: dict(kind="isotropic", E=1., nu=.5)}, {}),
    ]
    for labels, table, kw in invalid:
        try:
            mf.build_material_field(labels, table, device="cpu", **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    # Legacy labels remain confined to the optional adapter. No axes are needed
    # for an all-isotropic legacy input, even when its yarn numbers are arbitrary.
    legacy = dict(yarn_id=np.array([[[-1, 5], [8, -1]], [[5, -1], [-1, 8]]]))
    mapped, table, orientation = mf.from_legacy_yarn(legacy,
        dict(matrix=dict(E=3., Nu=.25), yarn=dict(E=7., Nu=.3)))
    assert orientation is None and np.array_equal(mapped, legacy["yarn_id"] >= 0)
    mf.build_material_field(mapped, table, device="cpu")
    chain = HERE.parents[1]/"experiments/chain_optimization_20261021"
    if not chain.exists():
        chain = Path(str(__import__('project_paths').source_path('solver_chain')))
    sys.path.insert(0, str(chain))
    from reference_closed_form import design_closed, isotropic_stiffness
    unordered = [isotropic_stiffness(10., 1.), isotropic_stiffness(1., 10.),
                 isotropic_stiffness(4., 3.)]
    baseline = isotropic_stiffness(3., 3.)
    ref, meta = design_closed(unordered, baseline, "fi")
    assert meta["phase_count"] == 3 and meta["status"] == "unsupported"
    assert np.array_equal(ref, baseline)
    return dict(passed=True, cases=rows, invalid_inputs_rejected=len(invalid),
                general_anisotropy=True, no_yarn_fields=True, batched=True, seed=20261021,
                legacy_adapter=True, unordered_three_phase_reference_fallback=True)


def gpu_checks(directory):
    rows = []
    for kind in ("isotropic", "mixed"):
        data, table, full, h = fixture(kind)
        fields = np.load(directory / (kind + "_fields.npz"))
        K, rhs, u, CH, recover = assemble(full, data["grid_shape"], h)
        loads = []
        for j in range(6):
            v = fields["U"][j]
            res = float(np.linalg.norm(rhs[:, j]-K@v.ravel())/np.linalg.norm(rhs[:, j]))
            rhs_err = relative(fields["RHS"][j].ravel(), rhs[:, j])
            stress_err = relative(fields["CH"][:, j], recover(v, j))
            assert res <= 1.01e-10 and rhs_err < 1e-12 and stress_err < 1e-12
            loads.append(dict(load=j, independent_residual=res, rhs_relative=rhs_err,
                              recovery_relative=stress_err, displacement_relative=relative(v, u[..., j])))
        err = relative(fields["CH"], CH)
        assert err < 1e-9
        rows.append(dict(case=kind, CH_relative=err, loads=loads))
    return dict(passed=True, cases=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-fields", type=Path)
    args = parser.parse_args()
    report = gpu_checks(args.gpu_fields) if args.gpu_fields else cpu_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
