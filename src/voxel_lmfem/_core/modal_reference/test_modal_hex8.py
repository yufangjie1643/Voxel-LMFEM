"""Mechanics and assembly checks against independently assembled conventional FE."""
from itertools import product
import ast

import numpy as np
import pytest
from numba import set_num_threads

from modal_hex8 import (PeriodicHex8Operator, b_matrices, independent_csr,
                       pack, weave_material, SIGNS, FEAD, materials)
import generated_kernels as kernels


@pytest.fixture(scope="module")
def random_problem():
    set_num_threads(2)
    shape = (3, 4, 5)
    rng = np.random.default_rng(1729)
    a = rng.normal(size=(np.prod(shape), 6, 6))
    c = pack(a @ a.swapaxes(1, 2)+np.eye(6))
    return c, shape, (0.7, 1.3, 0.9), rng.normal(size=(np.prod(shape), 3))


@pytest.mark.parametrize("kind,arithmetic,assembly", product(
    ("hg", "fi"), ("reference", "modal"), ("twostage", "tiled")))
def test_periodic_operator_rhs_diagonal_and_symmetry(random_problem, kind, arithmetic, assembly):
    c, shape, sizes, u = random_problem
    op = PeriodicHex8Operator(c, shape, sizes, kind, arithmetic, assembly, tile=2)
    matrix, reference_rhs, _ = independent_csr(c, shape, sizes, kind, op.scales)
    result = op.apply(u)
    np.testing.assert_allclose(result.ravel(), matrix@u.ravel(), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(op.apply(np.ones_like(u)), 0, atol=1e-13)
    v = np.random.default_rng(19).normal(size=u.shape)
    assert abs(np.vdot(v, result)-np.vdot(u, op.apply(v))) < 1e-12*np.linalg.norm(u)*np.linalg.norm(result)
    for load in range(6):
        rhs = op.rhs(np.eye(6)[load])
        np.testing.assert_allclose(rhs.ravel(), reference_rhs[:, load], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(op.diagonal.ravel(), matrix.diagonal(), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.sum(axis=0), 0, atol=1e-11)


@pytest.mark.parametrize("kind", ["hg", "fi"])
def test_local_rigid_modes_and_random_materials(kind):
    rng = np.random.default_rng(83)
    for _ in range(15):
        sizes = np.exp(rng.uniform(-2, 2, 3))
        factor = rng.normal(size=(6, 6))
        material = factor.T@factor+0.3*np.eye(6)
        c = pack(material)
        g = np.array([1/(4*h) for h in sizes]+[np.prod(sizes)])
        b, weight = b_matrices(sizes, kind)
        projector = np.zeros((24, 24))
        for m in (3, 5, 6, 7):
            scalar = np.prod(SIGNS[:, [d for d in range(3) if m & (1 << d)]], axis=1)
            projector += np.kron(np.outer(scalar, scalar)/8, np.eye(3))
        reference = sum(weight*bg.T@material@bg for bg in b)
        if kind == "hg":
            reference += 0.37*projector
        local = getattr(kernels, kind+"_modal_local")
        u = rng.normal(size=24)
        np.testing.assert_allclose(local(u, c, 0.37, g), reference@u, rtol=5e-13, atol=5e-13)
        coordinates = SIGNS*sizes/2
        for axis in np.eye(3):
            rotation = np.cross(np.tile(axis, (8, 1)), coordinates).ravel()
            assert np.linalg.norm(local(rotation, c, 0.37, g)) < 1e-12*np.linalg.norm(reference)*np.linalg.norm(rotation)


def test_tiled_scratch_and_invalid_inputs():
    c, sizes, _ = weave_material(4)
    op = PeriodicHex8Operator(c, (4, 4, 4), sizes, "hg", assembly="tiled", tile=3)
    assert op.forces is None
    assert op.storage()["global_force_buffer_bytes"] == 0
    u = np.ones((64, 3))
    with pytest.raises(ValueError):
        op.apply(u, out=u)
    with pytest.raises(ValueError):
        PeriodicHex8Operator(c, (4, 4, 4), sizes, kind="invalid")


@pytest.mark.parametrize("n", [8, 16, 32])
def test_woven_inputs_match_frozen_resolution_geometry(n):
    source = FEAD/"repro/extension_20261006/run_resolution.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "geometry")
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    data, _ = scope["geometry"](n)
    ids = data["yarn_id"].reshape(-1)
    c, _, meta = weave_material(n)
    import hashlib
    assert meta["phase_sha256"] == hashlib.sha256(data["yarn_id"][0].tobytes()).hexdigest()
    matrix = materials.isotropic_to_voigt(3500.0, 0.35)
    yarn = materials.engineering_to_voigt(135000.0, 9000.0, 9000.0, 0.30, 0.30, 0.35, 5000.0, 5000.0, 3500.0)
    expected = np.tile(pack(matrix), (n**3, 1))
    mask = ids >= 0
    r = materials.build_rotation_matrices(data["orient1"].reshape(-1, 3)[mask],
                                         data["orient2"].reshape(-1, 3)[mask])
    expected[mask] = pack(materials.rotate_voigt_batch(yarn, r))
    np.testing.assert_allclose(c, expected, rtol=2e-13, atol=1e-9)
