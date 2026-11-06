import json

import numpy as np
import pytest

from voxel_lmfem import homogenize, import_model
from voxel_lmfem.inputs import load_input
from test_cli import invoke


def materials_file(tmp_path, table):
    path = tmp_path / "materials.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def test_npy_cli_import_and_solve_outside_checkout(tmp_path):
    source = tmp_path / "external.npy"
    np.save(source, np.full((2, 3, 4), 17, dtype=">u2"))
    materials = materials_file(tmp_path, {17: dict(kind="isotropic", E=3500., nu=.35)})
    target = tmp_path / "converted"
    result = invoke("import-model", "--input", source, "--materials", materials, "--axes", "xyz",
                    "--spacing", .5, 2, 3, "--out", target, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    phase, _, domain, _ = load_input(target / "model.npz", target / "materials.json")
    assert phase.shape == (4, 3, 2) and phase.dtype.isnative
    np.testing.assert_array_equal(domain, [1., 6., 12.])
    report = json.loads((target / "import.json").read_text())
    assert report["phases"] == [dict(id=17, voxels=24, fraction=1.)]
    assert len(report["sources"]["volume"]["sha256"]) == 64
    solved = invoke("solve", "--input", target / "model.npz", "--materials", target / "materials.json",
                    "--backend", "cpu", "--output", tmp_path / "result.json", cwd=tmp_path)
    assert solved.returncode == 0, solved.stderr
    stiffness = json.loads((tmp_path / "result.json").read_text())["CH"]
    assert stiffness[0][0] == pytest.approx(3500 * .65 / (1.35 * .30))
    before = (target / "model.npz").read_bytes()
    with pytest.raises(FileExistsError):
        import_model(source, materials, target, spacing=[1, 1, 1])
    assert (target / "model.npz").read_bytes() == before


def test_spatial_labels_and_directions_follow_axis_permutation(tmp_path):
    phase = np.zeros((2, 3, 4), dtype=np.uint8)
    phase[1, 2, 3] = 1
    rotation = np.broadcast_to(np.eye(3), (2, 3, 4, 3, 3)).copy()
    turned = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotation[1, 2, 3] = turned
    source = tmp_path / "external.npz"
    np.savez(source, labels=phase, orientation=rotation, domain_size=[2, 3, 4])
    materials = materials_file(tmp_path, {0: dict(kind="isotropic", E=1., nu=.2),
                                         1: dict(kind="anisotropic", C=np.diag([10, 20, 30, 4, 5, 6]).tolist(), frame="local")})
    target = tmp_path / "converted"
    import_model(source, materials, target, axes="xyz", key="labels")
    actual, _, domain, directions = load_input(target / "model.npz", target / "materials.json")
    assert actual.shape == (4, 3, 2) and actual[3, 2, 1] == 1 and actual.sum() == 1
    np.testing.assert_array_equal(directions[3, 2, 1], turned)
    np.testing.assert_array_equal(domain, [2, 3, 4])


def test_external_direction_axes_and_rotated_anisotropic_solve(tmp_path):
    source, directions = tmp_path / "phase.npy", tmp_path / "axes.npz"
    np.save(source, np.zeros((2, 2, 2), dtype=np.uint8))
    np.savez(directions, axis1=[0, 1, 0], axis2=[-1, 0, 0])
    materials = materials_file(tmp_path, {0: dict(kind="anisotropic", frame="local",
                                                C=np.diag([10., 20., 30., 4., 5., 6.]).tolist())})
    target = tmp_path / "converted"
    import_model(source, materials, target, domain_size=[1, 2, 3], orientation_path=directions)
    phase, table, domain, orientation = load_input(target / "model.npz", target / "materials.json")
    result = homogenize(phase, table, domain, orientation=orientation, backend="cpu")
    np.testing.assert_allclose(result["CH"], np.diag([20., 10., 30., 5., 4., 6.]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case,message", [
    ("float", "integer label"), ("missing-size", "spacing"), ("unknown-phase", "Undefined material"),
    ("bad-size", "positive"), ("missing-direction", "require orientation"),
    ("bad-frame", "orthonormal"), ("ambiguous-direction", "rather than both"),
])
def test_rejected_models_leave_no_partial_output(tmp_path, case, message):
    source = tmp_path / "external.npz"
    phase = np.zeros((2, 2, 2), dtype=float if case == "float" else np.uint8)
    if case == "unknown-phase":
        phase[:] = 2
    arrays = dict(phase_id=phase)
    if case != "missing-size":
        arrays["domain_size"] = [1, 0 if case == "bad-size" else 1, 1]
    material = dict(kind="isotropic", E=1., nu=.2)
    if case in ("missing-direction", "bad-frame", "ambiguous-direction"):
        material = dict(kind="anisotropic", C=np.eye(6).tolist(), frame="local")
    if case == "bad-frame":
        arrays["orientation"] = np.ones((3, 3))
    if case == "ambiguous-direction":
        arrays.update(orientation=np.eye(3), axis1=[1, 0, 0], axis2=[0, 1, 0])
    np.savez(source, **arrays)
    materials = materials_file(tmp_path, {0: material})
    with pytest.raises(ValueError, match=message):
        import_model(source, materials, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()
