import numpy as np
import pytest

from voxel_lmfem import homogenize


def isotropic(E, nu):
    mu = E/(2*(1+nu))
    lam = E*nu/((1+nu)*(1-2*nu))
    C = np.zeros((6,6))
    C[:3,:3] = lam
    C[:3,:3] += 2*mu*np.eye(3)
    C[3:,3:] = mu*np.eye(3)
    return C


def test_homogeneous_unequal_voxels_and_nonzero_phase_label():
    result = homogenize(np.full((2,3,4), 17, dtype=np.int32),
        {17: dict(kind="isotropic", E=3500., nu=.35)}, [2.,3.,5.], backend="cpu")
    np.testing.assert_allclose(result["CH"], isotropic(3500., .35), rtol=2e-13, atol=2e-10)
    assert result["max_true_residual"] < 1e-10


def test_laminate_against_traction_continuity():
    # Independent Schur-complement laminate formula, normal to x3.
    phase = np.zeros((4,2,3), dtype=np.uint8)
    phase[:2] = 1
    specs = {0: dict(kind="isotropic", E=3500., nu=.35),
             1: dict(kind="isotropic", E=70000., nu=.2)}
    matrices = [isotropic(p["E"], p["nu"]) for p in specs.values()]
    a, b = [0,1,5], [2,3,4]
    A, B, D = [np.zeros((3,3)) for _ in range(3)]
    for C in matrices:
        inv = np.linalg.inv(C[np.ix_(b,b)])
        ba = C[np.ix_(b,a)]
        A += .5*inv
        B += .5*inv@ba
        D += .5*(C[np.ix_(a,a)]-C[np.ix_(a,b)]@inv@ba)
    H = np.linalg.inv(A)
    expected = np.zeros((6,6))
    expected[np.ix_(b,b)] = H
    expected[np.ix_(b,a)] = H@B
    expected[np.ix_(a,b)] = B.T@H
    expected[np.ix_(a,a)] = D+B.T@H@B
    result = homogenize(phase, specs, [1.,1.,1.], backend="cpu")
    np.testing.assert_allclose(result["CH"], expected, rtol=2e-12, atol=2e-9)


def test_rotated_anisotropic_homogeneous_cell():
    C = np.diag([10.,20.,30.,4.,5.,6.])
    R = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    result = homogenize(np.zeros((2,2,2),dtype=np.uint8),
        {0: dict(kind="anisotropic", C=C, frame="local")}, [1.,2.,3.],
        orientation=R, backend="cpu")
    np.testing.assert_allclose(result["CH"], np.diag([20.,10.,30.,5.,4.,6.]),
                               rtol=2e-13, atol=1e-12)


@pytest.mark.parametrize("phase,materials,domain,orientation,message", [
    (np.zeros((2,2,2)), {0: dict(kind="isotropic",E=1.,nu=.2)}, [1,1,1], None, "integer"),
    (np.ones((2,2,2),dtype=int), {0: dict(kind="isotropic",E=1.,nu=.2)}, [1,1,1], None, "Undefined"),
    (np.zeros((2,2,2),dtype=int), {0: dict(kind="isotropic",E=1.,nu=.2)}, [1,0,1], None, "positive"),
    (np.zeros((2,2,2),dtype=int), {0: dict(kind="anisotropic",C=-np.eye(6))}, [1,1,1], None, "positive definite"),
    (np.zeros((2,2,2),dtype=int), {0: dict(kind="anisotropic",C=np.eye(6),frame="local")},
     [1,1,1], None, "require orientation"),
    (np.zeros((2,2,2),dtype=int), {0: dict(kind="anisotropic",C=np.eye(6),frame="local")},
     [1,1,1], np.diag([1,1,-1]), "right-handed"),
])
def test_invalid_inputs_fail_before_solving(phase,materials,domain,orientation,message):
    with pytest.raises(ValueError, match=message):
        homogenize(phase, materials, domain, orientation=orientation, backend="cpu")


def test_cpu_size_guard():
    with pytest.raises(ValueError, match="4096"):
        homogenize(np.zeros((17,17,17),dtype=np.uint8),
                   {0: dict(kind="isotropic",E=1.,nu=.2)}, [1,1,1], backend="cpu")
