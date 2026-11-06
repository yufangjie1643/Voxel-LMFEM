"""Source-traced constituent law; engineering Voigt order 11,22,33,23,13,12."""
import json
from pathlib import Path
import numpy as np

PAIRS = tuple((i, j) for i in range(6) for j in range(i, 6))


def load_material():
    return json.loads(Path(__file__).with_name("material.json").read_text(encoding="utf8"))


def orthotropic(p):
    S = np.zeros((6, 6), dtype=np.float64)
    S[0, 0], S[1, 1], S[2, 2] = 1/p["E11"], 1/p["E22"], 1/p["E33"]
    S[0, 1] = S[1, 0] = -p["Nu12"]/p["E11"]
    S[0, 2] = S[2, 0] = -p["Nu13"]/p["E11"]
    S[1, 2] = S[2, 1] = -p["Nu23"]/p["E22"]
    S[3, 3], S[4, 4], S[5, 5] = 1/p["G23"], 1/p["G13"], 1/p["G12"]
    return np.linalg.inv(S)


def stiffness_states(material):
    """[phase, failed, i, j], phase=0 matrix, phase=1 fiber."""
    m, f = material["matrix"], material["fiber"]
    g = m["E"]/(2*(1+m["Nu"]))
    matrix = orthotropic(dict(E11=m["E"], E22=m["E"], E33=m["E"],
                              Nu12=m["Nu"], Nu13=m["Nu"], Nu23=m["Nu"],
                              G12=g, G13=g, G23=g))
    fd = dict(f)
    for key in ("E11", "Nu12", "Nu13", "G12", "G13"):
        fd[key] *= material["retained_factors"]["fiber"]
    states = np.array([[matrix, material["retained_factors"]["matrix"]*matrix],
                       [orthotropic(f), orthotropic(fd)]])
    if np.linalg.eigvalsh(states).min() <= 0:
        raise ValueError("Constituent stiffness must be positive definite")
    return states


def indices(stress, phase, material):
    """Source Table 10: maximum axial fiber stress; pressure-sensitive matrix."""
    s, p = np.asarray(stress), material["strength"]
    s11, s22, s33 = s[..., 0], s[..., 1], s[..., 2]
    vm2 = .5*((s11-s22)**2+(s22-s33)**2+(s11-s33)**2) + 3*(s[..., 3:]**2).sum(-1)
    matrix = vm2/(p["matrix_tension"]*p["matrix_compression"])
    matrix += (1/p["matrix_tension"]-1/p["matrix_compression"])*(s11+s22+s33)
    fiber = np.maximum(s11/p["fiber_tension"], -s11/p["fiber_compression"])
    return np.where(phase == 1, fiber, matrix)


def hexagonal_geometry(n, vf):
    """Two circular fibers per rectangular hexagonal lattice cell, parallel to x1.

    Domain is a nondimensional periodic cell. Without an energy/length-dependent
    damage law, its absolute size does not affect these quasistatic responses.
    """
    ny, nz, nx = n, int(round(np.sqrt(3)*n)), 4
    y = (np.arange(ny)+.5)/ny
    z = (np.arange(nz)+.5)*np.sqrt(3)/nz
    zz, yy = np.meshgrid(z, y, indexing="ij")
    dist2 = np.full_like(zz, np.inf)
    for cy, cz in [(0., 0.), (.5, np.sqrt(3)/2)]:
        dy = (yy-cy+.5) % 1-.5
        dz = (zz-cz+np.sqrt(3)/2) % np.sqrt(3)-np.sqrt(3)/2
        dist2 = np.minimum(dist2, dy*dy+dz*dz)
    radius = np.sqrt(vf*np.sqrt(3)/(2*np.pi))
    phase = np.broadcast_to((dist2 < radius**2)[..., None], (nz, ny, nx)).astype(np.int64).copy()
    domain = np.array([1., 1., np.sqrt(3)])
    return phase, domain, dict(lattice="hexagonal", fiber_axis=0,
                               fiber_radius=radius, requested_fiber_fraction=vf,
                               voxel_fiber_fraction=float(phase.mean()))
