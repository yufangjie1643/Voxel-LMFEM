"""Periodic interlaced yarns with a controlled, non-overlapping volume fraction.

Uses the manuscript's sinusoidal centre lines and vertical elliptical sections.
Array order is (z,y,x); orientations follow the local centre-line tangent.
"""
import hashlib
import json
import numpy as np

MATERIALS = {
    0: dict(kind='isotropic', E=3500., nu=.35),
    1: dict(kind='orthotropic', frame='local', E1=135000., E2=9000., E3=9000.,
            nu12=.30, nu13=.30, nu23=.35, G12=5000., G13=5000., G23=3500.),
}
FRACTIONS = [round(.05 + .014*i, 6) for i in range(21)]


def geometry(n, fraction, *, full_orientation=True):
    width, amplitude, height = .18, .14, .4
    thickness = float(fraction)*height/(4*np.pi*width)
    crossing_gap = 2*(amplitude*np.cos(2*np.pi*width)-thickness)
    periodic_gap = height-2*(amplitude+thickness)
    assert 0 < width < .25 and crossing_gap > 0 and periodic_gap > 0
    xy = (np.arange(n)+.5)/n
    X, Y = np.meshgrid(xy, xy, indexing='xy')
    ids = np.full((n,n,n), -1, dtype=np.int8)
    o1 = np.zeros((n,n,n,3), dtype=np.float64) if full_orientation else None
    o2 = np.zeros_like(o1) if full_orientation else None
    for k, z in enumerate((xy-.5)*height):
        for i, (axis, center, sign) in enumerate([(0,0.,1), (0,.5,-1), (1,0.,-1), (1,.5,1)]):
            s, t = (X,Y) if axis == 0 else (Y,X)
            distance = (t-center+.5)%1-.5
            zc = sign*amplitude*np.cos(2*np.pi*s)
            dz = (z-zc+height/2)%height-height/2
            mask = (distance/width)**2+(dz/thickness)**2 < 1
            if np.any(mask & (ids[k] >= 0)):
                raise ValueError('Intersecting yarns')
            ids[k][mask] = i
            if full_orientation:
                slope = -sign*amplitude*2*np.pi*np.sin(2*np.pi*s[mask])
                tangent = np.zeros((len(slope),3))
                tangent[:,axis] = 1/np.sqrt(1+slope*slope)
                tangent[:,2] = slope/np.sqrt(1+slope*slope)
                transverse = np.zeros_like(tangent)
                transverse[:,1-axis] = 1 if axis == 0 else -1
                o1[k][mask], o2[k][mask] = tangent, transverse
    phase = (ids >= 0).astype(np.uint8)[None]
    data = dict(grid_shape=(n,n,n), domain_size=np.array([[1.,1.,height]]), phase_id=phase)
    if full_orientation:
        data['orientation'] = dict(axis1=o1[None], axis2=o2[None])
    digest = hashlib.sha256(ids.tobytes())
    if full_orientation:
        digest.update(o1.tobytes()); digest.update(o2.tobytes())
    digest.update(json.dumps(MATERIALS, sort_keys=True).encode())
    meta = dict(width=width, thickness=thickness, amplitude=amplitude, height=height,
                exact_yarn_fraction=float(fraction), voxel_yarn_fraction=float(phase.mean()),
                crossing_gap_lower_bound=crossing_gap, periodic_gap_lower_bound=periodic_gap,
                overlap_voxels=0, yarn_voxel_counts=[int(np.sum(ids==i)) for i in range(4)],
                construction='periodic sinusoidal centre lines; vertical elliptical sections; local tangent orientation')
    return data, ids, meta, digest.hexdigest()
