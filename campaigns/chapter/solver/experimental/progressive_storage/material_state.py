"""CPU reference prototype: immutable material catalog + transactional history.

This is not a CUDA backend or a general constitutive evolution law. Discrete
states must be ordered irreversible stiffness states supplied by the caller.
Scalar mode implements exactly C(d)=(1-d)C0, with 0 <= d < 1.
Engineering Voigt order: 11,22,33,23,13,12. Histories live at elements or 8 QPs.
"""
from contextlib import contextmanager
import numpy as np

PAIRS = tuple((i, j) for i in range(6) for j in range(i, 6))


def pack(C):
    return np.stack([C[..., i, j] for i, j in PAIRS], axis=-1)


def unpack(c):
    C = np.zeros(c.shape[:-1] + (6, 6))
    for k, (i, j) in enumerate(PAIRS):
        C[..., i, j] = C[..., j, i] = c[..., k]
    return C


def packed_action(c, x):
    """Constitutive action without constructing a 6x6 material tensor."""
    y = np.zeros_like(x, dtype=np.float64)
    for k, (i, j) in enumerate(PAIRS):
        y[i] += c[k] * x[j]
        if i != j:
            y[j] += c[k] * x[i]
    return y


class SharedStateMaterial:
    def __init__(self, phase, catalog, *, mode="discrete", points=1):
        phase = np.asarray(phase)
        C = np.asarray(catalog, dtype=np.float64)
        if mode not in ("discrete", "scalar") or points not in (1, 8):
            raise ValueError("Supported modes: discrete/scalar; points: 1/8")
        if C.ndim != 4 or C.shape[-2:] != (6, 6) or min(C.shape[:2]) < 1:
            raise ValueError("catalog must have shape (P,S,6,6)")
        if phase.ndim != 1 or not phase.size or phase.dtype.kind not in "iu":
            raise ValueError("phase must be a nonempty integer row map")
        if phase.min() < 0 or phase.max() >= len(C):
            raise ValueError("Undefined material row")
        if not np.isfinite(C).all() or not np.allclose(C, C.swapaxes(-1, -2), rtol=1e-13, atol=0):
            raise ValueError("Symmetric finite constitutive tensors required")
        if np.linalg.eigvalsh(C).min() <= 0:
            raise ValueError("This frozen-damage PCG prototype requires SPD tensors")
        if mode == "scalar" and C.shape[1] != 1:
            raise ValueError("Scalar mode takes one virgin tensor per row")
        if C.shape[1] > 256:
            raise ValueError("Prototype discrete history supports at most 256 states")
        if mode == "discrete" and C.shape[1] > 1:
            loss = C[:, :-1] - C[:, 1:]
            if np.linalg.eigvalsh(loss).min() < -1e-11 * np.linalg.norm(C):
                raise ValueError("Ordered states must have nonnegative stiffness loss")
        self.mode, self.points = mode, points
        self.phase = phase.astype(np.int32, copy=True)
        self.catalog = pack(C).copy()
        self.phase.flags.writeable = self.catalog.flags.writeable = False
        self._committed = np.zeros((len(phase), points), dtype=np.uint8 if mode == "discrete" else np.float64)
        self._trial = None
        self._locks = 0
        self.version = 0

    @property
    def state(self):
        view = (self._committed if self._trial is None else self._trial).view()
        view.flags.writeable = False
        return view

    def _mutable(self):
        if self._locks:
            raise RuntimeError("Material is frozen during this linear solve")

    def begin_trial(self):
        self._mutable()
        if self._trial is not None:
            raise RuntimeError("Commit or roll back the previous trial first")
        self._trial = self._committed.copy()
        self.version += 1

    def update(self, elements, values):
        self._mutable()
        if self._trial is None:
            raise RuntimeError("begin_trial is required")
        ids, v = np.asarray(elements), np.asarray(values)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.unique(ids).size != ids.size:
            raise ValueError("Unique integer element indices required")
        if ids.size and (ids.min() < 0 or ids.max() >= len(self.phase)):
            raise ValueError("Invalid element index")
        if v.shape != (ids.size, self.points) or not np.isfinite(v).all():
            raise ValueError("Invalid state shape or nonfinite value")
        if self.mode == "discrete":
            if np.any(v != np.floor(v)) or np.any(v >= self.catalog.shape[1]):
                raise ValueError("Invalid discrete state")
        elif np.any(v >= 1):
            raise ValueError("Zero residual stiffness is outside this SPD prototype")
        if np.any(v < self._trial[ids]):
            raise ValueError("Irreversible history cannot decrease")
        changed = ids[np.any(v != self._trial[ids], axis=1)]
        self._trial[ids] = v
        if changed.size:
            self.version += 1
        return changed

    def commit(self, *, equilibrated):
        self._mutable()
        if self._trial is None or equilibrated is not True:
            raise RuntimeError("Only an equilibrated trial can be committed")
        self._committed = self._trial
        self._trial = None
        self.version += 1

    def rollback(self):
        self._mutable()
        if self._trial is None:
            raise RuntimeError("No trial to roll back")
        self._trial = None
        self.version += 1

    @contextmanager
    def frozen(self):
        self._locks += 1
        try:
            yield self
        finally:
            self._locks -= 1

    def action(self, element, qp, strain):
        s = self.state[element, 0 if self.points == 1 else qp]
        row = self.catalog[self.phase[element]]
        if self.mode == "discrete":
            return packed_action(row[int(s)], strain)
        return (1 - s) * packed_action(row[0], strain)

    def active_elements(self, background_phase):
        # A damaged background voxel must enter the correction kernel too.
        return np.flatnonzero((self.phase != background_phase) | np.any(self.state != 0, axis=1)).astype(np.int32)

    def storage_bytes(self):
        return dict(catalog=self.catalog.nbytes, phase=self.phase.nbytes,
                    committed=self._committed.nbytes,
                    trial=0 if self._trial is None else self._trial.nbytes)


def hex8_quadrature(h=(1., 1., 1.)):
    """Eight-point FI in a rectangular voxel; weights include det(J)."""
    from itertools import product
    corners = np.array(list(product((-1, 1), repeat=3)))
    Bs = []
    for q in np.array(list(product((-1/np.sqrt(3), 1/np.sqrt(3)), repeat=3))):
        grad = np.empty((8, 3))
        for j in range(3):
            others = [k for k in range(3) if k != j]
            grad[:, j] = corners[:, j] * np.prod(1 + corners[:, others] * q[others], axis=1) / (4*h[j])
        B = np.zeros((6, 24))
        for a, (gx, gy, gz) in enumerate(grad):
            B[:, 3*a:3*a+3] = [[gx,0,0],[0,gy,0],[0,0,gz],[0,gz,gy],[gz,0,gx],[gy,gx,0]]
        Bs.append(B)
    return np.array(Bs), np.full(8, np.prod(h)/8), ((corners+1)//2)


class ReferenceFI:
    """Small CPU matrix-free oracle for the proposed background + state split.

    Shared 24x24 background integration here is a validation convenience, not
    the GPU's 13-pair stencil or an implemented modal CUDA damage kernel.
    """
    def __init__(self, material, grid, h=(1., 1., 1.), background=0):
        self.material, self.background = material, background
        self.B, self.w, corners = hex8_quadrature(h)
        self.grid = tuple(grid)
        if np.prod(grid) != len(material.phase):
            raise ValueError("Grid does not match material map")
        self.dofs = []
        for cell in np.ndindex(*grid):
            nodes = np.ravel_multi_index(tuple(((np.array(cell)+corners) % grid).T), grid)
            self.dofs.append((3*nodes[:, None] + np.arange(3)).ravel())
        self.dofs = np.array(self.dofs)
        self.ndof = 3*len(self.dofs)
        self.Cb = material.catalog[background, 0]
        self.Kb = sum(w * B.T @ packed_action(self.Cb, B) for B, w in zip(self.B, self.w))

    def apply(self, u):
        out = np.zeros(self.ndof)
        for dofs in self.dofs:
            np.add.at(out, dofs, self.Kb @ u[dofs])
        for e in self.material.active_elements(self.background):
            dofs = self.dofs[e]
            f = np.zeros(24)
            for q, (B, w) in enumerate(zip(self.B, self.w)):
                strain = B @ u[dofs]
                delta = self.material.action(e, q, strain) - packed_action(self.Cb, strain)
                f += w * B.T @ delta
            np.add.at(out, dofs, f)
        return out

    def rhs(self, E):
        b = np.zeros(self.ndof)
        for e in self.material.active_elements(self.background):
            f = np.zeros(24)
            for q, (B, w) in enumerate(zip(self.B, self.w)):
                f -= w * B.T @ (self.material.action(e, q, E) - packed_action(self.Cb, E))
            np.add.at(b, self.dofs[e], f)
        return b

    def fields(self, u, E):
        stress = np.empty((len(self.dofs), 8, 6))
        energy = 0.
        for e, dofs in enumerate(self.dofs):
            for q, (B, w) in enumerate(zip(self.B, self.w)):
                strain = E + B @ u[dofs]
                stress[e, q] = self.material.action(e, q, strain)
                energy += .5*w*strain @ stress[e, q]
        volume = len(self.dofs)*self.w.sum()
        return np.einsum("eqi,q->i", stress, self.w)/volume, energy/volume, stress
