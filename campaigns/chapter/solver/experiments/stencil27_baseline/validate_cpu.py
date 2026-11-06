"""CPU checks of full and symmetric stencils against independent FI assembly."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch


def main():
    from project_paths import ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'src/runtime/fi/source'))
    sys.path.insert(0, str(ROOT/'src/runtime/fi'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import spatial_adapter as spatial
    from adapter import full_assembly
    from femlib.geometry import LOCAL_OFFSET, make_B_matrices
    from scipy import sparse
    torch.set_num_threads(1)
    assembly27, source = full_assembly(spatial)
    rng = np.random.default_rng(20261023)
    records = []
    for grid in [(2, 3, 4), (5, 6, 7)]:
        nz, ny, nx = grid
        n = nz*ny*nx
        voxel = (.3, .5, .7)
        B, weight = make_B_matrices(*voxel, element_type='C3D8')
        a = rng.standard_normal((2, n, 6, 6))
        full = np.swapaxes(a, -1, -2)@a + 2*np.eye(6)
        packed = torch.tensor(np.stack([full[..., i, j] for i in range(6) for j in range(i, 6)], -1))
        def operator():
            return SimpleNamespace(element_type='c3d8', _material_layout='packed_element_major',
                B=2, mesh=SimpleNamespace(Nz=nz, Ny=ny, Nx=nx), C=packed,
                device='cpu', dtype=torch.float64)
        spatial.VALIDATION_SLAB_LIMIT = 2
        # full_assembly captured the namespace before setting this limit.
        assembly27, source = full_assembly(spatial)
        op14, op27 = operator(), operator()
        spatial.slab_assembly(op14, packed, B, weight)
        assembly27(op27, packed, B, weight)
        s14, s27 = op14.S_flat.numpy(), op27.S_flat.numpy()
        reconstructed = np.zeros_like(s27)
        reconstructed[:, :14] = s14
        for tap in range(14, 27):
            dz, rem = divmod(tap, 9)
            dy, dx = divmod(rem, 3)
            shifted = np.roll(s14[:, 26-tap].reshape(2, 3, 3, nz, ny, nx),
                              (1-dz, 1-dy, 1-dx), axis=(3, 4, 5))
            reconstructed[:, tap] = shifted.swapaxes(1, 2).reshape(2, 9, nz, ny, nx)
        x = rng.standard_normal((2, n, 3))
        y = np.zeros_like(x)
        for tap in range(27):
            dz, rem = divmod(tap, 9)
            dy, dx = divmod(rem, 3)
            neighbor = np.roll(x.reshape(2, nz, ny, nx, 3),
                               (1-dz, 1-dy, 1-dx), axis=(1, 2, 3))
            blocks = s27[:, tap].reshape(2, 3, 3, nz, ny, nx)
            y += np.einsum('bpqzyx,bzyxq->bzyxp', blocks, neighbor).reshape(2, n, 3)
        ref = []
        for batch in range(2):
            rows, cols, values = [], [], []
            for element in range(n):
                z, rem = divmod(element, ny*nx)
                yy, xx = divmod(rem, nx)
                nodes = [((z+oz)%nz*ny+(yy+oy)%ny)*nx+(xx+ox)%nx for ox, oy, oz in LOCAL_OFFSET]
                dofs = np.array([3*node+d for node in nodes for d in range(3)])
                ke = sum(weight*point.T@full[batch, element]@point for point in B)
                rows.extend(np.repeat(dofs, 24)); cols.extend(np.tile(dofs, 24)); values.extend(ke.ravel())
            matrix = sparse.coo_matrix((values, (rows, cols)), shape=(3*n, 3*n)).tocsr()
            ref.append((matrix@x[batch].ravel()).reshape(n, 3))
        ref = np.asarray(ref)
        errors = dict(full27_vs_independent_CSR=float(np.linalg.norm(y-ref)/np.linalg.norm(ref)),
                      full27_vs_symmetric14=float(np.linalg.norm(s27-reconstructed)/np.linalg.norm(s27)))
        assert max(errors.values()) < 1e-12, errors
        records.append(dict(grid=grid, batches=2, slab_depth=op27._assembly_slab_depth, errors=errors))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(passed=True, setting='CPU FP64, heterogeneous random SPD anisotropy, unequal voxel edges, depth-2 slabs, periodic boundaries',
                  checks=records, generated_assembly_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  spatial_source_sha256=hashlib.sha256(Path(spatial.__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
