"""Direct ray sampling in (Sigma11,Sigma22,Sigma12); same frozen failure model."""
import argparse
from failure import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    gpu_guard()
    torch.set_num_threads(1)
    data = read_dataset(args.data, [0, 42, 63])
    # Latitude/longitude surface grid; duplicate seam and pole vertices are
    # retained to export a directly reproducible rectangular plotting mesh.
    theta = np.linspace(0, np.pi, 65)
    phi = np.linspace(0, 2*np.pi, 129)
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    d = np.zeros((tt.size, 6))
    d[:, 0] = (np.sin(tt)*np.cos(pp)).ravel()
    d[:, 1] = (np.sin(tt)*np.sin(pp)).ravel()
    d[:, 5] = np.cos(tt).ravel()
    d2 = directions(360)
    test_ids = [152, 1611, 2997, 4268, 5701, 7169]
    rows = []
    for i in range(3):
        gpu_guard()
        solver = build_batch(data, i, i+1, FAILURE)
        op = solver.op
        R, mask = rotation_fields(data, i, i+1, op.device)
        ch, S, reports = local_stress_basis(op, R)
        onset, mode = [], []
        for lo in range(0, len(d), 256):
            gpu_guard()
            a, b, _ = envelope(S, mask, d[lo:lo+256])
            onset.extend(a[0].tolist()); mode.extend(b[0].tolist())
        old, oldmode, _ = envelope(S, mask, d2)
        # Independent CPU criterion applied to newly sampled spatial rays.
        basis_cpu = S[0].cpu().numpy()
        mask_cpu = mask[0].cpu().numpy()
        cpu = []
        for j in test_ids:
            stress = np.einsum('nij,j->ni', basis_cpu, d[j])
            cpu.append(float(1/numpy_inverse(stress, mask_cpu).max()))
        err = float(np.max(np.abs(np.asarray(onset)[test_ids]/cpu-1)))
        assert err < 1e-10, err
        rows.append(dict(index=int(data['indices'][i]), sample_id=str(data['sample_ids'][i]),
            yarn_fraction=float(mask.double().mean()), C_H_raw=ch[0].tolist(),
            onset_MPa=onset, controlling_mode=mode, cross_sections_MPa=old[0].tolist(),
            true_residuals=[r['true_residuals'][0] for r in reports],
            solver_reports=reports, cpu_test_ray_indices=test_ids, cpu_onset_MPa=cpu,
            cpu_criterion_max_relative_difference=err))
        save_json(args.out, dict(status='running',results=rows))
        print('SURFACE', rows[-1]['index'], 'rays',len(d),'CPU error',err,flush=True)
        del solver, op, R, mask, S, basis_cpu
        torch.cuda.empty_cache()
    here = Path(__file__).parent
    save_json(args.out, dict(status='passed', material=FAILURE, provenance=provenance(),
        dataset_sha256=data['sha256'], stress_components=['11','22','12'],
        zero_stress_components=['33','23','13'], theta=theta.tolist(), phi=phi.tolist(),
        directions=d.tolist(), cross_section_directions=d2.tolist(), results=rows,
        driver_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest()
                       for n in ['run_failure_3d.py','failure.py','physics.py','failure.cu','failure.cpp']},
        note='Direct three-dimensional ray evaluation from six FP64 basis solves per cell; not reconstructed from planar cuts. No performance claim for this visualization run.'))


if __name__ == '__main__':
    main()
