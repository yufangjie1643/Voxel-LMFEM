"""Independent assembled CPU check of the final material/FFT/PCG combination."""
import argparse
import chain_study as cs
import closed_study as ds


def validate(kind):
    b = cs.s.load(kind)
    b.TOL = 1e-10
    torch, np = b.torch, b.np
    cs.a.native.extension()
    data, geometry, material, digest = b.inputs(8, "strong")
    original = b.build(data, material, "direct", "mixed", b.Stages())
    original._validation_kind = kind
    n = original.mesh.n_elem
    full = torch.empty((1, n, 6, 6), device="cuda", dtype=torch.float64)
    for k, (i, j) in enumerate(b.PAIRS):
        full[..., i, j] = full[..., j, i] = original.C[..., k]
    matrix, cpu_rhs, cpu_u = cs.s.f.previous.cpu_reference(b, original, full, original._modal_voxel)
    ds.install(b, kind)
    cs.a.configure(b, kind, "compact_combined" if kind == "hg" else "combined")
    op = b.build(data, material, "direct", "mixed", b.Stages())
    rows, columns, trusted = [], [], []
    for j, macro in enumerate(np.eye(6)):
        rhs = op.project_zero_mean(op.compute_rhs(macro))
        u, report = b.solve(op, rhs)
        values = u[0].cpu().numpy()
        true = float(np.linalg.norm(cpu_rhs[:, j]-matrix@values) / np.linalg.norm(cpu_rhs[:, j]))
        assert true <= 1.01e-10
        direct = torch.as_tensor(cpu_u[:, :, j].ravel()[None].copy(), device="cuda")
        columns.append(op.compute_avg_stress(u, macro)[0])
        trusted.append(original.compute_avg_stress(direct, macro)[0])
        # A previously converged nonzero initial guess must remain valid.
        ux, count, _ = b.solve_pcg_batched(op, rhs, x0=u, tol=1e-8, max_iter=100,
                                          update_mode="compiled", convergence_check_stride=4)
        ux = op.project_zero_mean(ux)
        initial_true = float(torch.linalg.vector_norm(rhs-op.apply(ux))/torch.linalg.vector_norm(rhs))
        assert initial_true <= 1e-8
        rows.append(dict(load=j, report=report, independent_true_residual=true,
                         displacement_vs_direct=cs.s.f.previous.relative(values.reshape(n, 3), cpu_u[:, :, j]),
                         converged_initial_guess_true_residual=initial_true,
                         converged_initial_guess_iterations=count))
    ch, reference = np.column_stack(columns), np.column_stack(trusted)
    error = cs.s.f.previous.relative(ch, reference)
    assert error < 1e-8
    cs.save(cs.OUT/f"closed_validate_{kind}.json", dict(complete=True, kind=kind, grid=8,
        input_hash=digest, CH=ch.tolist(), CPU_direct_CH=reference.tolist(),
        CH_relative_to_CPU=error, calls=rows, reference_design=b._reference_design,
        provenance=cs.provenance(b)))
    print(kind, error, max(r["independent_true_residual"] for r in rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("hg", "fi"), required=True)
    validate(p.parse_args().kind)
