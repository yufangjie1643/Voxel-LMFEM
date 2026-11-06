"""Run CPU storage/FI tests; writes a traceable JSON record, no GPU timing.

Usage: python verify.py
Constitutive states/strengths are loaded from the existing progressive study.
Continuous damage fields are prescribed algebraic fixtures, not a calibrated law.
"""
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from material_state import SharedStateMaterial, ReferenceFI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
from project_paths import source_path, ROOT as PACKAGE_ROOT
OLD = source_path("multiscale/progressive")
OUT = PACKAGE_ROOT / "validation/progressive_storage"


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def expand_reference(phase, states, history, mode):
    """Full material tensors only in the independent validation path."""
    if history.shape[1] == 1:
        history = np.repeat(history, 8, axis=1)
    if mode == "discrete":
        return states[phase[:, None], history.astype(int)]
    return (1-history[..., None, None])*states[phase, 0][:, None]


def assemble_reference(op, C):
    K = np.zeros((op.ndof, op.ndof))
    rhs = np.zeros((op.ndof, 6))
    for e, dofs in enumerate(op.dofs):
        Ke = np.einsum("qia,qij,qjb,q->ab", op.B, C[e], op.B, op.w)
        K[np.ix_(dofs, dofs)] += Ke
        rhs[dofs] -= np.einsum("qia,qij,q->aj", op.B, C[e], op.w)
    return K, rhs


def reference_fields(op, C, u, E):
    eps = np.einsum("qia,ea->eqi", op.B, u[op.dofs]) + E
    stress = np.einsum("eqij,eqj->eqi", C, eps)
    volume = len(op.dofs)*op.w.sum()
    avg = np.einsum("eqi,q->i", stress, op.w)/volume
    energy = .5*np.einsum("eqi,eqi,q->", eps, stress, op.w)/volume
    return avg, energy, stress


def solve(op, b, K):
    # Pin one periodic node to remove the three rigid translations.
    free = np.arange(3, op.ndof)
    def action(v):
        u = np.zeros(op.ndof); u[free] = v
        return op.apply(u)[free]
    A = LinearOperator((len(free), len(free)), matvec=action, dtype=np.float64)
    M = LinearOperator(A.shape, matvec=lambda v: v/np.diag(K)[free], dtype=np.float64)
    with op.material.frozen():
        x, info = cg(A, b[free], M=M, rtol=1e-11, atol=1e-13, maxiter=500)
    assert info == 0, info
    u = np.zeros(op.ndof); u[free] = x
    return u


def frozen_case(name, phase, states, history, mode):
    material = SharedStateMaterial(phase, states, mode=mode, points=history.shape[1])
    material.begin_trial(); material.update(np.arange(len(phase)), history)
    op = ReferenceFI(material, (3, 3, 3), h=(.8, 1.1, 1.3))
    C = expand_reference(phase, states, history, mode)
    K, bs = assemble_reference(op, C)
    rng = np.random.default_rng(429)
    u = rng.normal(size=op.ndof)
    row = dict(name=name, storage_bytes=material.storage_bytes(),
               active_elements=len(material.active_elements(0)),
               Ku_relative=relative(op.apply(u), K@u), loads=[])
    assert row["Ku_relative"] < 2e-13
    translation = np.tile([1., -.3, .9], len(phase))
    assert np.linalg.norm(op.apply(translation)) < 1e-8*np.linalg.norm(K)
    for j in range(6):
        E = np.eye(6)[j]
        b = op.rhs(E)
        rhs_error = np.linalg.norm(b-bs[:, j])/max(np.linalg.norm(K), 1.)
        assert rhs_error < 2e-13
        u = solve(op, b, K)
        ref = np.zeros(op.ndof); ref[3:] = np.linalg.solve(K[3:, 3:], bs[3:, j])
        avg, energy, _ = op.fields(u, E)
        avg_ref, energy_ref, _ = reference_fields(op, C, ref, E)
        residual = np.linalg.norm(K@u-bs[:, j])/max(np.linalg.norm(bs[:, j]), 1e-20)
        stress_error = relative(avg, avg_ref)
        assert residual < 1e-8 and stress_error < 1e-9
        row["loads"].append(dict(true_residual=float(residual), stress_relative=stress_error,
                                 displacement_relative=relative(u, ref),
                                 energy_relative=float(abs(energy-energy_ref)/abs(energy_ref))))
    material.commit(equilibrated=True)
    # A trial must not damage the shared catalog or committed history on rejection.
    before = material.state.copy(); catalog = material.catalog.copy()
    material.begin_trial()
    next_value = np.ones_like(before) if mode == "discrete" else before+.05*(1-before)
    material.update(np.arange(len(phase)), next_value)
    locked = False
    with material.frozen():
        try: material.rollback()
        except RuntimeError: locked = True
    assert locked
    rejected = False
    try: material.commit(equilibrated=False)
    except RuntimeError: rejected = True
    assert rejected
    material.rollback()
    assert np.array_equal(before, material.state) and np.array_equal(catalog, material.catalog)
    material.begin_trial()
    try: material.update(np.arange(len(phase)), np.zeros_like(before))
    except ValueError: pass
    else: raise AssertionError("Healing accepted")
    material.rollback()
    row["rollback_lock_and_irreversibility"] = "passed"
    return row


def progressive_path(law, params, states):
    # Verification geometry, not the previously benchmarked woven/hexagonal RVE.
    phase = np.array([0, 0, 0, 1, 0, 1, 1, 0])
    m = SharedStateMaterial(phase, states)
    op = ReferenceFI(m, (2, 2, 2))
    result = []
    for amplitude in (0., .002, .005, .01, .02, .05, .10, 0., .005):
        m.begin_trial()
        E = np.array([0., amplitude, 0., 0., 0., 0.])
        for correction in range(len(phase)+1):
            C = expand_reference(phase, states, m.state, "discrete")
            K, bs = assemble_reference(op, C)
            u = solve(op, op.rhs(E), K)
            ref = np.zeros(op.ndof); ref[3:] = np.linalg.solve(K[3:, 3:], (bs@E)[3:])
            avg, energy, stress = op.fields(u, E)
            avg_ref, _, stress_ref = reference_fields(op, C, ref, E)
            f = law.indices(stress, phase[:, None], params).max(axis=1) >= 1-1e-10
            f_ref = law.indices(stress_ref, phase[:, None], params).max(axis=1) >= 1-1e-10
            assert np.array_equal(f, f_ref)
            new = f & (m.state[:, 0] == 0)
            assert np.linalg.norm(avg-avg_ref) < 1e-8*max(np.linalg.norm(avg_ref), 1.)
            force = bs@E
            residual = np.linalg.norm(K@u-force)/max(np.linalg.norm(force), 1e-20)
            hm = abs(2*energy-E@avg)/max(abs(E@avg), 1e-20)
            assert residual < 1e-8 and hm < 1e-8
            if not new.any():
                m.commit(equilibrated=True)
                result.append(dict(strain22=amplitude, failed=int(m.state.sum()),
                                   corrections=correction, stress=avg.tolist(), energy=float(energy),
                                   stress_error=float(np.linalg.norm(avg-avg_ref)),
                                   true_residual=float(residual), hill_mandel_error=float(hm),
                                   displacement_norm=float(np.linalg.norm(u))))
                break
            ids = np.flatnonzero(new)
            m.update(ids, np.ones((len(ids), 1), dtype=np.uint8))
        else: raise AssertionError("Damage equilibrium did not converge")
    assert result[-1]["failed"] > 0
    assert max(r["displacement_norm"] for r in result) > 1e-6
    assert result[-1]["failed"] == result[-2]["failed"] == result[-3]["failed"]
    return result


def main():
    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location("existing_damage_law", OLD/"material.py")
    law = importlib.util.module_from_spec(spec); spec.loader.exec_module(law)
    params = law.load_material(); states = law.stiffness_states(params)
    rng = np.random.default_rng(431)
    phase = rng.integers(0, 2, 27)
    rows = []
    for mode in ("discrete", "scalar"):
        for points in (1, 8):
            history = rng.integers(0, 2, (27, points)) if mode == "discrete" else rng.uniform(0, .85, (27, points))
            rows.append(frozen_case(f"{mode}_{points}point_history", phase,
                                   states if mode == "discrete" else states[:, :1], history, mode))
    history = np.zeros((27, 1), dtype=np.uint8); history[[0, 8, 20]] = 1
    rows.append(frozen_case("damaged_background_only", np.zeros(27, dtype=int), states, history, "discrete"))
    # Counterexample: averaging different QP stiffnesses is not exact FI.
    m = SharedStateMaterial(phase, states[:, :1], mode="scalar", points=8)
    op = ReferenceFI(m, (3, 3, 3))
    d = np.array([0., .1, .2, .3, .5, .6, .8, .9])
    Cq = (1-d[:, None, None])*states[0, 0]
    Kq = np.einsum("qia,qij,qjb,q->ab", op.B, Cq, op.B, op.w)
    Kmean = np.einsum("qia,ij,qjb,q->ab", op.B, Cq.mean(axis=0), op.B, op.w)
    averaging_error = relative(Kmean, Kq)
    assert averaging_error > 1e-3
    path = progressive_path(law, params, states)
    N = 192**3
    memory = dict(N=N, unit="MiB", scope="logical arrays, not GPU/NVML measurements",
                  dense_C21_per_element=168*N/2**20,
                  dense_C21_per_qp=168*8*N/2**20,
                  discrete_catalog_four_C21_bytes=4*168,
                  int32_phase_map=4*N/2**20,
                  uint8_history_per_element=N/2**20,
                  scalar_FP64_history_per_element=8*N/2**20,
                  six_FP64_history_per_element=6*8*N/2**20,
                  uint8_history_eight_qp=8*N/2**20,
                  scalar_FP64_history_eight_qp=8*8*N/2**20,
                  six_FP64_history_eight_qp=6*8*8*N/2**20,
                  discrete_phase_and_double_history=6*N/2**20+4*168/2**20)
    sources = [HERE/"material_state.py", HERE/"verify.py", OLD/"material.py", OLD/"material.json"]
    output = dict(status="passed", platform=platform.platform(), numpy=np.__version__,
                  scope="CPU prototype; 5 frozen 3^3 FI cases x 6 loads, and 2^3 damage path; not GPU integration",
                  frozen_cases=rows, prescribed_qp_averaging_error=averaging_error,
                  progressive_path=path, memory_192=memory,
                  elapsed_seconds=time.perf_counter()-start,
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"validation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(dict(status=output["status"], cases=len(rows),
                         max_Ku_relative=max(r["Ku_relative"] for r in rows),
                         max_true_residual=max(x["true_residual"] for r in rows for x in r["loads"]),
                         max_stress_relative=max(x["stress_relative"] for r in rows for x in r["loads"]),
                         averaging_error=averaging_error, progressive_path=path, memory_192=memory,
                         output=str(OUT/"validation.json")), ensure_ascii=False))


if __name__ == "__main__":
    main()
